"""
FTW Predictor V33+ (pattern restored + adaptive model)
Python 3.10+
Deps: aiohttp

Environment variables:
  TELEGRAM_TOKEN    required, rotated bot token
  TELEGRAM_CHAT     required
  HISTORY_URLS      optional, comma-separated URLs
  DB_PATH           optional, default predictor_v33_plus.db
  PORT              optional, default 8080
  FETCH_EVERY       optional, default 3.0
  MIN_HISTORY       optional, default 180
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import json
import logging
import math
import os
import signal
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

DEFAULT_HISTORY_URL = (
    "https://sky-predictor-1012593186417.asia-southeast1.run.app/"
    "api/wingo-history-1m-1000"
)


# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_v33_plus.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3.0"))
    timeout: float = 18.0
    max_history: int = 3500
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (V33+ PATTERN)"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN is required")
        if not self.chat:
            raise ValueError("TELEGRAM_CHAT is required")
        if not self.urls:
            raise ValueError("At least one HISTORY_URL is required")
        if self.fetch_every <= 0:
            raise ValueError("FETCH_EVERY must be > 0")
        if self.port < 1 or self.port > 65535:
            raise ValueError("PORT must be valid")


CFG = Config()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v33_plus")


# ============================================================
# HELPERS
# ============================================================

def clip(value: float, low: float = 0.02, high: float = 0.98) -> float:
    return max(low, min(high, float(value)))


def sigmoid(z: float) -> float:
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float = 0.02) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def escape(value: Any) -> str:
    return html.escape(str(value))


def numeric(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text.isascii() or not text.isdigit():
        return None
    try:
        return int(text)
    except ValueError:
        return None


def weighted_probability(obs: list[tuple[int, float]]) -> tuple[float, float]:
    if not obs:
        return 0.5, 0.0
    sw = sum(w for _, w in obs)
    sw2 = sum(w * w for _, w in obs)
    eff = (sw * sw / sw2) if sw2 else 0.0
    p = (4.0 + sum(y * w for y, w in obs)) / (8.0 + sw)
    return clip(p), eff


def family_run(seq: list[int]) -> list[tuple[int, int]]:
    if not seq:
        return []
    out: list[tuple[int, int]] = []
    cur = seq[0]
    count = 1
    for x in seq[1:]:
        if x == cur:
            count += 1
        else:
            out.append((cur, count))
            cur = x
            count = 1
    out.append((cur, count))
    return out


def family_signature(seq: list[int]) -> tuple[tuple[int, int], ...]:
    return tuple((int(v), cnt) for v, cnt in family_run(seq))


def entropy_bits(seq: list[int]) -> float:
    if not seq:
        return 0.0
    c = Counter(seq)
    total = len(seq)
    e = 0.0
    for freq in c.values():
        p = freq / total
        e -= p * math.log2(p)
    return e


def get_regime(seq: list[int]) -> str:
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"

    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    flip_rate = flips / max(1, len(tail) - 1)

    run = 1
    for value in reversed(tail[:-1]):
        if value != tail[-1]:
            break
        run += 1

    if run >= 5:
        return "RUN"
    if flip_rate >= 0.72:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"


def trend_features(seq: list[int]) -> list[float]:
    if not seq:
        return [0.0] * 20

    def mean(k: int) -> float:
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5

    def ema(k: int) -> float:
        if not seq:
            return 0.5
        alpha = 2.0 / (k + 1)
        v = float(seq[0])
        for x in seq[1:]:
            v = alpha * x + (1.0 - alpha) * v
        return v

    def std(k: int) -> float:
        arr = seq[-k:]
        if len(arr) < 2:
            return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

    tail = seq[-24:]
    alt = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    run_score = (2 * seq[-1] - 1) * min(len(list(filter(lambda x: x == seq[-1], seq[-12:]))), 12) / 12.0
    macd = (ema(12) - ema(26)) * 2.0

    return [
        1.0,
        2 * mean(3) - 1,
        2 * mean(5) - 1,
        2 * mean(10) - 1,
        2 * mean(24) - 1,
        2 * mean(48) - 1,
        2 * mean(96) - 1,
        mean(3) - mean(10),
        mean(5) - mean(24),
        mean(10) - mean(48),
        run_score,
        2 * seq[-1] - 1,
        (2 * alt - 1) * (2 * seq[-1] - 1),
        macd,
        2 * std(10) - 1,
        2 * std(24) - 1,
        mean(3) - mean(12),
        mean(8) - mean(24),
        mean(12) - mean(48),
        mean(6) - mean(24),
    ]


def recency_weight(index: int, n: int, half_life: float = 330.0) -> float:
    age = max(0, n - 1 - index)
    return 0.5 ** (age / half_life)


def parse_history(payload: Any) -> list[dict[str, Any]]:
    obj = payload
    for _ in range(8):
        if isinstance(obj, list):
            break
        if not isinstance(obj, dict):
            return []

        next_obj = None
        for key in ("data", "list", "result", "items", "history", "records"):
            candidate = obj.get(key)
            if isinstance(candidate, (list, dict)):
                next_obj = candidate
                break
        if next_obj is None:
            return []
        obj = next_obj

    if not isinstance(obj, list):
        return []

    records: dict[str, dict[str, Any]] = {}

    for item in obj:
        if not isinstance(item, dict):
            continue

        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))

        if issue is None or number is None or not 0 <= number <= 9:
            continue

        key = str(issue)
        row = {"issue": key, "num": number, "size": int(number >= 5)}

        existing = records.get(key)
        if existing and existing["num"] != number:
            raise ValueError(f"Conflicting duplicate issue: {key}")

        records[key] = row

    return sorted(records.values(), key=lambda x: int(x["issue"]))


# ============================================================
# PATTERN TRACKER
# ============================================================

class QuantumTracker:
    def __init__(self, min_len: int = 3, max_len: int = 8, min_support: int = 12):
        self.min_len = min_len
        self.max_len = max_len
        self.min_support = min_support

    def match_tail_patterns(self, seq: list[int]) -> list[dict[str, Any]]:
        n = len(seq)
        if n < 18:
            return []

        results: list[dict[str, Any]] = []
        for length in range(self.max_len, self.min_len - 1, -1):
            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs: list[tuple[int, float]] = []

            for i in range(n - length):
                if family_signature(seq[i : i + length]) == sig:
                    w = 0.5 ** ((n - 1 - (i + length)) / 420.0)
                    obs.append((seq[i + length], w))

            if len(obs) < self.min_support:
                continue

            p, eff_n = weighted_probability(obs)
            score = abs(p - 0.5) * math.sqrt(max(1.0, eff_n))
            if score > 0.0:
                results.append({
                    "sig": sig,
                    "p": p,
                    "support": len(obs),
                    "eff_n": eff_n,
                    "score": score,
                })
            break

        return results


# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, path: str):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA temp_store=MEMORY")
        self.c.execute("PRAGMA busy_timeout=20000")
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS history(
                issue TEXT PRIMARY KEY,
                num INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions(
                issue TEXT PRIMARY KEY,
                created REAL NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                actual INTEGER,
                won_number INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS models(
                key TEXT PRIMARY KEY,
                n REAL NOT NULL,
                loss REAL NOT NULL,
                inverse_loss REAL NOT NULL,
                failures INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS meta(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                issue TEXT,
                text TEXT NOT NULL,
                created REAL NOT NULL,
                expires REAL,
                sent INTEGER NOT NULL DEFAULT 0
            );
        """)

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            if self.c.in_transaction:
                self.c.execute("ROLLBACK")
            raise
        else:
            if self.c.in_transaction:
                self.c.execute("COMMIT")

    def get(self, key: str, default: Any = None) -> Any:
        row = self.c.execute(
            "SELECT value FROM meta WHERE key=?",
            (key,),
        ).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def memory(self, key: str) -> dict[str, Any]:
        row = self.c.execute(
            "SELECT * FROM models WHERE key=?",
            (key,),
        ).fetchone()
        return dict(row) if row else {
            "n": 0.0,
            "loss": 0.0,
            "inverse_loss": 0.0,
            "failures": 0,
        }

    def learn_model(self, key: str, probability: float, actual: int) -> None:
        m = self.memory(key)
        n = CFG.fetch_every * 0.985 * m["n"] + 1.0
        loss = 0.985 * m["loss"] + (probability - actual) ** 2
        inv = 0.985 * m["inverse_loss"] + (1.0 - probability - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        failures = m["failures"] + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n, loss=excluded.loss, inverse_loss=excluded.inverse_loss, failures=excluded.failures",
            (key, n, loss, inv, failures),
        )

    def adapt_model(self, key: str, probability: float) -> tuple[float, float]:
        m = self.memory(key)
        n = m["n"]
        if n < 1:
            return probability, 0.7

        brier = m["loss"] / n
        inv = m["inverse_loss"] / n
        failures = m["failures"]

        flipped = False
        if failures >= 2 and (inv + 0.01 < brier):
            flipped = True
        elif failures >= 3 and brier > 0.26:
            flipped = True

        eff_loss = inv if flipped else brier
        p = (1.0 - probability) if flipped else probability

        reliability = clip((0.25 - eff_loss) / 0.12, 0.0, 1.0)
        shrink = (1.0 - min(n / 50.0, 1.0)) * 0.8 + min(n / 50.0, 1.0) * reliability
        p = 0.5 + (p - 0.5) * shrink

        weight = clip(math.exp(-8 * (eff_loss - 0.22)), 0.15, 2.0)
        if failures >= 2:
            weight *= 0.30
        if failures >= 4:
            weight *= 0.10

        return clip(p), weight

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for r in records:
                old = self.c.execute(
                    "SELECT num FROM history WHERE issue=?",
                    (r["issue"],),
                ).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute(
                    "INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)",
                    (r["issue"], r["num"]),
                )

            self.c.execute(
                """
                DELETE FROM history
                WHERE issue NOT IN (
                    SELECT issue FROM history
                    ORDER BY CAST(issue AS INTEGER) DESC
                    LIMIT ?
                )
                """,
                (CFG.max_history,),
            )

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute(
            """
            SELECT issue,num
            FROM history
            ORDER BY CAST(issue AS INTEGER)
            """
        ).fetchall()
        return [
            {"issue": row["issue"], "num": int(row["num"]), "size": int(int(row["num"]) >= 5)}
            for row in rows
        ]

    def get_loss_streak(self) -> int:
        rows = self.c.execute(
            """
            SELECT status
            FROM predictions
            WHERE status IN ('WIN','LOSS')
            ORDER BY created DESC
            LIMIT 12
            """
        ).fetchall()
        streak = 0
        for row in rows:
            if row["status"] == "LOSS":
                streak += 1
            else:
                break
        return streak

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute(
            """
            SELECT issue,payload
            FROM predictions
            WHERE status='PENDING'
            ORDER BY CAST(issue AS INTEGER)
            """
        ).fetchall()

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute(
            "INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires),
        )

    def health_stats(self) -> dict[str, Any]:
        settled = self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        wins = self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0]
        total_history = self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]

        return {
            "history": total_history,
            "settled": int(settled),
            "wins": int(wins),
            "win_rate": round((wins / settled * 100.0) if settled else 0.0, 2),
        }

    def close(self) -> None:
        self.c.close()


# ============================================================
# EXACT NUMBER MODELS
# ============================================================

def recent_frequency(nums: list[int], window: int = 200) -> list[float]:
    if not nums:
        return [0.1] * 10
    recent = nums[-window:]
    counts = Counter(recent)
    total = sum(counts.values())
    return [counts.get(i, 0) / total if total > 0 else 0.1 for i in range(10)]


def markov_predict(nums: list[int], order: int = 2) -> list[float]:
    if not nums:
        return [0.1] * 10
    if len(nums) <= order:
        return recent_frequency(nums)

    context = tuple(nums[-order:])
    counts = [0.0] * 10
    total = 0.0
    for i in range(order, len(nums)):
        if tuple(nums[i - order:i]) == context:
            counts[nums[i]] += 1.0
            total += 1.0
    if total <= 0:
        return recent_frequency(nums)
    return [counts[k] / total for k in range(10)]


def exact_number_model(history: list[dict[str, Any]]) -> list[float]:
    nums = [int(r["num"]) for r in history]
    if not nums:
        return [0.1] * 10

    d1 = recent_frequency(nums, 80)
    d2 = recent_frequency(nums, 200)
    d3 = markov_predict(nums, 1)
    d4 = markov_predict(nums, 2)
    d5 = markov_predict(nums, 3)

    final = [0.0] * 10
    for i in range(10):
        final[i] = 0.30 * d1[i] + 0.25 * d2[i] + 0.20 * d3[i] + 0.15 * d4[i] + 0.10 * d5[i]

    total = sum(final)
    return [x / total for x in final] if total > 0 else [0.1] * 10


# ============================================================
# WF900 CALIBRATION
# ============================================================

def walk_forward_900(history: list[dict[str, Any]], p_raw: float) -> tuple[float, float]:
    if len(history) < 100:
        return p_raw, 0.0

    window = history[-900:]
    sizes = [row["size"] for row in window]
    macro_big = sum(sizes) / len(sizes) if sizes else 0.5
    mid_big = sum(sizes[-150:]) / len(sizes[-150:]) if sizes else 0.5
    micro_big = sum(sizes[-30:]) / len(sizes[-30:]) if sizes else 0.5

    entropy = entropy_bits([row["num"] for row in window[-50:]]) / 3.32

    wf = 0.0
    wf += (micro_big - 0.5) * 0.45
    wf += (mid_big - 0.5) * 0.35
    wf += (macro_big - 0.5) * 0.20
    wf += (entropy - 0.5) * 0.30

    wf = clip(wf, -0.35, 0.35)
    p_adj = clip(p_raw + wf)
    return p_adj, wf


# ============================================================
# ENGINE
# ============================================================

EXPERT_NAMES = [
    "patterns", "markov1", "markov2", "markov3",
    "run", "regime", "base", "trend", "momentum",
    "reversal", "volatility", "entropy"
]


class Engine:
    def __init__(self, db: Database):
        self.db = db

    def mine_patterns(self, seq: list[int]) -> list[tuple[float, float, int, float, str]]:
        if len(seq) < 18:
            return []
        tracker = QuantumTracker(min_len=3, max_len=8, min_support=CFG.min_support)
        pat = tracker.match_tail_patterns(seq)
        out: list[tuple[float, float, int, float, str]] = []

        for item in pat[:6]:
            key = "q:" + "-".join(f"{'B' if v else 'S'}{c}" for v, c in item["sig"])
            mem = self.db.memory(key)
            if mem["failures"] >= 4 and (mem["loss"] / max(1.0, mem["n"]) > 0.28):
                continue
            out.append((item["score"], item["p"], item["support"], item["eff_n"], key))

        return out

    def mine_experts(self, seq: list[int]) -> tuple[dict[str, float], list[tuple[float, float, int, float, str]], list[float], str]:
        n = len(seq)
        experts: dict[str, float] = {}

        def weight(idx: int) -> float:
            return 0.5 ** ((n - 1 - idx) / 330.0)

        patterns = self.mine_patterns(seq)
        if patterns:
            candidates: list[tuple[float, float]] = []
            for _, p, _, eff_n, key in patterns:
                adjusted_p, w = self.db.adapt_model(key, p)
                candidates.append((adjusted_p, w * math.sqrt(max(1.0, eff_n))))
            if candidates:
                num = sum(p * w for p, w in candidates)
                den = sum(w for _, w in candidates)
                if den > 0:
                    experts["patterns"] = clip(num / den)

        for order in (1, 2, 3):
            suffix = tuple(seq[-order:])
            obs: list[tuple[int, float]] = []
            for i in range(order, n - order):
                if tuple(seq[i - order:i]) == suffix:
                    obs.append((seq[i], weight(i)))
            if len(obs) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(obs)[0]

        if n:
            run_at = [1] * n
            for i in range(1, n):
                run_at[i] = run_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
            cur_run = min(run_at[-1], 8)
            last_val = seq[-1]
            run_obs: list[tuple[int, float]] = []
            for i in range(24, n):
                if seq[i-1] == last_val and min(run_at[i-1], 8) == cur_run:
                    run_obs.append((seq[i], weight(i)))
            if len(run_obs) >= CFG.min_support:
                experts["run"] = weighted_probability(run_obs)[0]

        regime = get_regime(seq)
        regime_obs: list[tuple[int, float]] = []
        for i in range(24, n):
            if get_regime(seq[max(0, i - 96):i]) == regime:
                regime_obs.append((seq[i], weight(i)))
        if len(regime_obs) >= CFG.min_support:
            experts["regime"] = weighted_probability(regime_obs)[0]

        experts["base"] = weighted_probability([(y, weight(i)) for i, y in enumerate(seq)])[0]

        features = trend_features(seq)
        trend_weights = self.db.get("trend_weights")
        if not isinstance(trend_weights, list) or len(trend_weights) != len(features):
            trend_weights = [0.0] * len(features)
        experts["trend"] = sigmoid(sum(w * x for w, x in zip(trend_weights, features)))

        tail = seq[-16:]
        if len(tail) > 1:
            avg = sum(tail) / len(tail)
            vol = sum((x - avg) ** 2 for x in tail) / len(tail)
            experts["volatility"] = clip(0.5 + (0.5 - abs(0.5 - avg)) * 0.85 + vol * 0.1)
        else:
            experts["volatility"] = 0.5

        experts["entropy"] = clip(0.5 + (entropy_bits(seq[-12:]) / 3.32 - 0.5) * 0.6)

        momentum = sum(seq[-4:]) / 4.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)
        experts["reversal"] = 1.0 - experts["momentum"]

        for k, v in list(experts.items()):
            experts[k] = clip(v, 0.05, 0.95)

        return experts, patterns, features, regime

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(r["size"]) for r in history]
        loss_streak = self.db.get_loss_streak()

        experts, patterns, features, regime = self.mine_experts(seq)

        adjusted: list[tuple[float, float]] = []
        for name, prob in experts.items():
            corrected, weight = self.db.adapt_model(f"expert:{name}:{regime}", prob)
            if name == "patterns":
                weight *= 3.3
            if name == "reversal" and loss_streak >= 1:
                weight *= 3.8
            if name in ("momentum", "run", "trend") and loss_streak >= 1:
                weight *= 0.25
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5
        p_raw = clip(p_avg)

        p_raw, wf_bias = walk_forward_900(history, p_raw)

        pred_side = "BIG" if p_raw >= 0.5 else "SMALL"
        allowed = range(5, 10) if pred_side == "BIG" else range(0, 5)

        exact_dist = exact_number_model(history)
        side_adjusted = [0.0] * 10
        for k in range(10):
            if k in allowed:
                side_adjusted[k] = exact_dist[k] * 2.0
            else:
                side_adjusted[k] = exact_dist[k] * 0.1

        total = sum(side_adjusted)
        if total <= 0:
            side_adjusted = [0.1] * 10
            total = 10.0
        side_adjusted = [x / total for x in side_adjusted]

        num_pred = max(allowed, key=lambda k: side_adjusted[k])
        num_conf = side_adjusted[num_pred]

        payload = {
            "issue": issue,
            "p": p_raw,
            "side": pred_side,
            "num": num_pred,
            "num_probability": num_conf,
            "regime": regime,
            "source": "Adaptive Pattern + Expert",
            "pattern_str": self._pattern_str(patterns),
            "experts": experts,
            "features": features,
            "wf_bias": wf_bias,
            "patterns": [
                {"key": key, "p": p, "support": s}
                for _, p, s, _, key in patterns
            ],
            "exact_distribution": side_adjusted,
        }
        return payload

    def _pattern_str(self, patterns: list[tuple[float, float, int, float, str]]) -> str:
        if not patterns:
            return "None"
        _, p, _, _, key = patterns[0]
        tag = key.split(":")[-1] if ":" in key else key
        arrow = "⬆️" if p >= 0.5 else "⬇️"
        return f"<code>{tag}</code> {arrow}"

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        pending = self.db.pending_predictions()

        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None:
                continue

            payload = json.loads(row["payload"])
            actual_size = int(actual["size"])
            actual_num = int(actual["num"])

            won_side = int(float(payload["p"]) >= 0.5) == actual_size
            won_num = payload.get("num") == actual_num

            with self.db.transaction():
                self.db.c.execute(
                    """
                    UPDATE predictions
                    SET status=?, actual=?, won_number=?
                    WHERE issue=? AND status='PENDING'
                    """,
                    (
                        "WIN" if won_side else "LOSS",
                        actual_num,
                        int(won_num),
                        row["issue"],
                    ),
                )

                for name, val in payload.get("experts", {}).items():
                    self.db.learn_model(f"expert:{name}:{payload.get('regime', 'MIXED')}", float(val), actual_size)

                for pat in payload.get("patterns", []):
                    self.db.learn_model(pat["key"], float(pat["p"]), actual_size)

                if payload.get("features") and payload["experts"].get("trend"):
                    err = actual_size - payload["experts"]["trend"]
                    tw = self.db.get("trend_weights") or [0.0] * len(payload["features"])
                    self.db.set("trend_weights", [
                        clip(w + 0.06 * (err * x - 0.015 * w), -4.0, 4.0)
                        for w, x in zip(tw, payload["features"])
                    ])

                sticker = (
                    "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"
                    if won_side and won_num else
                    "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
                    if won_side else
                    "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
                )
                self.db.queue("sticker", row["issue"], sticker)


# ============================================================
# FORMATTING
# ============================================================

def format_prediction(payload: dict[str, Any]) -> str:
    prob = float(payload["p"])
    conf = max(prob, 1.0 - prob) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if payload["side"] == "BIG" else "SMALL"
    marker = "🔴" if payload["side"] == "BIG" else "🔵"

    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {marker} <b>{side}</b> → <code>{payload['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload.get('regime', 'N/A'))}\n"
        f"🧩 <b>PATTERN:</b> {payload.get('pattern_str', 'None')}\n"
        f"📈 <b>WF900:</b> {payload.get('wf_bias', 0):+.3f}\n"
        "ℹ️ Experimental estimate; not guaranteed.\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db: Database):
        self.db = db
        self.engine = Engine(db)
        self.stop = asyncio.Event()
        self.last_fetch = 0.0

    async def pause(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, seconds))
        except asyncio.TimeoutError:
            pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def one(url: str) -> list[dict[str, Any]]:
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    payload = await r.json(content_type=None)
                    return parse_history(payload)
            except Exception:
                return []

        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources:
            raise RuntimeError("HistoryUnavailable")

        merged: dict[str, dict[str, Any]] = {}
        for rows in sources:
            for row in rows:
                if row["issue"] in merged and merged[row["issue"]]["num"] != row["num"]:
                    raise ValueError("SourceConflict")
                merged[row["issue"]] = row

        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        while not self.stop.is_set():
            try:
                records = await self.fetch(session)
                self.db.ingest(records)
                history = self.db.history()
                self.last_fetch = time.time()

                self.engine.settle(history)

                latest = int(history[-1]["issue"])
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    exists = self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone()
                    if not exists:
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL,0)",
                                (issue, time.time(), "PENDING", json.dumps(pred),)
                            )
                            self.db.queue("prediction", issue, format_prediction(pred), time.time() + CFG.delivery_ttl)

                failures = 0

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.exception("Worker failure: %s", exc)
                failures += 1

            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        endpoint_msg = f"https://api.telegram.org/bot{CFG.token}/sendMessage"
        endpoint_sticker = f"https://api.telegram.org/bot{CFG.token}/sendSticker"
        backoff = 1.0

        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute(
                "SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if not row:
                await self.pause(0.5)
                continue

            delay = backoff
            try:
                url = endpoint_sticker if row["kind"] == "sticker" else endpoint_msg
                payload = {"chat_id": CFG.chat, "sticker": row["text"]} if row["kind"] == "sticker" else {
                    "chat_id": CFG.chat,
                    "text": row["text"],
                    "parse_mode": "HTML",
                }

                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)
                    if response.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue

                    code = data.get("error_code", response.status)
                    if code == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404):
                        delay = 60
            except Exception as e:
                LOG.warning("Telegram network fail: %s", e)
                delay = 10

            await self.pause(delay)
            backoff = min(60.0, backoff * 2)

    async def health(self, request: web.Request) -> web.Response:
        stats = self.db.health_stats()
        status = "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded"

        return web.json_response({
            "status": status,
            "last_fetch": self.last_fetch,
            **stats
        })


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    CFG.validate()
    db = Database(CFG.db_path)
    runtime = Runtime(db)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop.set)

    app = web.Application()
    app.router.add_get("/", runtime.health)

    runner = web.AppRunner(app)
    tasks: list[asyncio.Task] = []
    stop_task: asyncio.Task | None = None

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()

        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session), name="worker"),
                asyncio.create_task(runtime.sender(session), name="sender"),
            ]
            stop_task = asyncio.create_task(runtime.stop.wait())
            LOG.info("%s started", CFG.name)
            await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
            runtime.stop.set()

            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc:
                        raise exc

            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        runtime.stop.set()
        for task in tasks:
            task.cancel()
        if stop_task:
            stop_task.cancel()
        await asyncio.gather(*tasks, *([stop_task] if stop_task else []), return_exceptions=True)
        await runner.cleanup()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
