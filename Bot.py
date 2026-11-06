"""
FTW Predictor V50 — bug-fixed build
===================================
Fixes vs V49:
  1. Bootstrap retry (no more "predictions never start")
  2. Defensive format_prediction (no silent throws)
  3. Historical won_number repair on startup
  4. Explicit settle logging + /debug/recent endpoint
  5. Strict pred_num type checks (no bool-coerced false jackpots)
  6. Number ALWAYS shown, with honest signal-strength label
  7. Full schema migration for older DBs
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
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import aiohttp
from aiohttp import web

# ============================================================
# CONFIG
# ============================================================

DEFAULT_HISTORY_URL = (
    "https://sky-predictor-1012593186417.asia-southeast1.run.app/"
    "api/wingo-history-1m-1000"
)

DISCLOSURE = "<i>Statistical output — not a guarantee.</i>"
N_FEATURES = 20


@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_v50.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    delivery_ttl: float = 600.0
    sticker_ttl: float = 3600.0
    max_outbox_attempts: int = 3
    prediction_retention_days: int = 30
    backtest_min_samples: int = int(os.getenv("BACKTEST_MIN", "500"))
    bootstrap_retry_sec: float = float(os.getenv("BOOTSTRAP_RETRY_SEC", "60"))
    edge_warn: float = 0.01
    edge_ok: float = 0.03
    name: str = "FTW PREDICTOR (V50)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN is required")
        if not self.chat:
            raise ValueError("TELEGRAM_CHAT is required")


CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v50")


# ============================================================
# MATH
# ============================================================

def clip(v: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return max(lo, min(hi, float(v)))


def sigmoid(z: float) -> float:
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float = 0.02) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


def escape(v: Any) -> str:
    return html.escape(str(v))


def numeric(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    t = str(v).strip()
    if not t.isascii() or not t.isdigit():
        return None
    try:
        return int(t)
    except ValueError:
        return None


def as_int_in_range(v: Any, lo: int, hi: int) -> int | None:
    """Strict: returns int in [lo, hi] or None. Rejects bools and garbage."""
    if isinstance(v, bool):
        return None
    if not isinstance(v, int):
        try:
            v = int(v)
        except (TypeError, ValueError):
            return None
    return v if lo <= v <= hi else None


def as_float(v: Any, default: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ============================================================
# FEATURES (20)
# ============================================================

def extract_features(window: list[int]) -> list[float]:
    n = len(window)
    if n == 0:
        return [0.5] * N_FEATURES
    feats: list[float] = []

    for k in range(1, 6):
        feats.append(float(window[-k]) if k <= n else 0.5)

    for w in (5, 10, 20, 50):
        tail = window[-w:]
        feats.append(sum(tail) / len(tail))

    last = window[-1]
    run = 1
    for v in reversed(window[:-1]):
        if v != last:
            break
        run += 1
    feats.append(min(run, 10) / 10.0)
    feats.append(1.0 if last else -1.0)

    t10 = window[-10:]
    if len(t10) >= 2:
        flips = sum(1 for a, b in zip(t10, t10[1:]) if a != b)
        feats.append(flips / (len(t10) - 1))
    else:
        feats.append(0.5)

    t20 = window[-20:]
    if t20:
        p = sum(t20) / len(t20)
        e = -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) if 0 < p < 1 else 0.0
        feats.append(e)
    else:
        feats.append(1.0)

    r10 = sum(window[-10:]) / max(1, len(window[-10:]))
    r50 = sum(window[-50:]) / max(1, len(window[-50:]))
    feats.append(r10 - r50)

    def gap(target: int, cap: int = 20) -> float:
        for j, v in enumerate(reversed(window[-cap:])):
            if v == target:
                return j / cap
        return 1.0

    feats.append(gap(0))
    feats.append(gap(1))

    search = window[-500:]
    for order in (2, 3):
        if n >= order:
            ctx = tuple(search[-order:])
            matches = [search[i] for i in range(order, len(search))
                       if tuple(search[i - order:i]) == ctx]
            feats.append(sum(matches) / len(matches) if matches else 0.5)
        else:
            feats.append(0.5)

    if t20:
        mean = sum(t20) / len(t20)
        var = sum((x - mean) ** 2 for x in t20) / len(t20)
        feats.append(var * 4.0)
    else:
        feats.append(0.0)

    if len(t20) >= 2:
        mean = sum(t20) / len(t20)
        num = sum((t20[i] - mean) * (t20[i - 1] - mean) for i in range(1, len(t20)))
        den = sum((x - mean) ** 2 for x in t20)
        feats.append(num / den if den > 0 else 0.0)
    else:
        feats.append(0.0)

    return feats


# ============================================================
# ONLINE LOGISTIC REGRESSION
# ============================================================

class OnlineLogisticRegression:
    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4):
        self.w = [0.0] * n_features
        self.b = 0.0
        self.lr0 = lr
        self.l2 = l2
        self.n_updates = 0

    def predict_proba(self, x: list[float]) -> float:
        z = self.b + sum(wi * xi for wi, xi in zip(self.w, x))
        return sigmoid(z)

    def update(self, x: list[float], y: int, sample_weight: float = 1.0) -> None:
        p = self.predict_proba(x)
        err = (p - y) * sample_weight
        lr_t = self.lr0 / (1.0 + self.n_updates * 0.001)
        self.b -= lr_t * err
        for i in range(len(self.w)):
            self.w[i] -= lr_t * (err * x[i] + self.l2 * self.w[i])
        self.n_updates += 1

    def to_state(self) -> dict[str, Any]:
        return {"w": self.w, "b": self.b, "n": self.n_updates}

    def load_state(self, state: dict[str, Any]) -> None:
        self.w = list(state.get("w", self.w))
        self.b = float(state.get("b", self.b))
        self.n_updates = int(state.get("n", 0))


# ============================================================
# CALIBRATOR
# ============================================================

class TemperatureCalibrator:
    def __init__(self):
        self.T = 1.0
        self.n = 0
        self.sum_grad = 0.0

    def apply(self, p: float) -> float:
        if self.n < 50:
            return p
        return sigmoid(logit(p) / max(0.3, min(3.0, self.T)))

    def observe(self, p: float, y: int) -> None:
        self.n += 1
        z = logit(p)
        q = sigmoid(z / max(0.3, min(3.0, self.T)))
        self.sum_grad = 0.95 * self.sum_grad + (q - y) * (-z / (self.T ** 2 + 1e-6))
        if self.n >= 50:
            self.T = max(0.3, min(3.0, self.T + 0.02 * self.sum_grad))

    def to_state(self) -> dict[str, Any]:
        return {"T": self.T, "n": self.n, "g": self.sum_grad}

    def load_state(self, state: dict[str, Any]) -> None:
        self.T = float(state.get("T", 1.0))
        self.n = int(state.get("n", 0))
        self.sum_grad = float(state.get("g", 0.0))


# ============================================================
# PATTERN MINER
# ============================================================

class PatternMiner:
    def __init__(self, db: "Database"):
        self.db = db

    def vote(self, seq: list[int], regime: str) -> tuple[str, str, int, int, float, float]:
        n = len(seq)
        if n < 15:
            return "BIG", "—", 0, 0, 0.5, 0.0
        candidates: list[tuple[float, str, str, int, int, float]] = []
        for length in range(9, 4, -1):
            if n < length * 2:
                continue
            suffix = seq[-length:]
            obs: list[tuple[int, float]] = []
            for i in range(n - length):
                if seq[i:i + length] == suffix:
                    obs.append((seq[i + length], 0.5 ** ((n - i - length) / 420.0)))
            if len(obs) < CFG.min_support:
                continue
            weight_sum = sum(w for _, w in obs)
            sq_sum = sum(w * w for _, w in obs)
            eff_n = (weight_sum ** 2 / sq_sum) if sq_sum > 0 else 0.0
            p = (4.0 + sum(v * w for v, w in obs)) / (8.0 + weight_sum)
            big_c = sum(1 for v, _ in obs if v == 1)
            small_c = len(obs) - big_c
            pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in self._rle(suffix))
            key = f"family[{regime}]:{pattern_str}"
            rel = self.db.pattern_reliability(key)
            if rel < 0.10:
                continue
            score = rel * math.sqrt(max(1.0, eff_n))
            side = "BIG" if p >= 0.5 else "SMALL"
            candidates.append((score, side, pattern_str, big_c, small_c, p))
        if not candidates:
            return "BIG", "—", 0, 0, 0.5, 0.0
        big_s = sum(c[0] for c in candidates if c[1] == "BIG")
        small_s = sum(c[0] for c in candidates if c[1] == "SMALL")
        total = big_s + small_s
        if total <= 0:
            return "BIG", "—", 0, 0, 0.5, 0.0
        p_big = big_s / total
        side = "BIG" if p_big >= 0.5 else "SMALL"
        best = max((c for c in candidates if c[1] == side), key=lambda x: x[0])
        _, _, name, bc, sc, _ = best
        return side, name, bc, sc, clip(p_big), total

    @staticmethod
    def _rle(seq: list[int]) -> list[tuple[int, int]]:
        if not seq:
            return []
        out, cur, cnt = [], seq[0], 1
        for x in seq[1:]:
            if x == cur:
                cnt += 1
            else:
                out.append((cur, cnt))
                cur, cnt = x, 1
        out.append((cur, cnt))
        return out


# ============================================================
# NUMBER PREDICTOR
# ============================================================

class NumberPredictor:
    def __init__(self, allowed_digits: list[int]):
        self.allowed = allowed_digits or list(range(10))

    def predict(self, nums: list[int]) -> tuple[int, float, str]:
        if not nums:
            return self.allowed[0], 1.0 / len(self.allowed), "uniform"

        n = len(nums)
        base = [1.0] * 10
        for window, factor in ((30, 1.0), (100, 0.5), (300, 0.25)):
            for idx, val in enumerate(nums[-window:]):
                age = len(nums[-window:]) - idx
                base[val] += (0.5 ** (age / (window / 2))) * factor

        for order in (2, 3, 4):
            if n <= order:
                continue
            ctx = tuple(nums[-order:])
            obs = [0.0] * 10
            total = 0.0
            for i in range(order, n):
                if tuple(nums[i - order:i]) == ctx:
                    rw = 0.5 ** ((n - i) / 250.0)
                    obs[nums[i]] += rw
                    total += rw
            if total >= 0.5:
                for k in range(10):
                    base[k] += (obs[k] / total) * 4.0

        masked = [base[k] if k in self.allowed else 0.0 for k in range(10)]
        s = sum(masked)
        if s <= 0:
            return self.allowed[0], 1.0 / len(self.allowed), "fallback"
        dist = [x / s for x in masked]

        counts = Counter(nums[-100:])
        best = max(self.allowed, key=lambda k: (dist[k], counts.get(k, 0)))
        return best, dist[best], "ngram+recency"


# ============================================================
# WALK-FORWARD BACKTEST
# ============================================================

@dataclass
class BacktestResult:
    n_samples: int = 0
    correct: int = 0
    accuracy: float = 0.5
    brier: float = 0.25
    log_loss: float = 0.693
    recent_correct: int = 0
    recent_n: int = 0
    recent_accuracy: float = 0.5


def walkforward_backtest(seq: list[int], model: OnlineLogisticRegression,
                         warmup: int = 60) -> BacktestResult:
    n = len(seq)
    if n <= warmup:
        return BacktestResult()
    result = BacktestResult()
    brier_sum = 0.0
    ll_sum = 0.0
    recent_buf: deque[tuple[float, int]] = deque(maxlen=200)

    for i in range(warmup, n):
        x = extract_features(seq[:i])
        y = seq[i]
        p = model.predict_proba(x)
        p_c = max(1e-6, min(1 - 1e-6, p))

        result.n_samples += 1
        result.correct += int((p >= 0.5) == (y == 1))
        brier_sum += (p - y) ** 2
        ll_sum += -(y * math.log(p_c) + (1 - y) * math.log(1 - p_c))
        recent_buf.append((p, y))
        model.update(x, y)

    if result.n_samples > 0:
        result.accuracy = result.correct / result.n_samples
        result.brier = brier_sum / result.n_samples
        result.log_loss = ll_sum / result.n_samples
    if recent_buf:
        result.recent_n = len(recent_buf)
        result.recent_correct = sum(1 for p, y in recent_buf if (p >= 0.5) == (y == 1))
        result.recent_accuracy = result.recent_correct / result.recent_n
    return result


# ============================================================
# PARSER
# ============================================================

def parse_history(payload: Any) -> list[dict[str, Any]]:
    obj = payload
    for _ in range(8):
        if isinstance(obj, list):
            break
        if not isinstance(obj, dict):
            return []
        nxt = None
        for k in ("data", "list", "result", "items", "history", "records"):
            c = obj.get(k)
            if isinstance(c, (list, dict)):
                nxt = c
                break
        if nxt is None:
            return []
        obj = nxt
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
        records[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda r: int(r["issue"]))


# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, path: str):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA busy_timeout=20000")
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS history(issue TEXT PRIMARY KEY, num INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(
                issue TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, actual INTEGER,
                won_number INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pattern_models(
                name TEXT PRIMARY KEY, n REAL NOT NULL DEFAULT 0,
                loss REAL NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                issue TEXT, text TEXT NOT NULL, created REAL NOT NULL,
                expires REAL, sent INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._migrate()

    def _migrate(self) -> None:
        def cols(table: str) -> set[str]:
            return {r["name"] for r in self.c.execute(f"PRAGMA table_info({table})")}

        pred_cols = cols("predictions")
        for name, spec in (
            ("won_number", "INTEGER NOT NULL DEFAULT 0"),
            ("actual", "INTEGER"),
        ):
            if name not in pred_cols:
                LOG.warning("Migrating: adding predictions.%s", name)
                self.c.execute(f"ALTER TABLE predictions ADD COLUMN {name} {spec}")

        pm = cols("pattern_models")
        for name, spec in (
            ("wins", "INTEGER NOT NULL DEFAULT 0"),
            ("total", "INTEGER NOT NULL DEFAULT 0"),
            ("n", "REAL NOT NULL DEFAULT 0"),
            ("loss", "REAL NOT NULL DEFAULT 0"),
        ):
            if name not in pm:
                LOG.warning("Migrating: adding pattern_models.%s", name)
                self.c.execute(f"ALTER TABLE pattern_models ADD COLUMN {name} {spec}")

        ob = cols("outbox")
        if "attempts" not in ob:
            LOG.warning("Migrating: adding outbox.attempts")
            self.c.execute("ALTER TABLE outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

    def repair_won_number(self) -> int:
        """Recompute won_number for all settled predictions from actual+payload.
        Fixes historical data corruption from older versions."""
        rows = self.c.execute(
            "SELECT issue, payload, actual, won_number FROM predictions "
            "WHERE status IN ('WIN','LOSS') AND actual IS NOT NULL"
        ).fetchall()
        fixed = 0
        for r in rows:
            try:
                p = json.loads(r["payload"])
                pred_num = p.get("number")
                actual = int(r["actual"])
                correct = 1 if (isinstance(pred_num, int) and not isinstance(pred_num, bool)
                                and pred_num == actual) else 0
                if int(r["won_number"]) != correct:
                    self.c.execute(
                        "UPDATE predictions SET won_number=? WHERE issue=?",
                        (correct, r["issue"]),
                    )
                    fixed += 1
            except Exception:
                continue
        if fixed:
            LOG.warning("Repaired won_number on %d predictions", fixed)
        return fixed

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
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for r in records:
                self.c.execute(
                    "INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)",
                    (r["issue"], r["num"]),
                )
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN "
                "(SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,),
            )

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute(
            "SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER) ASC"
        ).fetchall()
        return [{"issue": r["issue"], "num": int(r["num"]),
                 "size": int(int(r["num"]) >= 5)} for r in rows]

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute(
            "SELECT issue,payload FROM predictions WHERE status='PENDING' "
            "ORDER BY CAST(issue AS INTEGER) ASC"
        ).fetchall()

    def learn_pattern(self, key: str, probability: float, actual: int) -> None:
        row = self.c.execute(
            "SELECT wins, total FROM pattern_models WHERE name=?", (key,)
        ).fetchone()
        old_w = int(row["wins"] or 0) if row else 0
        old_t = int(row["total"] or 0) if row else 0
        correct = int((probability >= 0.5) == (actual >= 0.5))
        wins, total = old_w + correct, old_t + 1
        if total > 300:
            wins //= 2
            total //= 2
        self.c.execute(
            "INSERT INTO pattern_models(name,n,loss,failures,wins,total) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "wins=excluded.wins, total=excluded.total",
            (key, 0, 0, 0, wins, total),
        )

    def pattern_reliability(self, key: str) -> float:
        row = self.c.execute(
            "SELECT wins, total FROM pattern_models WHERE name=?", (key,)
        ).fetchone()
        if not row:
            return 1.0
        total = int(row["total"] or 0)
        wins = int(row["wins"] or 0)
        if total < 3:
            return 1.0
        alpha = wins + 2
        beta = (total - wins) + 2
        win_rate = alpha / (alpha + beta)
        evidence = total / (total + 15.0)
        skill = max(0.0, (win_rate - 0.5) * 2.0)
        return max(0.05, min(2.0, 0.2 + 0.8 * skill * evidence))

    def queue(self, kind: str, issue: str | None, text: str,
              expires: float | None = None) -> None:
        self.c.execute(
            "INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires),
        )

    def prune(self) -> None:
        cutoff = time.time() - CFG.prediction_retention_days * 86400
        with self.transaction():
            self.c.execute("DELETE FROM predictions WHERE status != 'PENDING' AND created < ?",
                           (cutoff,))
            self.c.execute("DELETE FROM outbox WHERE sent IN (1, 2) AND created < ?", (cutoff,))

    def health_stats(self) -> dict[str, Any]:
        settled = int(self.c.execute(
            "SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')"
        ).fetchone()[0])
        wins = int(self.c.execute(
            "SELECT COUNT(*) FROM predictions WHERE status='WIN'"
        ).fetchone()[0])
        nh = int(self.c.execute(
            "SELECT COUNT(*) FROM predictions WHERE won_number=1"
        ).fetchone()[0])
        pending_out = int(self.c.execute(
            "SELECT COUNT(*) FROM outbox WHERE sent=0"
        ).fetchone()[0])
        return {
            "history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]),
            "settled": settled,
            "side_wins": wins,
            "side_win_rate": round(100.0 * wins / settled, 2) if settled else None,
            "number_hits": nh,
            "outbox_pending": pending_out,
        }

    def close(self) -> None:
        self.c.close()


# ============================================================
# ENGINE
# ============================================================

class Engine:
    def __init__(self, db: Database):
        self.db = db
        self.miner = PatternMiner(db)
        self.model = OnlineLogisticRegression(N_FEATURES)
        self.calibrator = TemperatureCalibrator()
        self.backtest = BacktestResult()
        self.model_ready = False
        self._load_state()

    def _load_state(self) -> None:
        state = self.db.get("lr_state")
        if state:
            self.model.load_state(state)
            self.model_ready = True
        cstate = self.db.get("cal_state")
        if cstate:
            self.calibrator.load_state(cstate)
        bt = self.db.get("backtest")
        if bt:
            try:
                self.backtest = BacktestResult(**bt)
            except TypeError:
                pass

    def save_state(self) -> None:
        self.db.set("lr_state", self.model.to_state())
        self.db.set("cal_state", self.calibrator.to_state())
        self.db.set("backtest", self.backtest.__dict__)

    def bootstrap(self, history: list[dict[str, Any]]) -> bool:
        """Returns True if bootstrap succeeded."""
        seq = [int(r["size"]) for r in history]
        if len(seq) < CFG.backtest_min_samples:
            LOG.info("Bootstrap skipped: %d < %d samples required",
                     len(seq), CFG.backtest_min_samples)
            return False
        LOG.info("Bootstrapping on %d samples…", len(seq))
        fresh = OnlineLogisticRegression(N_FEATURES)
        self.backtest = walkforward_backtest(seq, fresh)
        self.model = fresh
        self.model_ready = True
        self.save_state()
        LOG.info(
            "Bootstrap done: acc=%.2f%% brier=%.4f logloss=%.4f (n=%d) | recent=%.2f%% (n=%d)",
            self.backtest.accuracy * 100, self.backtest.brier, self.backtest.log_loss,
            self.backtest.n_samples, self.backtest.recent_accuracy * 100,
            self.backtest.recent_n,
        )
        return True

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any] | None:
        seq = [int(r["size"]) for r in history]
        nums = [int(r["num"]) for r in history]
        if len(seq) < CFG.min_history or not self.model_ready:
            return None

        x = extract_features(seq)
        p_raw = self.model.predict_proba(x)
        p_cal = self.calibrator.apply(p_raw)

        tail = seq[-24:]
        flips = sum(a != b for a, b in zip(tail, tail[1:])) if len(tail) > 1 else 0
        flip_rate = flips / max(1, len(tail) - 1)
        last_run = 1
        for v in reversed(tail[:-1]):
            if v != tail[-1]:
                break
            last_run += 1
        if last_run >= 5:
            regime = "RUN"
        elif flip_rate >= 0.72:
            regime = "ALT"
        elif abs(sum(tail) / len(tail) - 0.5) > 0.20:
            regime = "IMBAL"
        else:
            regime = "MIXED"

        pat_side, pat_name, pat_big, pat_small, pat_p, pat_weight = self.miner.vote(seq, regime)

        if pat_name != "—" and pat_weight > 0:
            w_lr = 1.0
            w_pat = math.sqrt(pat_weight)
            p_final = (p_cal * w_lr + pat_p * w_pat) / (w_lr + w_pat)
            reason = "LR + Pattern"
        else:
            p_final = p_cal
            reason = "LR only"

        p_final = clip(p_final)
        side = "BIG" if p_final >= 0.5 else "SMALL"
        allowed = list(range(5, 10)) if side == "BIG" else list(range(0, 5))

        # Number prediction — MUST succeed (fall back on any exception)
        try:
            num_pred, num_conf, num_method = NumberPredictor(allowed).predict(nums)
            if not isinstance(num_pred, int) or not (0 <= num_pred <= 9):
                raise ValueError(f"invalid num_pred={num_pred!r}")
            num_conf = max(0.0, min(1.0, float(num_conf)))
        except Exception:
            LOG.exception("Number prediction failed — using fallback")
            num_pred, num_conf, num_method = allowed[0], 1.0 / len(allowed), "fallback"

        # Sanity: predicted number MUST match side
        if (side == "BIG" and num_pred < 5) or (side == "SMALL" and num_pred >= 5):
            LOG.error("Number/side mismatch (side=%s num=%d) — correcting", side, num_pred)
            num_pred = allowed[0]

        b = self.backtest
        edge = (b.accuracy - 0.5) if b.n_samples >= CFG.backtest_min_samples else 0.0

        return {
            "issue": issue,
            "features": x,
            "p_raw": p_raw,
            "p_cal": p_cal,
            "p": p_final,
            "side": side,
            "number": num_pred,
            "num_conf": num_conf,
            "num_method": num_method,
            "regime": regime,
            "pattern": pat_name,
            "pat_big": pat_big,
            "pat_small": pat_small,
            "reason": reason,
            "edge": edge,
            "backtest_acc": b.accuracy,
            "backtest_n": b.n_samples,
            "backtest_brier": b.brier,
            "recent_acc": b.recent_accuracy,
            "recent_n": b.recent_n,
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        for row in self.db.pending_predictions():
            actual = lookup.get(row["issue"])
            if not actual:
                continue
            try:
                payload = json.loads(row["payload"])
            except Exception:
                LOG.exception("Corrupt payload for issue %s — skipping", row["issue"])
                continue

            actual_size = int(actual["size"])
            actual_num = int(actual["num"])
            actual_side = "BIG" if actual_size == 1 else "SMALL"

            pred_side = payload.get("side")
            pred_num_raw = payload.get("number")

            # Validate prediction data — treat anything odd as a miss (never a false jackpot)
            if pred_side not in ("BIG", "SMALL"):
                LOG.warning("Issue %s: bad pred_side=%r — treating as loss",
                            row["issue"], pred_side)
                pred_side = "INVALID"
            pred_num = as_int_in_range(pred_num_raw, 0, 9)
            if pred_num is None:
                LOG.warning("Issue %s: bad pred_num=%r — treating as miss",
                            row["issue"], pred_num_raw)
                pred_num = -1  # impossible value, guaranteed mismatch

            won_side = pred_side == actual_side
            won_num = pred_num == actual_num

            # Explicit sticker logic — dead simple, no ambiguity
            if won_side and won_num:
                sticker, sticker_kind = CFG.jack_sticker, "JACKPOT"
            elif won_side:
                sticker, sticker_kind = CFG.win_sticker, "WIN"
            else:
                sticker, sticker_kind = CFG.loss_sticker, "LOSS"

            LOG.info(
                "Settle %s: pred=(%s,%s) actual=(%s,%d) won_side=%s won_num=%s → %s",
                row["issue"], pred_side, pred_num, actual_side, actual_num,
                won_side, won_num, sticker_kind,
            )

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=?, won_number=? "
                    "WHERE issue=? AND status='PENDING'",
                    ("WIN" if won_side else "LOSS", actual_num, int(won_num), row["issue"]),
                )
                self.db.queue("sticker", row["issue"], sticker,
                              expires=time.time() + CFG.sticker_ttl)

            # Online updates (outside transaction is fine)
            feats = payload.get("features")
            if feats and isinstance(feats, list) and len(feats) == N_FEATURES:
                try:
                    self.model.update([float(f) for f in feats], actual_size)
                except Exception:
                    LOG.exception("Model update failed for %s", row["issue"])
            p_used = as_float(payload.get("p_cal", payload.get("p", 0.5)), 0.5)
            self.calibrator.observe(p_used, actual_size)

            pat_name = payload.get("pattern")
            regime = payload.get("regime")
            if pat_name and pat_name != "—" and regime:
                try:
                    self.db.learn_pattern(
                        f"family[{regime}]:{pat_name}",
                        as_float(payload.get("p", 0.5), 0.5),
                        actual_size,
                    )
                except Exception:
                    LOG.exception("learn_pattern failed for %s", row["issue"])

        self.save_state()


def format_prediction(p: dict[str, Any]) -> str:
    """Never throws — every field has a safe default."""
    prob = as_float(p.get("p"), 0.5)
    conf = max(prob, 1.0 - prob) * 100
    filled = max(0, min(10, int(conf // 10)))
    bar = "█" * filled + "░" * (10 - filled)
    side = p.get("side") if p.get("side") in ("BIG", "SMALL") else "BIG"
    marker = "🔴" if side == "BIG" else "🔵"

    pat_line = ""
    if p.get("pattern", "—") != "—":
        pat_line = (f"\n🧩 <b>PATTERN:</b> <code>{escape(p['pattern'])}</code> "
                    f"(B[{p.get('pat_big', 0)}] S[{p.get('pat_small', 0)}])")

    # NUMBER — always shown, even if unavailable
    num_val = as_int_in_range(p.get("number"), 0, 9)
    num_conf = as_float(p.get("num_conf"), 0.1)
    num_conf = max(0.0, min(1.0, num_conf))

    if num_val is None:
        num_line = "\n🔢 <b>TOP NUMBER:</b> <i>unavailable this round</i>"
    else:
        mult = num_conf / 0.1 if num_conf > 0 else 0.0
        if num_conf < 0.105:
            strength = "⚠️ very weak signal"
        elif num_conf < 0.12:
            strength = "🟡 weak signal"
        elif num_conf < 0.15:
            strength = "🟢 moderate signal"
        else:
            strength = "🟢 strong signal"
        num_line = (
            f"\n🔢 <b>TOP NUMBER:</b> <code>{num_val}</code> "
            f"({num_conf * 100:.1f}% · ×{mult:.2f} vs uniform · {strength})"
        )

    bt_acc = as_float(p.get("backtest_acc"), 0.5) * 100
    bt_n = int(p.get("backtest_n", 0) or 0)
    bt_brier = as_float(p.get("backtest_brier"), 0.25)
    recent_acc = as_float(p.get("recent_acc"), 0.5) * 100
    recent_n = int(p.get("recent_n", 0) or 0)
    edge = as_float(p.get("edge"), 0.0) * 100

    if edge < CFG.edge_warn * 100 or recent_acc < 51.0:
        edge_line = "⚠️ <b>EDGE WARNING:</b> near chance — entertainment only"
    elif edge < CFG.edge_ok * 100:
        edge_line = "🟡 <b>EDGE:</b> tiny"
    else:
        edge_line = "🟢 <b>EDGE:</b> positive (still not a guarantee)"

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(p.get('issue', '—'))}</code>"
        f"{num_line}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime', '—'))}{pat_line}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"<b>📈 MODEL STATS</b>\n"
        f"• Backtest acc: <b>{bt_acc:.2f}%</b> (n={bt_n})\n"
        f"• Recent acc: <b>{recent_acc:.1f}%</b> (n={recent_n})\n"
        f"• Brier: {bt_brier:.3f}  (0.25 = chance)\n"
        f"• Reason: {escape(p.get('reason', '—'))}\n"
        f"{edge_line}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{DISCLOSURE}"
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
        self.sources_ok = 0
        self.sources_total = 0
        self.last_source_error: str | None = None

    async def pause(self, sec: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, sec))
        except asyncio.TimeoutError:
            pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def one(url: str) -> list[dict[str, Any]] | None:
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception as e:
                LOG.warning("Source fetch failed: %s — %s", url, e)
                return None

        results = await asyncio.gather(*(one(u) for u in CFG.urls))
        sources = [r for r in results if r]
        self.sources_total = len(CFG.urls)
        self.sources_ok = len(sources)
        if not sources:
            self.last_source_error = "all_sources_failed"
            raise RuntimeError("HistoryUnavailable")

        def max_issue(src: list[dict[str, Any]]) -> int:
            return max((int(r["issue"]) for r in src), default=0)

        sources.sort(key=max_issue, reverse=True)

        if len(sources) > 1:
            top_map = {r["issue"]: r["num"] for r in sources[0][-20:]}
            for other in sources[1:]:
                other_map = {r["issue"]: r["num"] for r in other[-20:]}
                for iss, num in top_map.items():
                    if iss in other_map and other_map[iss] != num:
                        LOG.warning("Source disagreement issue=%s: %s vs %s",
                                    iss, num, other_map[iss])

        merged: dict[str, dict[str, Any]] = {}
        for src in sources:
            for row in src:
                merged.setdefault(row["issue"], row)

        self.last_source_error = None
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        last_prune = 0.0
        last_bootstrap_attempt = 0.0

        while not self.stop.is_set():
            try:
                self.db.ingest(await self.fetch(session))
                history = self.db.history()
                self.engine.settle(history)
                self.last_fetch = time.time()

                # Retry bootstrap every N sec until model is ready
                if (not self.engine.model_ready
                        and len(history) >= CFG.backtest_min_samples
                        and time.time() - last_bootstrap_attempt > CFG.bootstrap_retry_sec):
                    last_bootstrap_attempt = time.time()
                    LOG.info("Attempting bootstrap (history=%d)", len(history))
                    try:
                        self.engine.bootstrap(history)
                    except Exception:
                        LOG.exception("Bootstrap attempt failed")

                if time.time() - last_prune > 3600:
                    self.db.prune()
                    last_prune = time.time()

                # Predict only if model ready AND enough history
                if len(history) >= CFG.min_history and self.engine.model_ready:
                    issue = str(int(history[-1]["issue"]) + 1)
                    exists = self.db.c.execute(
                        "SELECT 1 FROM predictions WHERE issue=?", (issue,)
                    ).fetchone()
                    if not exists:
                        pred = self.engine.predict(history, issue)
                        if pred:
                            try:
                                msg = format_prediction(pred)
                            except Exception:
                                LOG.exception("format_prediction failed for %s", issue)
                                msg = None
                            if msg:
                                with self.db.transaction():
                                    self.db.c.execute(
                                        "INSERT INTO predictions(issue,created,status,payload) "
                                        "VALUES (?,?,?,?)",
                                        (issue, time.time(), "PENDING", json.dumps(pred)),
                                    )
                                    self.db.queue(
                                        "prediction", issue, msg,
                                        time.time() + CFG.delivery_ttl,
                                    )
                                LOG.info("Prediction queued: %s → %s #%d",
                                         issue, pred["side"], pred["number"])
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                LOG.exception("Worker error (failures=%d, sources=%d/%d, model_ready=%s)",
                              failures, self.sources_ok, self.sources_total,
                              self.engine.model_ready)
            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            self.db.c.execute(
                "UPDATE outbox SET sent=2 WHERE sent=0 AND expires IS NOT NULL AND expires <= ?",
                (time.time(),),
            )
            row = self.db.c.execute(
                "SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if not row:
                await self.pause(0.5)
                continue
            try:
                url = (f"https://api.telegram.org/bot{CFG.token}/"
                       + ("sendSticker" if row["kind"] == "sticker" else "sendMessage"))
                body = ({"chat_id": CFG.chat, "sticker": row["text"]}
                        if row["kind"] == "sticker"
                        else {"chat_id": CFG.chat, "text": row["text"],
                              "parse_mode": "HTML"})
                async with session.post(url, json=body) as res:
                    data = await res.json(content_type=None)
                    if res.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue
                    code = data.get("error_code")
                    if code == 401:
                        LOG.critical("Telegram 401 — invalid token. Stopping.")
                        self.stop.set()
                        return
                    if code == 429:
                        retry = float(data.get("parameters", {}).get("retry_after", 10))
                        await self.pause(retry)
                        continue
                    if code in (400, 403, 404):
                        attempts = int(row["attempts"] or 0) + 1
                        if attempts >= CFG.max_outbox_attempts:
                            LOG.error("Dropping outbox id=%s after %d attempts: %s",
                                      row["id"], attempts, data.get("description"))
                            self.db.c.execute(
                                "UPDATE outbox SET sent=2, attempts=? WHERE id=?",
                                (attempts, row["id"]),
                            )
                        else:
                            self.db.c.execute(
                                "UPDATE outbox SET attempts=? WHERE id=?",
                                (attempts, row["id"]),
                            )
                            await self.pause(5.0)
                        continue
                    await self.pause(backoff)
                    backoff = min(60.0, backoff * 2)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Sender error")
                await self.pause(backoff)
                backoff = min(60.0, backoff * 2)

    async def health(self, request: web.Request) -> web.Response:
        now = time.time()
        b = self.engine.backtest
        return web.json_response({
            "status": "ok" if self.last_fetch and now - self.last_fetch < 90 else "degraded",
            "last_fetch_age": round(now - self.last_fetch, 1) if self.last_fetch else None,
            "sources_ok": self.sources_ok,
            "sources_total": self.sources_total,
            "last_source_error": self.last_source_error,
            "model_ready": self.engine.model_ready,
            "backtest": {
                "n": b.n_samples,
                "accuracy": round(b.accuracy * 100, 2),
                "brier": round(b.brier, 4),
                "log_loss": round(b.log_loss, 4),
                "recent_n": b.recent_n,
                "recent_accuracy": round(b.recent_accuracy * 100, 2),
            },
            "calibrator_T": round(self.engine.calibrator.T, 3),
            **self.db.health_stats(),
        })

    async def debug_recent(self, request: web.Request) -> web.Response:
        """Returns last 20 settled predictions with computed won_side/won_num.
        Use to verify sticker behavior against actual data."""
        rows = self.db.c.execute(
            "SELECT issue, created, status, actual, won_number, payload "
            "FROM predictions WHERE status IN ('WIN','LOSS') "
            "ORDER BY created DESC LIMIT 20"
        ).fetchall()
        out = []
        for r in rows:
            try:
                p = json.loads(r["payload"])
                actual_num = int(r["actual"]) if r["actual"] is not None else None
                pred_num = p.get("number")
                pred_side = p.get("side")
                actual_side = ("BIG" if actual_num >= 5 else "SMALL") if actual_num is not None else None
                won_side = pred_side == actual_side if actual_side else None
                won_num = (isinstance(pred_num, int) and not isinstance(pred_num, bool)
                           and pred_num == actual_num) if actual_num is not None else None
                expected_sticker = (
                    "JACKPOT" if (won_side and won_num)
                    else "WIN" if won_side
                    else "LOSS"
                ) if won_side is not None else "?"
                out.append({
                    "issue": r["issue"],
                    "status": r["status"],
                    "pred_side": pred_side,
                    "pred_num": pred_num,
                    "actual_side": actual_side,
                    "actual_num": actual_num,
                    "won_side": won_side,
                    "won_num": won_num,
                    "expected_sticker": expected_sticker,
                    "stored_won_number": r["won_number"],
                })
            except Exception as e:
                out.append({"issue": r["issue"], "error": str(e)})
        return web.json_response(out)


async def main() -> None:
    CFG.validate()
    db = Database(CFG.db_path)

    # Repair historical won_number anomalies once at startup
    try:
        db.repair_won_number()
    except Exception:
        LOG.exception("repair_won_number failed (non-fatal)")

    runtime = Runtime(db)

    app = web.Application()
    app.router.add_get("/", runtime.health)
    app.router.add_get("/debug/recent", runtime.debug_recent)
    runner = web.AppRunner(app)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop.set)

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session)),
                asyncio.create_task(runtime.sender(session)),
            ]
            await asyncio.gather(*tasks)
    finally:
        await runner.cleanup()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
