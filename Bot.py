"""
FTW Predictor V36 — HYBRID X (Super Advanced + Trend + Seed Patterns)
Python 3.10+
Deps: aiohttp
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

# ============================================================
# CONFIG
# ============================================================

DEFAULT_HISTORY_URL = (
    "https://sky-predictor-1012593186417.asia-southeast1.run.app/"
    "api/wingo-history-1m-1000"
)

@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_hybrid_v36.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500 
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (HYBRID V36 X)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token: raise ValueError("TELEGRAM_TOKEN is required")
        if not self.chat: raise ValueError("TELEGRAM_CHAT is required")

CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v36")


# ============================================================
# HELPERS & MATH
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
    if isinstance(value, bool): return None
    text = str(value).strip()
    if not text.isascii() or not text.isdigit(): return None
    try: return int(text)
    except ValueError: return None

def weighted_probability(observations: list[tuple[int, float]]) -> tuple[float, float]:
    if not observations: return 0.5, 0.0
    weight_sum = sum(weight for _, weight in observations)
    squared_weight_sum = sum(weight * weight for _, weight in observations)
    effective_n = (weight_sum * weight_sum / squared_weight_sum) if squared_weight_sum > 0 else 0.0
    probability = (4.0 + sum(value * weight for value, weight in observations)) / (8.0 + weight_sum)
    return clip(probability), effective_n

def recency_weight(index: int, n: int, half_life: float = 330.0) -> float:
    age = max(0, n - 1 - index)
    return 0.5 ** (age / half_life)

def get_regime(seq: list[int]) -> str:
    tail = seq[-24:]
    if len(tail) < 8: return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    flip_rate = flips / max(1, len(tail) - 1)
    run = 1
    for value in reversed(tail[:-1]):
        if value != tail[-1]: break
        run += 1
    if run >= 5: return "RUN"
    if flip_rate >= 0.72: return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20: return "IMBAL"
    return "MIXED"

def trend_features(seq: list[int]) -> list[float]:
    if not seq: return [0.0] * 20
    def mean(k: int) -> float:
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5
    def ema(k: int) -> float:
        if not seq: return 0.5
        alpha = 2.0 / (k + 1)
        v = float(seq[0])
        for x in seq[1:]: v = alpha * x + (1.0 - alpha) * v
        return v
    def std(k: int) -> float:
        arr = seq[-k:]
        if len(arr) < 2: return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))
    
    tail = seq[-24:]
    alt = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    run_score = (2 * seq[-1] - 1) * min(len(list(filter(lambda x: x == seq[-1], seq[-12:]))), 12) / 12.0
    macd = (ema(12) - ema(26)) * 2.0
    
    return [1.0, 2*mean(3)-1, 2*mean(5)-1, 2*mean(10)-1, 2*mean(24)-1, 2*mean(48)-1, 
            2*mean(96)-1, mean(3)-mean(10), mean(5)-mean(24), mean(10)-mean(48), 
            run_score, 2*seq[-1]-1, (2*alt-1)*(2*seq[-1]-1), macd, 2*std(10)-1, 
            2*std(24)-1, mean(3)-mean(12), mean(8)-mean(24), mean(12)-mean(48), mean(6)-mean(24)]

def logistic(weights: list[float], features: list[float]) -> float:
    return sigmoid(sum(w * x for w, x in zip(weights, features)))

def family_run(seq: list[int]) -> list[tuple[int, int]]:
    if not seq: return []
    out, cur, count = [], seq[0], 1
    for x in seq[1:]:
        if x == cur: count += 1
        else:
            out.append((cur, count))
            cur, count = x, 1
    out.append((cur, count))
    return out

def family_signature(seq: list[int]) -> tuple:
    return tuple((int(v), cnt) for v, cnt in family_run(seq))

def parse_history(payload: Any) -> list[dict[str, Any]]:
    obj = payload
    for _ in range(8):
        if isinstance(obj, list): break
        if not isinstance(obj, dict): return []
        next_obj = None
        for key in ("data", "list", "result", "items", "history", "records"):
            candidate = obj.get(key)
            if isinstance(candidate, (list, dict)):
                next_obj = candidate
                break
        if next_obj is None: return []
        obj = next_obj
    if not isinstance(obj, list): return []

    records = {}
    for item in obj:
        if not isinstance(item, dict): continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))
        if issue is None or number is None or not 0 <= number <= 9: continue
        issue_key = str(issue)
        record = {"issue": issue_key, "num": number, "size": int(number >= 5)}
        records[issue_key] = record
    return sorted(records.values(), key=lambda row: int(row["issue"]))


# ============================================================
# PATTERN TRACKER (Family + Seed)
# ============================================================

class QuantumTracker:
    def __init__(self, min_len=3, max_len=8, min_support=12):
        self.min_len, self.max_len, self.min_support = min_len, max_len, min_support

    def match_tail_patterns(self, seq: list[int]) -> list[dict]:
        n = len(seq)
        if n < 18: return []
        results = []
        for length in range(self.max_len, self.min_len - 1, -1):
            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs = []
            for i in range(n - length):
                if family_signature(seq[i:i + length]) == sig:
                    obs.append((seq[i + length], 0.5 ** ((n - 1 - (i + length)) / 420.0)))
            if len(obs) < self.min_support: continue
            p, eff_n = weighted_probability(obs)
            score = abs(p - 0.5) * math.sqrt(max(1.0, eff_n))
            if score > 0.0:
                pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in sig)
                results.append({"sig": sig, "p": p, "support": len(obs), "eff_n": eff_n, "score": score, "pattern_str": pattern_str})
            break
        return results


# ============================================================
# DATABASE (Advanced Learning & Blacklist)
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
            CREATE TABLE IF NOT EXISTS predictions(issue TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, actual INTEGER, won_number INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS expert_models(name TEXT NOT NULL, regime TEXT NOT NULL, n REAL NOT NULL, loss REAL NOT NULL, failures INTEGER NOT NULL, PRIMARY KEY(name, regime));
            CREATE TABLE IF NOT EXISTS blacklist(pattern TEXT PRIMARY KEY, fails INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, issue TEXT, text TEXT NOT NULL, created REAL NOT NULL, expires REAL, sent INTEGER NOT NULL DEFAULT 0);
        """)

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try: yield
        except Exception:
            if self.c.in_transaction: self.c.execute("ROLLBACK")
            raise
        else:
            if self.c.in_transaction: self.c.execute("COMMIT")

    def get(self, key: str, default: Any = None) -> Any:
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute("INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for r in records:
                self.c.execute("INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute("DELETE FROM history WHERE issue NOT IN (SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)", (CFG.max_history,))

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute("SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER) ASC").fetchall()
        return [{"issue": r["issue"], "num": int(r["num"]), "size": int(int(r["num"]) >= 5)} for r in rows]

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute("SELECT issue,payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER) ASC").fetchall()

    def learn_expert(self, name: str, regime: str, probability: float, actual: int) -> None:
        row = self.c.execute("SELECT n, loss, failures FROM expert_models WHERE name=? AND regime=?", (name, regime)).fetchone()
        old_n, old_loss, old_fails = (float(row["n"]), float(row["loss"]), int(row["failures"])) if row else (0.0, 0.0, 0)
        n = 0.985 * old_n + 1.0
        loss = 0.985 * old_loss + (clip(probability) - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        fails = old_fails + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO expert_models(name,regime,n,loss,failures) VALUES (?,?,?,?,?) ON CONFLICT(name,regime) DO UPDATE SET n=excluded.n, loss=excluded.loss, failures=excluded.failures",
            (name, regime, n, loss, fails)
        )

    def adapt_expert(self, name: str, regime: str, probability: float) -> tuple[float, float]:
        row = self.c.execute("SELECT n, loss, failures FROM expert_models WHERE name=? AND regime=?", (name, regime)).fetchone()
        if not row or float(row["n"]) < 1.0: return probability, 1.0
        n, mean_loss, fails = float(row["n"]), float(row["loss"]) / float(row["n"]), int(row["failures"])
        reliability = math.exp(-3.0 * max(-0.10, mean_loss - 0.25))
        if fails >= 2: reliability *= 0.3
        if fails >= 4: reliability *= 0.1
        if fails >= 3 and mean_loss > 0.26: probability = 1.0 - probability
        return clip(probability), max(0.1, min(1.8, reliability))

    def is_blacklisted(self, pattern: str) -> bool:
        row = self.c.execute("SELECT fails FROM blacklist WHERE pattern=?", (pattern,)).fetchone()
        return bool(row and row["fails"] >= 3)

    def add_blacklist(self, pattern: str, wrong: bool):
        row = self.c.execute("SELECT fails FROM blacklist WHERE pattern=?", (pattern,)).fetchone()
        fails = (row["fails"] if row else 0) + (1 if wrong else 0)
        self.c.execute("INSERT INTO blacklist(pattern,fails) VALUES (?,?) ON CONFLICT(pattern) DO UPDATE SET fails=excluded.fails", (pattern, fails))

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)", (kind, issue, text, time.time(), expires))

    def health_stats(self) -> dict[str, Any]:
        settled = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0])
        wins = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0])
        return {"history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]), "settled": settled, "side_wins": wins, "side_win_rate": round(100.0 * wins / settled, 2) if settled else None}

    def close(self) -> None: self.c.close()


# ============================================================
# ENGINE (META-LEARNER + SEED PATTERNS + TREND)
# ============================================================

EXPERT_NAMES = ["patterns", "seed", "base", "recent", "markov1", "markov2", "markov3", "run", "regime", "momentum", "reversal", "trend"]

class Engine:
    def __init__(self, db: Database): self.db = db

    def mine_patterns(self, seq: list[int]) -> tuple[list[tuple], list[tuple]]:
        if len(seq) < 18: return [], []
        n = len(seq)
        
        # 1. Family Patterns (QuantumTracker)
        tracker = QuantumTracker(min_len=3, max_len=10, min_support=CFG.min_support)
        family_pats = []
        for item in tracker.match_tail_patterns(seq)[:4]:
            key = "q:" + item["pattern_str"]
            if not self.db.is_blacklisted(key):
                family_pats.append((item["score"], item["p"], item["support"], item["eff_n"], key))
        
        # 2. Exact Seed Patterns
        seed_pats = []
        for length in range(4, 13): # seed length from 4 to 12
            suffix = tuple(seq[-length:])
            obs = []
            for i in range(n - length):
                if tuple(seq[i:i + length]) == suffix:
                    obs.append((seq[i + length], 0.5 ** ((n - 1 - (i + length)) / 420.0)))
            if len(obs) >= CFG.min_support:
                p, eff_n = weighted_probability(obs)
                score = abs(p - 0.5) * math.sqrt(max(1.0, eff_n))
                if score > 0.0:
                    bits = "".join("B" if x else "S" for x in suffix)
                    key = f"seed:{bits}"
                    if not self.db.is_blacklisted(key):
                        seed_pats.append((score, p, len(obs), eff_n, key))
                        
        return family_pats, seed_pats

    def _mine_experts(self, seq: list[int]) -> tuple[dict[str, float], str]:
        n = len(seq)
        if not n: return {name: 0.5 for name in EXPERT_NAMES}, "—"

        experts = {}
        pattern_display = "—"

        # 1. Patterns & Seed
        fam_pats, seed_pats = self.mine_patterns(seq)
        
        if fam_pats:
            pattern_display = fam_pats[0][4].split(":")[-1]
            candidates = [(self.db.adapt_expert(k, get_regime(seq), p)[0], self.db.adapt_expert(k, get_regime(seq), p)[1] * math.sqrt(max(1.0, eff))) for _, p, _, eff, k in fam_pats]
            if candidates: experts["patterns"] = clip(sum(p * w for p, w in candidates) / sum(w for _, w in candidates))

        if seed_pats:
            candidates = [(self.db.adapt_expert(k, get_regime(seq), p)[0], self.db.adapt_expert(k, get_regime(seq), p)[1] * math.sqrt(max(1.0, eff))) for _, p, _, eff, k in seed_pats]
            if candidates: experts["seed"] = clip(sum(p * w for p, w in candidates) / sum(w for _, w in candidates))

        # 2. Baseline & Markov
        experts["base"] = weighted_probability([(val, recency_weight(i, n)) for i, val in enumerate(seq)])[0]
        experts["recent"] = (4.0 + sum(seq[-100:])) / (8.0 + len(seq[-100:]))

        for order in (1, 2, 3):
            if n <= order: continue
            context = tuple(seq[-order:])
            obs = [(seq[i], recency_weight(i, n)) for i in range(order, n) if tuple(seq[i - order:i]) == context]
            if len(obs) >= CFG.min_support: experts[f"markov{order}"] = weighted_probability(obs)[0]

        # 3. Run Continuation
        if n >= 2:
            last_val, current_run = seq[-1], 1
            for v in reversed(seq[:-1]):
                if v != last_val: break
                current_run += 1
            run_at = [1] * n
            for i in range(1, n): run_at[i] = run_at[i-1] + 1 if seq[i] == seq[i-1] else 1
            obs = [(seq[i], recency_weight(i, n)) for i in range(1, n) if seq[i-1] == last_val and min(run_at[i-1], 8) == min(current_run, 8)]
            if len(obs) >= CFG.min_support: experts["run"] = weighted_probability(obs)[0]

        # 4. Regime
        regime = get_regime(seq)
        obs = [(seq[i], recency_weight(i, n)) for i in range(24, n) if get_regime(seq[max(0, i - 96):i]) == regime]
        if len(obs) >= CFG.min_support: experts["regime"] = weighted_probability(obs)[0]

        # 5. Trend (MACD & RSI)
        features = trend_features(seq)
        tw = self.db.get("trend_weights")
        if not isinstance(tw, list) or len(tw) != len(features): tw = [0.0] * len(features)
        experts["trend"] = logistic(tw, features)

        # 6. Momentum & Reversal
        momentum = sum(seq[-4:]) / 4.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)
        experts["reversal"] = 1.0 - experts["momentum"]

        for k in EXPERT_NAMES: experts[k] = clip(experts.get(k, 0.5), 0.05, 0.95)
        return experts, pattern_display

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(row["size"]) for row in history]
        experts, pattern_display = self._mine_experts(seq)
        regime = get_regime(seq)
        
        current_run = 1
        if len(seq) >= 2:
            last_val = seq[-1]
            for v in reversed(seq[:-1]):
                if v != last_val: break
                current_run += 1
        is_trend_breaking = current_run >= 4

        adjusted = []
        for name, prob in experts.items():
            corrected, weight = self.db.adapt_expert(f"expert:{name}:{regime}", prob)
            if name in ("patterns", "seed"): weight *= 2.5
            if is_trend_breaking and name in ("momentum", "run", "recent", "base"): weight *= 0.10
            if is_trend_breaking and name == "reversal": weight *= 3.50
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5

        # 🔥 Meta-Learner
        m_feats = [logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES] + [1.0]
        mw = self.db.get("meta_weights")
        if not isinstance(mw, list) or len(mw) != len(m_feats): mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
        p_meta = sigmoid(sum(w * f for w, f in zip(mw, m_feats)))
        
        alpha = min(self.db.get("meta_n", 0) / 80.0, 1.0)
        p = clip(alpha * p_meta + (1.0 - alpha) * p_avg)

        side = "BIG" if p >= 0.5 else "SMALL"
        allowed_nums = range(5, 10) if side == "BIG" else range(0, 5)
        
        num_dist = exact_number_distribution(history)
        side_adj = [num_dist[k] * 1.5 if k in allowed_nums else num_dist[k] * 0.5 for k in range(10)]
        total = sum(side_adj)
        side_adj = [x / total for x in side_adj]
        num_pred = max(allowed_nums, key=lambda k: side_adj[k])

        return {
            "issue": issue,
            "p": p,
            "side": side,
            "number": num_pred,
            "regime": regime,
            "pattern": pattern_display,
            "experts": experts,
            "features": trend_features(seq),
            "meta_feats": m_feats,
            "source": "Hybrid V36 X (Super)",
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        for row in self.db.pending_predictions():
            actual = lookup.get(row["issue"])
            if not actual: continue

            payload = json.loads(row["payload"])
            actual_size = int(actual["size"])
            won_side = int(float(payload["p"]) >= 0.5) == actual_size
            won_num = payload.get("number") == actual["num"]

            with self.db.transaction():
                self.db.c.execute("UPDATE predictions SET status=?, actual=?, won_number=? WHERE issue=? AND status='PENDING'", ("WIN" if won_side else "LOSS", actual["num"], int(won_num), row["issue"]))
                
                regime = payload.get("regime", "MIXED")
                for name, prob in payload.get("experts", {}).items():
                    self.db.learn_expert(f"expert:{name}:{regime}", regime, float(prob), actual_size)
                
                pat_str = payload.get("pattern")
                if pat_str and pat_str != "—":
                    self.db.add_blacklist("q:" + pat_str, wrong=not won_side)

                # 🔥 Learn Trend Weights
                if payload.get("features") and payload["experts"].get("trend"):
                    err = actual_size - payload["experts"]["trend"]
                    tw = self.db.get("trend_weights") or [0.0] * len(payload["features"])
                    self.db.set("trend_weights", [clip(w + 0.05 * (err * x - 0.015 * w), -4.0, 4.0) for w, x in zip(tw, payload["features"])])

                # 🔥 Learn Meta Weights
                if payload.get("meta_feats"):
                    feats = payload["meta_feats"]
                    if len(feats) == len(EXPERT_NAMES) + 1:
                        mw = self.db.get("meta_weights") or [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
                        mp = sigmoid(sum(w * f for w, f in zip(mw, feats)))
                        grad = 2.0 * (actual_size - mp) * mp * (1.0 - mp)
                        self.db.set("meta_weights", [clip(w * 0.999 + 0.06 * grad * f, -4.0, 4.0) for w, f in zip(mw, feats)])
                        self.db.set("meta_n", self.db.get("meta_n", 0) + 1)

                sticker = CFG.jack_sticker if (won_side and won_num) else CFG.win_sticker if won_side else CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)


def exact_number_distribution(history: list[dict[str, Any]]) -> list[float]:
    nums = [int(row["num"]) for row in history]
    if not nums: return [0.1] * 10
    counts = Counter(nums[-200:])
    total = sum(counts.values())
    base = [counts.get(n, 0) / total if total > 0 else 0.1 for n in range(10)]
    for order in (1, 2):
        if len(nums) <= order: continue
        ctx = tuple(nums[-order:])
        trans = Counter(nums[i] for i in range(order, len(nums)) if tuple(nums[i-order:i]) == ctx)
        tot = sum(trans.values())
        if tot > 5:
            for n in range(10): base[n] = base[n] * 0.6 + (trans.get(n, 0) / tot) * 0.4
    total = sum(base)
    return [x / total for x in base]


def format_prediction(payload: dict[str, Any]) -> str:
    prob = float(payload["p"])
    conf = max(prob, 1.0 - prob) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if payload["side"] == "BIG" else "SMALL"
    marker = "🔴" if side == "BIG" else "🔵"
    pat = payload.get("pattern", "—")
    pat_line = f"\n🧩 <b>PATTERN:</b> <code>{escape(pat)}</code>" if pat != "—" else ""

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"🔢 <b>TOP NUMBER:</b> <code>{int(payload['number'])}</code>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])}{pat_line}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# RUNTIME & MAIN
# ============================================================

class Runtime:
    def __init__(self, db: Database):
        self.db, self.engine = db, Engine(db)
        self.stop, self.last_fetch = asyncio.Event(), 0.0

    async def pause(self, sec: float) -> None:
        try: await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, sec))
        except asyncio.TimeoutError: pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def one(url: str):
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception: return []
        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources: raise RuntimeError("HistoryUnavailable")
        merged = {row["issue"]: row for rows in sources for row in rows}
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        while not self.stop.is_set():
            try:
                self.db.ingest(await self.fetch(session))
                history = self.db.history()
                self.engine.settle(history)
                self.last_fetch = time.time()
                
                if len(history) >= CFG.min_history:
                    issue = str(int(history[-1]["issue"]) + 1)
                    if not self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone():
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute("INSERT INTO predictions VALUES (?,?,?,?,NULL,0)", (issue, time.time(), "PENDING", json.dumps(pred)))
                            self.db.queue("prediction", issue, format_prediction(pred), time.time() + CFG.delivery_ttl)
                failures = 0
            except asyncio.CancelledError: raise
            except Exception as exc:
                failures += 1
                LOG.exception("Worker error")
            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
            if not row:
                await self.pause(0.5)
                continue
            
            delay = backoff
            try:
                url = f"https://api.telegram.org/bot{CFG.token}/{'sendSticker' if row['kind'] == 'sticker' else 'sendMessage'}"
                payload = {"chat_id": CFG.chat, "sticker": row["text"]} if row["kind"] == "sticker" else {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}
                async with session.post(url, json=payload) as res:
                    data = await res.json(content_type=None)
                    if res.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue
                    if data.get("error_code") == 429: delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif data.get("error_code") in (400, 401, 403, 404): delay = 60
            except Exception as e:
                delay = 10
            await self.pause(delay)
            backoff = min(60.0, backoff * 2)

    async def health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded", **self.db.health_stats()})


async def main() -> None:
    CFG.validate()
    db = Database(CFG.db_path)
    runtime = Runtime(db)
    
    app = web.Application()
    app.router.add_get("/", runtime.health)
    runner = web.AppRunner(app)
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError): loop.add_signal_handler(sig, runtime.stop.set)

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [asyncio.create_task(runtime.worker(session)), asyncio.create_task(runtime.sender(session))]
            await asyncio.gather(*tasks)
    finally:
        await runner.cleanup()
        db.close()

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
