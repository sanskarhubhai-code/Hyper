# -*- coding: utf-8 -*-
"""
FTW VIP PREDICTOR — Ultra Advanced v24 (Fixed & Ready)
Python 3.10+
Deps: aiohttp
"""

import asyncio
import contextlib
import html
import json
import logging
import math
import os
import signal
import socket
import sqlite3
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple

import aiohttp
from aiohttp import web


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    # 🔥 Yahan tera original token aur chat id set kar diya hai
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls: tuple = (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "predictor_ultra_v33.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "2.5"))
    timeout: float = 20.0
    max_history: int = 3000 # 🔥 5000 se 3000 kar diya (Timeout fix)
    min_history: int = int(os.getenv("MIN_HISTORY", "160"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    learning_rate: float = 0.04
    meta_lr: float = 0.045
    decay: float = 0.985
    max_pattern: int = 8
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (ULTRA V24)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config")


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor_ultra")


# ─────────────────────────────────────────────────────────────────────────────
# CORE MATH / UTIL
# ─────────────────────────────────────────────────────────────────────────────

def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))

def logit(p, eps=0.02):
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))

def sigmoid(z):
    z = max(-20.0, min(20.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))

def escape(x):
    return html.escape(str(x))

def numeric(value):
    if isinstance(value, bool):
        return None
    s = str(value).strip()
    if s.isdigit() and s.isascii():
        return int(s)
    return None

def run_length(seq):
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x != last:
            break
        n += 1
    return n

def majority(seq):
    if not seq:
        return 0
    c = Counter(seq)
    return c.most_common(1)[0][0]

def weighted_probability(observations):
    if not observations:
        return 0.5, 0.0
    sw = sum(w for _, w in observations)
    sw2 = sum(w * w for _, w in observations)
    effective_n = (sw * sw / sw2) if sw2 else 0.0
    prob = (4.0 + sum(y * w for y, w in observations)) / (8.0 + sw)
    return clip(prob), effective_n

def family_run(seq):
    if not seq:
        return []
    out = []
    current = seq[0]
    count = 1
    for x in seq[1:]:
        if x == current:
            count += 1
        else:
            out.append((current, count))
            current = x
            count = 1
    out.append((current, count))
    return out

def family_signature(seq):
    return tuple((int(v), cnt) for v, cnt in family_run(seq))

def entropy_bits(seq):
    if not seq:
        return 0.0
    c = Counter(seq)
    n = len(seq)
    entr = 0.0
    for freq in c.values():
        p = freq / n
        entr -= p * math.log2(p)
    return entr

def get_regime(seq):
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    rate = flips / max(1, len(tail) - 1)
    if run_length(tail) >= 5:
        return "RUN"
    if rate >= 0.70:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"


# ─────────────────────────────────────────────────────────────────────────────
# HISTORY PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_history(payload):
    obj = payload
    for _ in range(6):
        if isinstance(obj, list):
            break
        if not isinstance(obj, dict):
            return []
        obj = next((obj[k] for k in ("data", "list", "result", "items", "history") if isinstance(obj.get(k), (list, dict))), None)
    if not isinstance(obj, list):
        return []

    records = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))
        if issue is None or number is None or not 0 <= number <= 9:
            continue
        key = str(issue)
        if key in records and records[key]["num"] != number:
            raise ValueError(f"Conflicting duplicate issue: {key}")
        records[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda x: int(x["issue"]))


# ─────────────────────────────────────────────────────────────────────────────
# QUANTUM PATTERN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class QuantumTracker:
    def __init__(self, min_len=3, max_len=8, min_support=12):
        self.min_len = min_len
        self.max_len = max_len
        self.min_support = min_support

    def match_tail_patterns(self, seq):
        n = len(seq)
        if n < 18:
            return []
        results = []

        for length in range(self.max_len, self.min_len - 1, -1):
            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs = []
            for i in range(n - length):
                if family_signature(seq[i:i + length]) == sig:
                    w = 0.5 ** ((n - 1 - (i + length)) / 350.0)
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


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    def __init__(self, path):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA temp_store=MEMORY")
        self.c.executescript("""
            CREATE TABLE IF NOT EXISTS history(issue TEXT PRIMARY KEY, num INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS predictions(issue TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL, payload TEXT NOT NULL, actual INTEGER);
            CREATE TABLE IF NOT EXISTS models(key TEXT PRIMARY KEY, n REAL NOT NULL, loss REAL NOT NULL, inverse_loss REAL NOT NULL, failures INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                issue TEXT,
                text TEXT NOT NULL,
                created REAL NOT NULL,
                expires REAL,
                sent INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS stats(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
        finally:
            self.c.execute("COMMIT" if self.c.in_transaction else "ROLLBACK")

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.c.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value))
        )

    def stat(self, key, default=None):
        row = self.c.execute("SELECT value FROM stats WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_stat(self, key, value):
        self.c.execute(
            "INSERT INTO stats VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value))
        )

    def ingest(self, records):
        with self.transaction():
            for r in records:
                old = self.c.execute("SELECT num FROM history WHERE issue=?", (r["issue"],)).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute("INSERT OR IGNORE INTO history VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN (SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,)
            )

    def history(self):
        rows = self.c.execute("SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER)").fetchall()
        return [{"issue": r["issue"], "num": r["num"], "size": int(r["num"] >= 5)} for r in rows]

    def get_loss_streak(self):
        rows = self.c.execute("""
            SELECT status FROM predictions
            WHERE status IN ('WIN','LOSS')
            ORDER BY created DESC
            LIMIT 8
        """).fetchall()
        streak = 0
        for r in rows:
            if r["status"] == "LOSS":
                streak += 1
            else:
                break
        return streak

    def memory(self, key):
        row = self.c.execute("SELECT * FROM models WHERE key=?", (key,)).fetchone()
        return dict(row) if row else {"n": 0.0, "loss": 0.0, "inverse_loss": 0.0, "failures": 0}

    def learn(self, key, probability, actual):
        m = self.memory(key)
        n = CFG.decay * m["n"] + 1
        loss = CFG.decay * m["loss"] + (probability - actual) ** 2
        inv = CFG.decay * m["inverse_loss"] + (1 - probability - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        failures = m["failures"] + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n, loss=excluded.loss, inverse_loss=excluded.inverse_loss, failures=excluded.failures",
            (key, n, loss, inv, failures)
        )

    def adapt(self, key, probability):
        m = self.memory(key)
        n = m["n"]
        if n < 1:
            return probability, 0.65
        brier, inv = m["loss"] / n, m["inverse_loss"] / n
        flipped = (inv + 0.03 < brier) and (brier > 0.27) and (m["failures"] >= 3)
        eff_loss = inv if flipped else brier
        p = (1 - probability) if flipped else probability
        reliability = clip((0.25 - eff_loss) / 0.12, 0.0, 1.0)
        shrink = (1 - min(n / 50.0, 1.0)) * 0.75 + min(n / 50.0, 1.0) * reliability
        p = 0.5 + (p - 0.5) * shrink
        weight = clip(math.exp(-7 * (eff_loss - 0.22)), 0.18, 2.0)
        if m["failures"] >= 4 and not flipped:
            weight *= 0.55
        return clip(p), weight

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind, issue, text, created, expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires)
        )

    def close(self):
        self.c.close()


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def trend_features(seq):
    if not seq:
        return [0.0] * 18

    def mean(k):
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5

    def ema(k):
        if not seq:
            return 0.5
        alpha = 2.0 / (k + 1)
        val = float(seq[0])
        for x in seq[1:]:
            val = alpha * x + (1.0 - alpha) * val
        return val

    def std(k):
        arr = seq[-k:]
        if len(arr) < 2:
            return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

    tail = seq[-24:]
    alt = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    run_score = (2 * seq[-1] - 1) * min(run_length(seq), 12) / 12.0
    macd = (ema(12) - ema(26)) * 2.0
    short = mean(3)
    mid = mean(10)
    long = mean(24)
    slope = (mean(3) - mean(12))
    slope2 = (mean(8) - mean(24))
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
        slope,
        slope2,
    ]


def logistic(weights, features):
    return sigmoid(sum(w * x for w, x in zip(weights, features)))


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_NAMES = [
    "patterns", "markov1", "markov2", "markov3",
    "run", "regime", "base", "trend", "momentum",
    "reversal", "volatility", "entropy"
]

class RecoveryController:
    def __init__(self):
        self.last_loss_epoch = 0.0
        self.recovery_mode = 0

    def update(self, loss_streak):
        if loss_streak >= 1:
            self.recovery_mode = max(self.recovery_mode, 1)
            self.last_loss_epoch = time.time()
        else:
            self.recovery_mode = 0

    def intensify(self, seq, base_p, experts, loss_streak):
        if loss_streak <= 0:
            return base_p, False

        recent_big = sum(seq[-5:]) / max(1, len(seq[-5:]))
        recent_small = 1.0 - recent_big

        if loss_streak == 1:
            if recent_big > 0.72:
                return 0.28, True
            if recent_small > 0.72:
                return 0.72, True
            return 0.5 + (base_p - 0.5) * 0.7, True

        if recent_big > 0.70:
            return 0.25, True
        if recent_small > 0.70:
            return 0.75, True

        return 0.5 + (base_p - 0.5) * 0.55, True

class Engine:
    def __init__(self, db):
        self.db = db
        self.recovery = RecoveryController()

    def mine_patterns(self, seq):
        if len(seq) < 18:
            return []
        tracker = QuantumTracker(min_len=3, max_len=8, min_support=CFG.min_support)
        results = tracker.match_tail_patterns(seq)
        if not results:
            return []
        out = []
        for item in results[:CFG.max_pattern]:
            out.append((
                item["score"],
                item["p"],
                item["support"],
                item["eff_n"],
                "q:" + "-".join(f"{'B' if v else 'S'}{c}" for v, c in item["sig"])
            ))
        return out

    def mine_experts(self, seq, is_recovery):
        n = len(seq)
        experts = {}

        def weight(idx):
            return 0.5 ** ((n - 1 - idx) / 320.0)

        patterns = self.mine_patterns(seq)
        if patterns:
            weighted = []
            for _, p, _, eff_n, key in patterns:
                adjusted_p, w = self.db.adapt(key, p)
                weighted.append((adjusted_p, w * math.sqrt(max(1.0, eff_n))))
            if weighted:
                num = sum(p * w for p, w in weighted)
                den = sum(w for _, w in weighted)
                if den > 0:
                    experts["patterns"] = clip(num / den)

        for order in (1, 2, 3):
            suffix = tuple(seq[-order:])
            obs = []
            for i in range(order, n - order):
                if tuple(seq[i - order:i]) == suffix:
                    obs.append((seq[i], weight(i)))
            if len(obs) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(obs)[0]

        if n:
            run_at = [1] * n
            for i in range(1, n):
                run_at[i] = run_at[i-1] + 1 if seq[i] == seq[i-1] else 1

            cr = min(run_at[-1], 8)
            lv = seq[-1]
            run_obs = []
            for i in range(24, n):
                if seq[i - 1] == lv and min(run_at[i - 1], 8) == cr:
                    run_obs.append((seq[i], weight(i)))
            if len(run_obs) >= CFG.min_support:
                experts["run"] = weighted_probability(run_obs)[0]

        current_regime = get_regime(seq)
        regime_obs = []
        for i in range(24, n):
            if get_regime(seq[max(0, i - 96):i]) == current_regime:
                regime_obs.append((seq[i], weight(i)))
        if len(regime_obs) >= CFG.min_support:
            experts["regime"] = weighted_probability(regime_obs)[0]

        experts["base"] = weighted_probability([(y, weight(i)) for i, y in enumerate(seq)])[0]

        features = trend_features(seq)
        tw = self.db.get("trend_weights")
        if not isinstance(tw, list) or len(tw) != len(features):
            tw = [0.0] * len(features)
        experts["trend"] = logistic(tw, features)

        tail = seq[-16:]
        if len(tail) > 1:
            mean_t = sum(tail) / len(tail)
            vol = sum((x - mean_t) ** 2 for x in tail) / len(tail)
            experts["volatility"] = clip(0.5 + (0.5 - abs(0.5 - mean_t)) * 0.9 + vol * 0.1)
        else:
            experts["volatility"] = 0.5

        e = entropy_bits(seq[-12:]) / 3.32
        experts["entropy"] = clip(0.5 + (e - 0.5) * 0.6)

        momentum = sum(seq[-4:]) / 4.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)
        experts["reversal"] = 1.0 - experts["momentum"]

        for k, p in list(experts.items()):
            experts[k] = clip(p, 0.05, 0.95)

        return experts, patterns, features, current_regime

    def _predict_number(self, seq_nums):
        ngram_db = self.db.get("num_ngrams", {})
        freq = self.db.get("num_freq", {})
        prior = [4.0 + 20.0 * freq.get(str(i), 0) / max(1, sum(freq.values())) for i in range(10)] if freq else [1.0] * 10
        dist = list(prior)

        for length in range(2, min(10, len(seq_nums)) + 1):
            key = ",".join(str(x) for x in seq_nums[-length:])
            trans = ngram_db.get(key, {})
            total = sum(trans.values())
            if total < 3:
                continue
            w = min(1.0, total / 18.0) * (0.9 ** (length - 2))
            for i in range(10):
                dist[i] = dist[i] * (1.0 - w) + (trans.get(str(i), 0) / total) * w

        if sum(dist) <= 0:
            return 4, 0.10

        best = max(range(10), key=lambda i: dist[i])
        return best, dist[best] / sum(dist)

    def predict(self, history, issue):
        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]

        loss_streak = self.db.get_loss_streak()
        is_recovery = loss_streak > 0
        self.recovery.update(loss_streak)

        experts, patterns, features, current_regime = self.mine_experts(seq, is_recovery)

        adjusted = []
        for name, p in experts.items():
            corrected, weight = self.db.adapt(f"expert:{name}:{current_regime}", p)
            if name == "patterns":
                weight *= 3.2
            if name in ("momentum", "run", "trend"):
                if loss_streak >= 2:
                    weight *= 0.32
            if name == "reversal" and loss_streak >= 1:
                weight *= 2.7
            if loss_streak == 1 and name in ("momentum", "run"):
                weight *= 1.6
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5

        m_feats = [logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES] + [1.0]
        mw = self.db.get("meta_weights")
        if not isinstance(mw, list) or len(mw) != len(m_feats):
            mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
        p_meta = sigmoid(sum(w * f for w, f in zip(mw, m_feats)))

        alpha = min(self.db.get("meta_n", 0) / 100.0, 1.0)
        p_raw = alpha * p_meta + (1.0 - alpha) * p_avg

        if loss_streak >= 1:
            if loss_streak == 1:
                recent_bias = sum(seq[-3:]) / 3.0 if len(seq) >= 3 else 0.5
                if recent_bias > 0.68:
                    p_raw = 0.28
                elif recent_bias < 0.32:
                    p_raw = 0.72
                else:
                    p_raw = 0.5 + (p_raw - 0.5) * 0.7
            elif loss_streak >= 2:
                recent_bias = sum(seq[-4:]) / 4.0 if len(seq) >= 4 else 0.5
                if recent_bias > 0.7:
                    p_raw = 0.22
                elif recent_bias < 0.3:
                    p_raw = 0.78
                else:
                    p_raw = 0.5 + (p_raw - 0.5) * 0.5

        if len(seq) >= 8:
            short = sum(seq[-6:]) / 6.0
            if short > 0.80 and p_raw > 0.65:
                p_raw = 0.65
            if short < 0.20 and p_raw < 0.35:
                p_raw = 0.35

        bucket = min(19, int(p_raw * 20))
        n_g, s_g = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((30 * p_raw + s_g) / (30 + n_g))

        pred_side = "BIG" if p_cal >= 0.5 else "SMALL"
        valid_nums = range(5, 10) if pred_side == "BIG" else range(0, 5)

        num_pred, num_conf = self._predict_number(nums)
        if num_pred not in valid_nums:
            num_pred = max(valid_nums, key=lambda x: (x if x in valid_nums else -1))

        prev_regime = get_regime(seq[:-1]) if len(seq) > 1 else "N/A"
        trend_shift = f"{prev_regime} ➡️ {current_regime}" if prev_regime != current_regime else None

        payload = {
            "issue": issue,
            "p": p_cal,
            "p_raw": p_raw,
            "side": pred_side,
            "num": num_pred,
            "num_probability": num_conf,
            "regime": current_regime,
            "source": "🎯 Ultra Quantum Engine",
            "trend_shift": trend_shift,
            "pattern_str": _pattern_str(patterns),
            "edge": abs(p_cal - 0.5),
            "confidence": max(p_cal, 1.0 - p_cal),
            "experts": experts,
            "features": features,
            "meta_feats": m_feats,
            "is_recovery": is_recovery,
            "loss_streak": loss_streak,
            "patterns": [{"key": k, "p": p, "support": s} for _, p, s, _, k in patterns],
        }

        return payload

    def settle(self, history):
        lookup = {r["issue"]: r for r in history}
        if history:
            sig = f"{history[0]['issue']}-{history[-1]['issue']}-{len(history)}"
            if self.db.get("ngram_sig") != sig:
                num_ngrams, num_freq = {}, {}
                for i in range(1, len(history)):
                    curr = history[i]["num"]
                    num_freq[str(curr)] = num_freq.get(str(curr), 0) + 1
                    for length in range(2, 11):
                        if i >= length:
                            seq_key = ",".join(str(history[j]["num"]) for j in range(i - length, i))
                            num_ngrams.setdefault(seq_key, {})
                            num_ngrams[seq_key][str(curr)] = num_ngrams[seq_key].get(str(curr), 0) + 1
                self.db.set("num_ngrams", num_ngrams)
                self.db.set("num_freq", num_freq)
                self.db.set("ngram_sig", sig)

        pending = self.db.c.execute(
            "SELECT issue,payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)"
        ).fetchall()

        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None:
                continue

            payload = json.loads(row["payload"])
            y = actual["size"]
            won_side = int(payload["p"] >= 0.5) == y
            won_num = payload.get("num") == actual["num"]

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=? WHERE issue=?",
                    ("WIN" if won_side else "LOSS", actual["num"], row["issue"])
                )

                for name, val in payload["experts"].items():
                    self.db.learn(f"expert:{name}:{payload['regime']}", val, y)

                for pat in payload["patterns"]:
                    self.db.learn(pat["key"], pat["p"], y)

                if payload.get("features") and payload["experts"].get("trend"):
                    err = y - payload["experts"]["trend"]
                    tw = self.db.get("trend_weights") or [0.0] * len(payload["features"])
                    self.db.set("trend_weights", [
                        clip(w + CFG.learning_rate * (err * x - 0.015 * w), -4, 4)
                        for w, x in zip(tw, payload["features"])
                    ])

                target = len(EXPERT_NAMES) + 1
                if payload.get("meta_feats"):
                    feats = payload["meta_feats"]
                    if len(feats) == target:
                        mw = self.db.get("meta_weights") or [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
                        mp = sigmoid(sum(w * f for w, f in zip(mw, feats)))
                        grad = 2.0 * (y - mp) * mp * (1.0 - mp)
                        self.db.set("meta_weights", [
                            clip(w * 0.999 + CFG.meta_lr * grad * f, -4, 4)
                            for w, f in zip(mw, feats)
                        ])
                        self.db.set("meta_n", self.db.get("meta_n", 0) + 1)

                bucket = min(19, int(payload.get("p_raw", payload["p"]) * 20))
                n, s = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
                self.db.set(f"cal20:global:{bucket}", [0.995 * n + 1, 0.995 * s + y])

                sticker = CFG.jack_sticker if (won_side and won_num) else CFG.win_sticker if won_side else CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)
                self.db.c.execute("UPDATE outbox SET sent=2 WHERE kind='prediction' AND issue=? AND sent=0", (row["issue"],))


def _pattern_str(patterns):
    if not patterns:
        return "None"
    top = patterns[0]
    key = top[4].split(":")[-1] if ":" in top[4] else top[4]
    return f"<code>{key}</code> {'⬆️' if top[1] >= 0.5 else '⬇️'}"


def format_prediction(p):
    conf = max(p["p"], 1.0 - p["p"]) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if p["side"] == "BIG" else "SMALL"
    quality = "🔥 HIGH VIP" if conf >= 65 else "⭐ VIP" if conf >= 60 else "✅ NORMAL"
    stake = "2.0x" if conf >= 65 else "1.5x" if conf >= 60 else "1x"

    rec_tag = ""
    if p.get("is_recovery"):
        if p.get("loss_streak") == 1:
            rec_tag = "⚠️ <b>RECOVERY LVL 1</b> ⚠️\n"
        else:
            rec_tag = "🚨 <b>ULTRA RECOVERY LVL 2</b> 🚨\n"

    shift = f"\n🔄 <b>TREND SHIFT:</b> {escape(p['trend_shift'])}" if p.get("trend_shift") else ""
    pat = f"\n🧩 <b>PATTERN:</b> {p.get('pattern_str', 'None')}" if p.get("pattern_str") and p.get("pattern_str") != "None" else ""

    return (
        f"{rec_tag}⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {'🔴' if side == 'BIG' else '🔵'} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake})\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime', 'N/A'))}\n"
        f"🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}{shift}{pat}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME
# ─────────────────────────────────────────────────────────────────────────────

class Runtime:
    def __init__(self, db):
        self.db = db
        self.engine = Engine(db)
        self.stop = asyncio.Event()
        self.last_fetch = 0.0
        self.error = None

    async def pause(self, seconds):
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, seconds))
        except asyncio.TimeoutError:
            pass

    async def fetch(self, session):
        async def one(url):
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception:
                return []

        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources:
            raise RuntimeError("HistoryUnavailable")
        merged = {}
        for rows in sources:
            for row in rows:
                if row["issue"] in merged and merged[row["issue"]]["num"] != row["num"]:
                    raise ValueError("SourceConflict")
                merged[row["issue"]] = row
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session):
        failures = 0
        while not self.stop.is_set():
            try:
                records = await self.fetch(session)
                self.db.ingest(records)
                history = self.db.history()
                self.last_fetch = time.time()
                self.error = None
                self.engine.settle(history)

                latest = int(history[-1]["issue"])
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    if not self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone():
                        p = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", json.dumps(p))
                            )
                            self.db.queue("prediction", issue, format_prediction(p), time.time() + CFG.delivery_ttl)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = type(exc).__name__
                failures += 1
                LOG.error("Worker failure: %s", self.error)
            await self.pause(min(60, CFG.fetch_every * 2 ** min(failures, 4)))

    async def sender(self, session):
        endpoint_msg = f"https://api.telegram.org/bot{CFG.token}/sendMessage"
        endpoint_sticker = f"https://api.telegram.org/bot{CFG.token}/sendSticker"
        backoff = 1.0

        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
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
                    LOG.error("Telegram API Error: %s", data)
                    if code == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404):
                        delay = 60
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning("Telegram Network Fail: %s", e)
                delay = 10

            await self.pause(delay)
            backoff = min(60.0, backoff * 2)

    async def health(self, request):
        total = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        wins = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0]
        acc = (wins / total * 100) if total else 0.0
        return web.json_response({
            "status": "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded",
            "history": self.db.c.execute("SELECT COUNT(*) FROM history").fetchone()[0],
            "settled": total,
            "wins": wins,
            "win_rate": f"{acc:.1f}%"
        })


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
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
    tasks, stop_task = [], None

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300, ssl=False, family=socket.AF_INET)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session), name="worker"),
                asyncio.create_task(runtime.sender(session), name="sender"),
            ]
            stop_task = asyncio.create_task(runtime.stop.wait())
            LOG.info(f"{CFG.name} Started")
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
    asyncio.run(main())
