# bot.py — FTW VIP PREDICTOR (HYPER V18 — Ultra Recovery Engine)
# Python 3.10+ | aiohttp

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
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import aiohttp
from aiohttp import web

# -----------------------------
# CONFIG
# -----------------------------
@dataclass(frozen=True)
class Config:
    token: str = os.getenv("BOT_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("CHAT_ID", "-1004485961768")
    urls: tuple = tuple(
        x.strip()
        for x in os.getenv("HISTORY_URLS", "").split(",")
        if x.strip()
    ) or (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "predictor_hyper_v18.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 4000
    min_history: int = int(os.getenv("MIN_HISTORY", "120"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    learning_rate: float = 0.04
    meta_lr: float = 0.035
    decay: float = 0.985
    max_pattern: int = 18
    delivery_ttl: float = 45.0
    name: str = "FTW VIP PREDICTOR (HYPER V18)"
    min_edge: float = float(os.getenv("MIN_EDGE", "0.03"))
    recovery_min_edge: float = float(os.getenv("RECOVERY_MIN_EDGE", "0.015"))
    win_sticker: str = os.getenv("WIN_STICKER", "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E")
    loss_sticker: str = os.getenv("LOSS_STICKER", "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E")
    jack_sticker: str = os.getenv("JACK_STICKER", "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E")

    def validate(self):
        if not self.token or self.token == "REPLACE_ME":
            raise ValueError("BOT_TOKEN is missing or invalid")
        if not self.chat or self.chat == "-1000000000000":
            raise ValueError("CHAT_ID is missing or invalid")
        if not self.urls:
            raise ValueError("History URLs are missing")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor")

# -----------------------------
# UTILS
# -----------------------------
def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))

def logit(p, eps=0.02):
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1 - p))

def sigmoid(z):
    z = max(-20.0, min(20.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))

def escape(x):
    return html.escape(str(x))

def numeric(value):
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    return int(text) if text.isascii() and text.isdigit() else None

def run_length(seq):
    if not seq:
        return 0
    n = 0
    last = seq[-1]
    for x in reversed(seq):
        if x != last:
            break
        n += 1
    return n

def get_regime(seq):
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    rate = flips / max(1, len(tail) - 1)
    if run_length(tail) >= 4:
        return "RUN"
    if rate >= 0.70:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"

def weighted_probability(observations):
    sw = sum(w for _, w in observations)
    sw2 = sum(w * w for _, w in observations)
    effective_n = sw * sw / sw2 if sw2 else 0.0
    if sw <= 0:
        return 0.5, 0.0
    probability = (4 + sum(y * w for y, w in observations)) / (8 + sw)
    return probability, effective_n

def trend_features(seq):
    if not seq:
        return [0.0] * 20

    def mean(k):
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5

    def ema(k):
        if not seq:
            return 0.5
        alpha = 2.0 / (k + 1)
        val = float(seq[0])
        for x in seq[1:]:
            val = alpha * x + (1 - alpha) * val
        return val

    def rsi(period=14):
        if len(seq) < period + 1:
            return 50.0
        gains = 0.0
        losses = 0.0
        for i in range(-period, 0):
            change = seq[i] - seq[i - 1]
            if change > 0:
                gains += change
            else:
                losses -= change
        if losses == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + gains / losses))

    def std(k):
        arr = seq[-k:]
        if len(arr) < 2:
            return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

    tail = seq[-24:]
    alternation = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    signed_run = (2 * seq[-1] - 1) * min(run_length(seq), 10) / 10.0
    rsi_n = (rsi(14) / 100.0) * 2.0 - 1.0
    macd = (ema(12) - ema(26)) * 2.0

    autocorr = 0.0
    if len(seq) >= 25:
        m24 = mean(24)
        s24 = std(24)
        if s24 > 0:
            num = sum((seq[i] - m24) * (seq[i - 1] - m24) for i in range(-23, 0))
            den = 24.0 * s24 ** 2
            autocorr = num / den if den > 0 else 0.0

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
        signed_run,
        2 * seq[-1] - 1,
        (2 * alternation - 1) * (2 * seq[-1] - 1),
        rsi_n,
        macd,
        (2 * std(10) - 1),
        (2 * std(24) - 1),
        (sum(seq[-10:]) / 10.0) - 0.5,
        (sum(seq[-5:]) / 5.0) - 0.5,
        autocorr,
    ]

EXPERT_NAMES = ["patterns", "markov1", "markov2", "markov3", "markov4",
                "run", "regime", "base", "neighbors", "trend", "momentum", "cyclic"]

def logistic(weights, features):
    return sigmoid(sum(w * x for w, x in zip(weights, features)))

# -----------------------------
# HISTORY PARSER
# -----------------------------
def parse_history(payload):
    obj = payload
    for _ in range(6):
        if isinstance(obj, list):
            break
        if not isinstance(obj, dict):
            return []
        obj = next(
            (obj[k] for k in ("data", "list", "result", "items", "history")
             if isinstance(obj.get(k), (list, dict))),
            None
        )
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
        prev = records.get(key)
        if prev is not None and prev["num"] != number:
            raise ValueError(f"Conflicting duplicate: {key}")
        records[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda x: int(x["issue"]))

# -----------------------------
# DATABASE
# -----------------------------
class Database:
    def __init__(self, path):
        self.c = sqlite3.connect(path, isolation_level=None, timeout=20)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
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
        """)

        self.c.execute("""
            CREATE TABLE IF NOT EXISTS state(
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        self.set_value("meta_n", "0")
        self.set_value("last_result", '"NONE"')
        self.set_value("consecutive_losses", "0")
        self.set_value("recovery_mode", "0")

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.c.execute("COMMIT")
        except BaseException:
            self.c.execute("ROLLBACK")
            raise

    def set_value(self, key, value):
        self.c.execute(
            "INSERT INTO state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value)
        )

    def get_value(self, key, default=None):
        row = self.c.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def set_meta(self, key, value):
        self.c.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value))
        )

    def get_meta(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def ingest(self, records):
        with self.transaction():
            for r in records:
                old = self.c.execute("SELECT num FROM history WHERE issue=?", (r["issue"],)).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute("INSERT OR IGNORE INTO history VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN "
                "(SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,)
            )

    def history(self):
        rows = self.c.execute("SELECT issue, num FROM history ORDER BY CAST(issue AS INTEGER)").fetchall()
        return [{"issue": r["issue"], "num": r["num"], "size": int(r["num"] >= 5)} for r in rows]

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
            return probability, 0.65, False
        brier = m["loss"] / n
        inv = m["inverse_loss"] / n
        flipped = (inv + 0.02 < brier) and (brier > 0.24) and (m["failures"] >= 3)
        effective_loss = inv if flipped else brier
        p = (1 - probability) if flipped else probability

        reliability = clip((0.25 - effective_loss) / 0.10, 0.0, 1.0)
        maturity = min(n / 40.0, 1.0)
        shrink = (1 - maturity) * 0.65 + maturity * reliability
        p = 0.5 + (p - 0.5) * shrink

        weight = clip(math.exp(-6 * (effective_loss - 0.22)), 0.20, 2.0)
        if m["failures"] >= 4 and not flipped:
            weight *= 0.5
        return clip(p), weight, flipped

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind, issue, text, created, expires) VALUES (?, ?, ?, ?, ?)",
            (kind, issue, text, time.time(), expires)
        )

    def update_recovery_state(self, settled_result):
        last_result = self.get_value("last_result", "NONE")
        consecutive_losses = int(self.get_value("consecutive_losses", 0))

        if settled_result == "LOSS":
            consecutive_losses += 1
        elif settled_result in ("WIN", "JACKPOT"):
            consecutive_losses = 0

        # Recovery immediately after first loss
        recovery_mode = 1 if consecutive_losses >= 1 else 0

        self.set_value("last_result", json.dumps(settled_result))
        self.set_value("consecutive_losses", str(consecutive_losses))
        self.set_value("recovery_mode", str(recovery_mode))

    def get_recovery_state(self):
        return {
            "last_result": self.get_value("last_result", "NONE"),
            "consecutive_losses": int(self.get_value("consecutive_losses", 0)),
            "recovery_mode": int(self.get_value("recovery_mode", 0)),
        }

# -----------------------------
# ENGINE
# -----------------------------
class Engine:
    def __init__(self, db):
        self.db = db

    def get_runs(self, seq):
        if not seq:
            return []
        runs = []
        curr = seq[0]
        length = 1
        for x in seq[1:]:
            if x == curr:
                length += 1
            else:
                runs.append((curr, length))
                curr = x
                length = 1
        runs.append((curr, length))
        return runs

    def mine_family_patterns(self, seq):
        if not seq:
            return []
        runs = self.get_runs(seq)
        if len(runs) < 2:
            return []
        for k in range(min(8, len(runs)), 1, -1):
            target = runs[-k:]
            observations = []
            for i in range(len(runs) - k - 1, -1, -1):
                if runs[i:i + k] == target and i + k < len(runs):
                    observations.append((runs[i + k][0], 1.0))
            if observations:
                p, eff_n = weighted_probability(observations)
                bits = "-".join(f"{'B' if v else 'S'}{c}" for v, c in target)
                key = f"family:{bits}"
                return [(abs(p - 0.5) * math.sqrt(eff_n), p, len(observations), eff_n, key)]
        return []

    def mine_patterns(self, seq):
        n = len(seq)
        fam = self.mine_family_patterns(seq)
        if fam:
            return fam

        candidates = {}

        def weight(idx):
            return 0.5 ** ((n - 1 - idx) / 300.0)

        for length in range(5, min(CFG.max_pattern, n // 2) + 1):
            if n < length * 2:
                continue
            suffix = tuple(seq[-length:])
            obs = []
            last = -length - 1
            for i in range(length, n - length):
                if tuple(seq[i - length:i]) == suffix and i - last >= length + 1:
                    obs.append((seq[i], weight(i)))
                    last = i
            if len(obs) >= CFG.min_support:
                p, eff_n = weighted_probability(obs)
                if eff_n >= CFG.min_support * 0.5:
                    bits = "".join("B" if x else "S" for x in suffix)
                    candidates[f"exact:{bits}"] = (
                        abs(p - 0.5) * math.sqrt(eff_n),
                        p,
                        len(obs),
                        eff_n,
                        f"exact:{bits}",
                    )
        return sorted(candidates.values(), key=lambda x: x[0], reverse=True)[:10]

    def mine_experts(self, seq):
        experts = {}
        n = len(seq)

        patterns = self.mine_patterns(seq)
        if patterns:
            num = 0.0
            den = 0.0
            for _, p, _, eff_n, key in patterns:
                adjusted, w, _ = self.db.adapt(key, p)
                num += adjusted * (w * math.sqrt(eff_n))
                den += (w * math.sqrt(eff_n))
            if den > 0:
                experts["patterns"] = num / den

        for order in (1, 2, 3, 4):
            suffix = tuple(seq[-order:])
            obs = []
            for i in range(order, n - order):
                if tuple(seq[i - order:i]) == suffix:
                    obs.append((seq[i], 0.5 ** ((n - 1 - i) / 300.0)))
            if len(obs) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(obs)[0]

        if n:
            run_at = [1] * n
            for i in range(1, n):
                run_at[i] = run_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
            last_val = seq[-1]
            current_run = run_at[-1]
            run_obs = []
            for i in range(24, n):
                if seq[i - 1] == last_val and min(run_at[i - 1], 8) == current_run:
                    run_obs.append((seq[i], 0.5 ** ((n - 1 - i) / 300.0)))
            if len(run_obs) >= CFG.min_support:
                experts["run"] = weighted_probability(run_obs)[0]

        current_regime = get_regime(seq)
        regime_obs = []
        for i in range(24, n):
            if get_regime(seq[max(0, i - 96): i]) == current_regime:
                regime_obs.append((seq[i], 0.5 ** ((n - 1 - i) / 300.0)))
        if len(regime_obs) >= CFG.min_support:
            experts["regime"] = weighted_probability(regime_obs)[0]

        experts["base"] = weighted_probability([(y, 0.5 ** idx) for idx, y in enumerate(seq)])[0] if seq else 0.5

        nearest = []
        for i in range(10, n - 10):
            diff = sum(a != b for a, b in zip(seq[i - 10:i], seq[-10:]))
            if diff <= 2:
                nearest.append((seq[i], 0.5 ** ((n - 1 - i) / 300.0) * (0.4 ** diff)))
        if len(nearest) >= 20:
            experts["neighbors"] = weighted_probability(nearest)[0]

        features = trend_features(seq)
        tw = self.db.get_meta("trend_weights")
        if not isinstance(tw, list) or len(tw) != len(features):
            tw = [0.0] * len(features)
        experts["trend"] = logistic(tw, features)

        momentum = sum(seq[-5:]) / 5.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)

        cyc = 0.5
        if len(seq) >= 8:
            diff = abs(sum(seq[-4:]) - sum(seq[-8:-4]))
            cyc = clip(0.5 + diff / 8.0, 0.05, 0.95)
        experts["cyclic"] = cyc

        return experts, patterns, features, current_regime

    def _predict_number(self, seq_nums):
        freq = self.db.get_meta("num_freq", {})
        ngram_db = self.db.get_meta("num_ngrams", {})
        dist = [1.0] * 10

        if freq:
            total = sum(freq.values())
            if total > 0:
                base = [1.0 + 5.0 * (freq.get(str(i), 0) / total) for i in range(10)]
                dist = base[:]

        for length in range(min(10, len(seq_nums)), 1, -1):
            seq_key = ",".join(str(x) for x in seq_nums[-length:])
            trans = ngram_db.get(seq_key)
            if not trans:
                continue
            total = sum(trans.values())
            if total < 3:
                continue
            weight = min(1.0, total / 22.0) * (0.9 ** (length - 2))
            for i in range(10):
                dist[i] = dist[i] * (1 - weight) + (trans.get(str(i), 0) / total) * weight

        total = sum(dist)
        if total <= 0:
            return 4, 0.10

        probs = [x / total for x in dist]
        best = max(range(10), key=lambda i: probs[i])
        return best, probs[best]

    def predict(self, history, issue):
        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]

        experts, patterns, features, current_regime = self.mine_experts(seq)

        recovery = self.db.get_recovery_state()
        loss_streak = max(0, int(recovery["consecutive_losses"]))
        recovery_mode = bool(recovery["recovery_mode"]) or loss_streak > 0

        rec_boost = 1.0 + (0.30 if recovery_mode else 0.0) + (0.14 * min(loss_streak, 6))

        adjusted = []
        for name, p in experts.items():
            corrected, weight, _ = self.db.adapt(f"expert:{name}:{current_regime}", p)
            if name == "patterns":
                weight *= 2.8
            if name in ("momentum", "run", "trend"):
                weight *= 1.8
            if recovery_mode:
                weight *= rec_boost
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5

        m_feats = [logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES]

        # Extra deep recovery features
        if seq:
            tail_big = sum(seq[-10:]) / max(1, len(seq[-10:]))
            momentum_shift = 0.5 + ((sum(seq[-5:]) / 5.0) - 0.5) * 0.8
            m_feats += [
                logit(clip(0.5 + (tail_big - 0.5) * 0.9, 0.05, 0.95)),
                logit(clip(momentum_shift, 0.05, 0.95)),
            ]
        else:
            m_feats += [0.0, 0.0]

        mw = self.db.get_meta("meta_weights")
        if not isinstance(mw, list) or len(mw) != len(m_feats):
            mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.5, 0.0, 0.5, 0.0]

        p_meta = sigmoid(sum(w * f for w, f in zip(mw, m_feats)))

        alpha = min(self.db.get_meta("meta_n", 0) / 100.0, 1.0)
        p_raw = alpha * p_meta + (1 - alpha) * p_avg

        if recovery_mode:
            p_raw = clip(p_raw + 0.12 * min(loss_streak + 1, 5) + 0.03, 0.05, 0.95)

        if len(seq) >= 8:
            last = seq[-1]
            if get_regime(seq[:-1]) != current_regime:
                p_raw = clip(p_raw + 0.04 if last == 1 else -0.04, 0.05, 0.95)

        bucket = min(19, max(0, int(p_raw * 20)))
        n_g, s_g = self.db.get_meta(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((35 * p_raw + s_g) / (35 + n_g))

        side = "BIG" if p_cal >= 0.5 else "SMALL"
        valid_nums = list(range(5, 10)) if side == "BIG" else list(range(0, 5))

        num_dist = [1.0] * 10
        if nums:
            for length in range(min(10, len(nums)), 1, -1):
                seq_key = ",".join(str(x) for x in nums[-length:])
                trans = self.db.get_meta("num_ngrams", {}).get(seq_key)
                if trans:
                    total = sum(trans.values())
                    if total >= 3:
                        w = min(1.0, total / 18.0) * (0.95 ** (length - 2))
                        for i in range(10):
                            num_dist[i] *= (1 - w) + (trans.get(str(i), 0) / total) * w

        total_dist = sum(num_dist)
        if total_dist <= 0:
            num_dist = [1.0] * 10
            total_dist = 10.0

        probs = [x / total_dist for x in num_dist]
        choices = [i for i in valid_nums if probs[i] > 0]
        num_pred = max(choices, key=lambda x: probs[x]) if choices else (9 if side == "BIG" else 4)

        edge = abs(p_cal - 0.5)
        confidence = max(p_cal, 1 - p_cal)
        if recovery_mode:
            confidence = min(0.99, confidence + 0.05)
            edge = max(edge, 0.12)

        return {
            "issue": issue,
            "p": clip(p_cal, 0.02, 0.98),
            "p_raw": clip(p_raw, 0.02, 0.98),
            "side": side,
            "num": num_pred,
            "edge": edge,
            "confidence": confidence,
            "regime": current_regime,
            "source": "🧠 Recovery Ensemble" if recovery_mode else "📊 Statistical Ensemble",
            "patterns": [{"key": k, "p": p, "support": s} for _, p, s, _, k in patterns],
            "experts": experts,
            "features": features,
            "meta_feats": m_feats,
            "recovery_mode": recovery_mode,
        }

# -----------------------------
# PATTERN STRING
# -----------------------------
def _pattern_str(patterns):
    if not patterns:
        return "None"
    top = patterns[0]

    if isinstance(top, dict):
        key = top.get("key", "")
        p_val = top.get("p", 0.5)
    else:
        key = top[4]
        p_val = top[1]

    bits = key.split(":")[-1]
    arrow = "⬆️" if p_val >= 0.5 else "⬇️"
    return f"<code>{bits}</code> {arrow}"

# -----------------------------
# FORMATTER
# -----------------------------
def format_prediction(p):
    conf = max(p["p"], 1 - p["p"]) * 100
    filled = min(10, max(1, int(conf // 10)))
    bar = "█" * filled + "░" * (10 - filled)
    side = p["side"]
    color = "🔴" if side == "BIG" else "🔵"
    quality = "🔥 HIGH VIP" if conf >= 65 else "⭐ VIP" if conf >= 60 else "✅ NORMAL"
    stake = "2x" if conf >= 70 else "1.5x" if conf >= 65 else "1x" if conf >= 60 else "0.5x"

    shift = ""
    if p.get("regime"):
        shift = f"\n🌊 <b>REGIME:</b> {escape(p['regime'])}"

    pattern = p.get("pattern_str") or _pattern_str(p.get("patterns", []))
    pattern_text = f"\n🧩 <b>PATTERN:</b> {pattern}" if pattern and pattern != "None" else ""

    rec_txt = ""
    if p.get("recovery_mode"):
        rec_txt = "\n⚡ <b>RECOVERY MODE:</b> ACTIVE"

    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {color} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake})\n"
        f"📐 <b>EDGE:</b> {p.get('edge', 0) * 100:.1f}%{rec_txt}{shift}{pattern_text}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )

# -----------------------------
# RUNTIME
# -----------------------------
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
                async with session.get(url, timeout=CFG.timeout) as r:
                    r.raise_for_status()
                    payload = await r.json(content_type=None)
                    return parse_history(payload)
            except Exception as exc:
                LOG.warning("Fetch error from %s: %s", url, exc)
                return []
        sources = [x for x in await asyncio.gather(*(one(u) for u in CFG.urls)) if x]
        if not sources:
            raise RuntimeError("HistoryUnavailable")
        merged = {}
        for rows in sources:
            for row in rows:
                issue = row["issue"]
                if issue in merged and merged[issue]["num"] != row["num"]:
                    raise ValueError(f"SourceConflict:{issue}")
                merged[issue] = row
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
                failures = 0

                # settle as soon as valid data arrives
                settle_history(self.db, history)

                # Build number ngrams
                num_ngrams = {}
                num_freq = {}
                for i in range(1, len(history)):
                    curr = history[i]["num"]
                    num_freq[str(curr)] = num_freq.get(str(curr), 0) + 1
                    for length in range(2, min(10, i) + 1):
                        seq_key = ",".join(str(history[j]["num"]) for j in range(i - length, i))
                        num_ngrams.setdefault(seq_key, {})
                        num_ngrams[seq_key][str(curr)] = num_ngrams[seq_key].get(str(curr), 0) + 1

                self.db.set_meta("num_ngrams", num_ngrams)
                self.db.set_meta("num_freq", num_freq)

                if history:
                    latest = int(history[-1]["issue"])
                    self.db.set_meta("latest_seen", latest)

                    if len(history) >= CFG.min_history:
                        issue = str(latest + 1)
                        exists = self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone()
                        if not exists:
                            p = self.engine.predict(history, issue)
                            payload = json.dumps(p)
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", payload)
                            )
                            self.db.queue("prediction", issue, format_prediction(p), time.time() + CFG.delivery_ttl)

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = type(exc).__name__
                failures += 1
                LOG.error("Worker failure: %s", exc)

            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

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

            try:
                if row["kind"] == "sticker":
                    url = endpoint_sticker
                    payload = {"chat_id": CFG.chat, "sticker": row["text"]}
                else:
                    url = endpoint_msg
                    payload = {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}

                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)
                    if response.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue

                    code = data.get("error_code", response.status)
                    if code == 429:
                        retry_after = float(data.get("parameters", {}).get("retry_after", 10))
                        await self.pause(retry_after)
                    elif code in (400, 401, 403, 404):
                        LOG.error("Telegram API error %s: %s", code, data)
                        self.db.c.execute("UPDATE outbox SET sent=2 WHERE id=?", (row["id"],))
                        await self.pause(60)
                    else:
                        await self.pause(min(30, backoff))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("Telegram send failed: %s", exc)
                await self.pause(min(30, backoff))

            backoff = min(60.0, backoff * 2)

    async def health(self, request):
        total = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        wins = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0]
        skips = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='SKIP'").fetchone()[0]
        acc = (wins / total * 100) if total else 0.0
        return web.json_response({
            "status": "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded",
            "history": self.db.c.execute("SELECT COUNT(*) FROM history").fetchone()[0],
            "settled": total,
            "wins": wins,
            "win_rate": f"{acc:.1f}%",
            "skipped": skips,
            "recovery_mode": self.db.get_recovery_state()["recovery_mode"],
        })

# -----------------------------
# SETTLEMENT LOGIC
# -----------------------------
def settle_history(db, history):
    pending = db.c.execute(
        "SELECT issue, payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)"
    ).fetchall()

    lookup = {r["issue"]: r for r in history}

    for row in pending:
        issue = row["issue"]
        actual = lookup.get(issue)
        if actual is None:
            continue

        payload = json.loads(row["payload"])
        actual_num = actual["num"]
        actual_side = "BIG" if actual["size"] == 1 else "SMALL"

        won_side = payload["side"] == actual_side
        won_num = payload["num"] == actual_num

        final_status = "WIN" if (won_side and won_num) else ("WIN" if won_side else "LOSS")
        with db.transaction():
            db.c.execute(
                "UPDATE predictions SET status=?, actual=? WHERE issue=?",
                (final_status, actual_num, issue)
            )

            for name, val in payload.get("experts", {}).items():
                db.learn(f"expert:{name}:{payload.get('regime', 'N/A')}", val, int(actual["size"]))

            for pat in payload.get("patterns", []):
                db.learn(pat["key"], pat["p"], int(actual["size"]))

            features = payload.get("features", [])
            if features and payload.get("experts", {}).get("trend") is not None:
                err = int(actual["size"]) - payload["experts"]["trend"]
                tw = db.get_meta("trend_weights") or [0.0] * len(features)
                db.set_meta("trend_weights",
                    [clip(w + CFG.learning_rate * (err * x - 0.01 * w), -4, 4)
                     for w, x in zip(tw, features)]
                )

            meta_feats = payload.get("meta_feats", [])
            mw = db.get_meta("meta_weights")
            if not isinstance(mw, list) or len(mw) != len(meta_feats):
                mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.5, 0.0, 0.5, 0.0]
            if len(meta_feats) == len(mw):
                target = int(actual["size"])
                mp = sigmoid(sum(w * f for w, f in zip(mw, meta_feats)))
                grad = 2.0 * (target - mp) * mp * (1.0 - mp)
                db.set_meta("meta_weights",
                    [clip(w + CFG.meta_lr * grad * f, -4, 4) for w, f in zip(mw, meta_feats)]
                )
                db.set_meta("meta_n", int(db.get_meta("meta_n", 0)) + 1)

            bucket = min(19, int(payload.get("p_raw", payload["p"]) * 20))
            n, s = db.get_meta(f"cal20:global:{bucket}", [0.0, 0.0])
            db.set_meta(f"cal20:global:{bucket}", [0.995 * n + 1, 0.995 * s + int(actual["size"])])

            if won_side and won_num:
                db.update_recovery_state("WIN")
                sticker = CFG.jack_sticker
            elif won_side:
                db.update_recovery_state("WIN")
                sticker = CFG.win_sticker
            else:
                db.update_recovery_state("LOSS")
                sticker = CFG.loss_sticker

            db.queue("sticker", issue, sticker)
            db.c.execute("UPDATE outbox SET sent=2 WHERE kind='prediction' AND issue=? AND sent=0", (issue,))

# -----------------------------
# MAIN
# -----------------------------
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
    tasks = []
    stop_task = None

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

            LOG.info("%s V18 Started (Ultra Recovery Engine enabled)", CFG.name)

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
        db.c.close()

if __name__ == "__main__":
    asyncio.run(main())
