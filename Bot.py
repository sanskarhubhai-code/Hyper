
# FTW PREDICTOR — FIXED FINAL VERSION
# Python 3.10+
# Deps: aiohttp

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

# ============================================================
# CONFIG
# ============================================================

@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls: tuple = (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "predictor_fixed_final.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "2.0"))
    timeout: float = 18.0
    max_history: int = 12000
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    learning_rate: float = 0.058
    meta_lr: float = 0.07
    decay: float = 0.985
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (FIXED FINAL)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ftw_fixed_final")


# ============================================================
# MATH / HELPERS
# ============================================================

def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))

def sigmoid(z):
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))

def logit(p, eps=0.02):
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))

def escape(x):
    return html.escape(str(x))

def numeric(value):
    if isinstance(value, bool):
        return None
    s = str(value).strip()
    if s.isascii() and s.isdigit():
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

def weighted_probability(obs):
    if not obs:
        return 0.5, 0.0
    sw = sum(w for _, w in obs)
    sw2 = sum(w * w for _, w in obs)
    eff = (sw * sw / sw2) if sw2 else 0.0
    p = (4.0 + sum(y * w for y, w in obs)) / (8.0 + sw)
    return clip(p), eff

def family_run(seq):
    if not seq:
        return []
    out = []
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

def family_signature(seq):
    return tuple((int(v), cnt) for v, cnt in family_run(seq))

def entropy_bits(seq):
    if not seq:
        return 0.0
    c = Counter(seq)
    total = len(seq)
    e = 0.0
    for freq in c.values():
        p = freq / total
        e -= p * math.log2(p)
    return e

def get_regime(seq):
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    rate = flips / max(1, len(tail) - 1)
    if run_length(tail) >= 5:
        return "RUN"
    if rate >= 0.72:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"

def trend_features(seq):
    if not seq:
        return [0.0] * 20

    def mean(k):
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5

    def ema(k):
        if not seq:
            return 0.5
        a = 2.0 / (k + 1)
        v = float(seq[0])
        for x in seq[1:]:
            v = a * x + (1.0 - a) * v
        return v

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
        (mean(12) - mean(48)),
        (mean(6) - mean(24))
    ]

def logistic(weights, features):
    return sigmoid(sum(w * x for w, x in zip(weights, features)))

# ============================================================
# HISTORY PARSER
# ============================================================

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
        records[key] = {
            "issue": key,
            "num": number,
            "size": int(number >= 5),
        }
    return sorted(records.values(), key=lambda x: int(x["issue"]))


# ============================================================
# QUANTUM PATTERN TRACKER
# ============================================================

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
            CREATE TABLE IF NOT EXISTS stats(key TEXT PRIMARY KEY, value TEXT NOT NULL);
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
            ORDER BY created DESC LIMIT 12
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
        inv = CFG.decay * m["inverse_loss"] + (1.0 - probability - actual) ** 2
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
            return probability, 0.7

        brier = m["loss"] / n
        inv = m["inverse_loss"] / n
        flipped = (inv + 0.02 < brier) and (brier > 0.25) and (m["failures"] >= 3)
        eff_loss = inv if flipped else brier
        p = (1.0 - probability) if flipped else probability

        reliability = clip((0.25 - eff_loss) / 0.12, 0.0, 1.0)
        shrink = (1.0 - min(n / 60.0, 1.0)) * 0.75 + min(n / 60.0, 1.0) * reliability
        p = 0.5 + (p - 0.5) * shrink

        weight = clip(math.exp(-7 * (eff_loss - 0.22)), 0.18, 2.0)
        if m["failures"] >= 4 and not flipped:
            weight *= 0.55
        return clip(p), weight

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires)
        )

    def close(self):
        self.c.close()


# ============================================================
# FLEXIBLE MODEL ENGINE — 900 WALK-FORWARD + EXACT NUMBER DIST
# ============================================================

MODEL_NAMES = ("global", "recent60", "recent120", "recent300", "markov1", "markov2", "markov3")

def _normalize(values):
    total = sum(values)
    if total <= 0:
        return [0.1] * 10
    return [float(v) / total for v in values]

def _prior(nums, smoothing=1.5):
    if not nums:
        return [0.1] * 10
    counts = Counter(nums)
    return _normalize([counts.get(k, 0) + smoothing for k in range(10)])

def _forecast_one(nums, model):
    if not nums:
        return [0.1] * 10

    if model == "global":
        return _prior(nums[-1000:])

    window_map = {"recent60": 60, "recent120": 120, "recent300": 300}
    if model in window_map:
        return _prior(nums[-window_map[model]:])

    order = {"markov1": 1, "markov2": 2, "markov3": 3}.get(model)
    if order is None:
        raise ValueError(f"Unknown model: {model}")

    prior = _prior(nums[-300:])
    if len(nums) <= order:
        return prior

    context = tuple(nums[-order:])
    counts = [0.0] * 10
    support = 0.0

    for target_idx in range(order, len(nums)):
        if tuple(nums[target_idx - order:target_idx]) != context:
            continue
        age = len(nums) - 1 - target_idx
        recency = 0.5 ** (age / 250.0)
        counts[nums[target_idx]] += recency
        support += recency

    if support <= 0:
        return prior

    prior_strength = 12.0
    return _normalize([
        counts[k] + prior_strength * prior[k]
        for k in range(10)
    ])

def _brier10(probabilities, actual):
    return sum(
        (probabilities[k] - (1.0 if k == actual else 0.0)) ** 2
        for k in range(10)
    ) / 2.0

def _score_models_from_walkforward(nums, steps=900, min_train=100, loss_streak=0):
    if len(nums) <= min_train:
        e = 1.0 / len(MODEL_NAMES)
        return (
            {name: e for name in MODEL_NAMES},
            {"steps": 0, "scores": None, "recovery": loss_streak >= 1},
        )

    start = max(min_train, len(nums) - steps)
    model_losses = {name: [] for name in MODEL_NAMES}

    for target_idx in range(start, len(nums)):
        past = nums[:target_idx]
        actual = nums[target_idx]

        for name in MODEL_NAMES:
            forecast = _forecast_one(past, name)
            model_losses[name].append(_brier10(forecast, actual))

    decay = 0.94 if loss_streak >= 1 else 0.985
    scores = {}
    for name, losses in model_losses.items():
        weights = [decay ** (len(losses) - 1 - i) for i in range(len(losses))]
        scores[name] = sum(w * v for w, v in zip(weights, losses)) / sum(weights)

    best_score = min(scores.values())
    raw = {
        name: math.exp(-8.0 * max(0.0, score - best_score))
        for name, score in scores.items()
    }
    raw_total = sum(raw.values())
    learned = {name: raw[name] / raw_total for name in MODEL_NAMES}
    uniform = 1.0 / len(MODEL_NAMES)
    learned_share = 0.60
    final = {
        name: learned_share * learned[name] + (1.0 - learned_share) * uniform
        for name in MODEL_NAMES
    }
    return final, {
        "steps": len(next(iter(model_losses.values()))),
        "scores": scores,
        "recovery": loss_streak >= 1,
    }

def exact_number_model(history, side_bias):
    if not history:
        return [1.0] * 10, "cold"

    nums = [x["num"] for x in history]
    sizes = [x["size"] for x in history]
    regime = get_regime([x["size"] for x in history])
    base = [1.0] * 10

    freq = Counter(nums)
    for k in range(10):
        base[k] *= 1.0 + 2.8 * freq.get(k, 0)

    recent = nums[-12:]
    recent_counter = Counter(recent)
    for k in range(10):
        base[k] *= 1.0 + 1.6 * recent_counter.get(k, 0)

    trans = {}
    for i in range(1, len(nums)):
        prev = nums[i - 1]
        cur = nums[i]
        trans[(prev, cur)] = trans.get((prev, cur), 0) + 1

    last_num = nums[-1] if nums else 0
    for k in range(10):
        base[k] *= 1.0 + 0.9 * trans.get((last_num, k), 0)
        base[k] *= 1.0 + 0.45 * trans.get((k, last_num), 0)

    # side bias is only soft; do not force BIG
    if side_bias >= 0.5:
        for k in range(0, 5):
            base[k] *= 0.75
        for k in range(5, 10):
            base[k] *= 1.25
    else:
        for k in range(5, 10):
            base[k] *= 0.75
        for k in range(0, 5):
            base[k] *= 1.25

    if regime == "RUN":
        if len(nums) >= 3:
            if nums[-1] >= 5:
                for k in range(5, 10):
                    base[k] *= 1.1
            else:
                for k in range(0, 5):
                    base[k] *= 1.1
    elif regime == "ALT":
        if len(nums) >= 4:
            a, b = nums[-2], nums[-1]
            for k in range(10):
                if k in (a, b):
                    base[k] *= 1.08
    elif regime == "IMBAL":
        avg = sum(recent) / len(recent)
        if avg >= 5:
            for k in range(5, 10):
                base[k] *= 1.08
        else:
            for k in range(0, 5):
                base[k] *= 1.08

    if len(sizes) >= 4:
        recent_size = sum(sizes[-6:]) / len(sizes[-6:])
        if recent_size > 0.7:
            for k in range(5, 10):
                base[k] *= 1.08
        elif recent_size < 0.3:
            for k in range(0, 5):
                base[k] *= 1.08

    total = sum(base)
    if total <= 0:
        return [1.0] * 10, regime
    return [x / total for x in base], regime


# ============================================================
# RECOVERY CONTROLLER
# ============================================================

class RecoveryController:
    def __init__(self):
        self.mode = 0

    def update(self, loss_streak):
        self.mode = 1 if loss_streak >= 1 else 0

    def adjust(self, seq, base_p, loss_streak):
        if loss_streak <= 0:
            return base_p, False

        recent = sum(seq[-5:]) / max(1, len(seq[-5:])) if seq else 0.5

        if loss_streak == 1:
            if recent > 0.70:
                return 0.24, True
            if recent < 0.30:
                return 0.76, True
            return 0.5 + (base_p - 0.5) * 0.58, True

        if recent > 0.72:
            return 0.19, True
        if recent < 0.28:
            return 0.81, True
        return 0.5 + (base_p - 0.5) * 0.44, True


# ============================================================
# ENGINE
# ============================================================

EXPERT_NAMES = [
    "patterns", "markov1", "markov2", "markov3",
    "run", "regime", "base", "trend", "momentum",
    "reversal", "volatility", "entropy"
]

class Engine:
    def __init__(self, db):
        self.db = db
        self.recovery = RecoveryController()

    def mine_patterns(self, seq):
        if len(seq) < 18:
            return []
        tracker = QuantumTracker(min_len=3, max_len=8, min_support=CFG.min_support)
        pat = tracker.match_tail_patterns(seq)
        out = []
        for item in pat[:6]:
            out.append((
                item["score"],
                item["p"],
                item["support"],
                item["eff_n"],
                "q:" + "-".join(f"{'B' if v else 'S'}{c}" for v, c in item["sig"])
            ))
        return out

    def mine_experts(self, seq):
        n = len(seq)
        experts = {}

        def weight(idx):
            return 0.5 ** ((n - 1 - idx) / 330.0)

        patterns = self.mine_patterns(seq)
        if patterns:
            candidates = []
            for _, p, _, eff_n, key in patterns:
                adjusted_p, w = self.db.adapt(key, p)
                candidates.append((adjusted_p, w * math.sqrt(max(1.0, eff_n))))
            if candidates:
                num = sum(p * w for p, w in candidates)
                den = sum(w for _, w in candidates)
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
                run_at[i] = run_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
            cur_run = min(run_at[-1], 8)
            last_val = seq[-1]
            run_obs = []
            for i in range(24, n):
                if seq[i-1] == last_val and min(run_at[i-1], 8) == cur_run:
                    run_obs.append((seq[i], weight(i)))
            if len(run_obs) >= CFG.min_support:
                experts["run"] = weighted_probability(run_obs)[0]

        regime = get_regime(seq)
        regime_obs = []
        for i in range(24, n):
            if get_regime(seq[max(0, i - 96):i]) == regime:
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
            avg = sum(tail) / len(tail)
            vol = sum((x - avg) ** 2 for x in tail) / len(tail)
            experts["volatility"] = clip(0.5 + (0.5 - abs(0.5 - avg)) * 0.85 + vol * 0.1)
        else:
            experts["volatility"] = 0.5

        experts["entropy"] = clip(0.5 + (entropy_bits(seq[-12:]) / 3.32 - 0.5) * 0.6)

        momentum = sum(seq[-4:]) / 4.0 if seq else 0.5
        experts["momentum"] = 0.2 + (momentum * 0.6)
        experts["reversal"] = 1.0 - experts["momentum"]

        for k, val in list(experts.items()):
            experts[k] = clip(val, 0.05, 0.95)

        return experts, patterns, features, regime

    def predict(self, history, issue):
        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]
        loss_streak = self.db.get_loss_streak()
        self.recovery.update(loss_streak)

        experts, patterns, features, regime = self.mine_experts(seq)

        adjusted = []
        for name, p in experts.items():
            corrected, weight = self.db.adapt(f"expert:{name}:{regime}", p)
            if name == "patterns":
                weight *= 3.3
            if name == "reversal" and loss_streak >= 1:
                weight *= 3.8
            if name in ("momentum", "run", "trend") and loss_streak >= 1:
                weight *= 0.25
            adjusted.append((corrected, weight))

        total_w = sum(w for _, w in adjusted)
        p_avg = sum(p * w for p, w in adjusted) / total_w if total_w > 0 else 0.5

        # direct exact distribution from walk-forward-scored models
        wf_weights, wf_info = _score_models_from_walkforward(
            nums,
            steps=900,
            min_train=100,
            loss_streak=loss_streak,
        )

        model_dist = [0.0] * 10
        for name in MODEL_NAMES:
            forecast = _forecast_one(nums, name)
            for k in range(10):
                model_dist[k] += wf_weights[name] * forecast[k]

        model_dist = _normalize(model_dist)

        # exact-number model (soft)
        exact_dist, _ = exact_number_model(history, p_avg)
        final_dist = [0.0] * 10
        for k in range(10):
            final_dist[k] = 0.55 * exact_dist[k] + 0.45 * model_dist[k]
        final_dist = _normalize(final_dist)

        # side from same final distribution
        p_big = sum(final_dist[5:10])
        p_small = sum(final_dist[:5])

        if p_big >= p_small:
            pred_side = "BIG"
            allowed = range(5, 10)
        else:
            pred_side = "SMALL"
            allowed = range(0, 5)

        # ensure final distribution is consistent with side
        side_adjusted = [0.0] * 10
        for k in range(10):
            if k in allowed:
                side_adjusted[k] = final_dist[k] * 1.8
            else:
                side_adjusted[k] = final_dist[k] * 0.2

        side_total = sum(side_adjusted)
        if side_total <= 0:
            side_adjusted = [0.1] * 10
            side_total = 10.0
        side_adjusted = [x / side_total for x in side_adjusted]

        # side probability used for output
        if pred_side == "BIG":
            p_out = p_big
        else:
            p_out = p_small

        # recovery: controlled, not blind flip
        if loss_streak >= 1:
            p_out, recovery_used = self.recovery.adjust(seq, p_out, loss_streak)
        else:
            recovery_used = False

        # anti-stubborn but not random reverse
        if len(seq) >= 8:
            short = sum(seq[-6:]) / 6.0
            if short > 0.80 and p_out > 0.66:
                p_out = 0.66
            if short < 0.20 and p_out < 0.34:
                p_out = 0.34

        # final side decision must be from the same final distribution
        big_prob = sum(side_adjusted[5:10])
        small_prob = sum(side_adjusted[:5])

        if big_prob >= small_prob:
            final_side = "BIG"
            final_allowed = range(5, 10)
        else:
            final_side = "SMALL"
            final_allowed = range(0, 5)

        num_pred = max(final_allowed, key=lambda k: side_adjusted[k])
        num_conf = side_adjusted[num_pred]

        # Replace the old p calculation with the same final distribution
        p_final = big_prob if final_side == "BIG" else small_prob

        # calibrate with global bucket
        bucket = min(19, int(p_final * 20))
        n_g, s_g = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((32 * p_final + s_g) / (32 + n_g))

        prev_regime = get_regime(seq[:-1]) if len(seq) > 1 else "N/A"
        trend_shift = f"{prev_regime} ➡️ {regime}" if prev_regime != regime else None

        payload = {
            "issue": issue,
            "p": p_cal,
            "p_raw": p_final,
            "side": final_side,
            "num": num_pred,
            "num_probability": num_conf,
            "regime": regime,
            "source": "🎯 Fixed Final Engine",
            "trend_shift": trend_shift,
            "pattern_str": _pattern_str(patterns),
            "edge": abs(p_cal - 0.5),
            "confidence": max(p_cal, 1.0 - p_cal),
            "experts": experts,
            "features": features,
            "meta_feats": m_feats,
            "is_recovery": recovery_used or (loss_streak > 0),
            "loss_streak": loss_streak,
            "patterns": [{"key": k, "p": p, "support": s} for _, p, s, _, k in patterns],
            "exact_distribution": side_adjusted,
            "model_weights": wf_weights,
            "walk_forward": wf_info,
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
            actual_side = "BIG" if actual["num"] >= 5 else "SMALL"
            predicted_side = payload.get("side", "BIG" if payload["p"] >= 0.5 else "SMALL")
            won_side = predicted_side == actual_side
            won_num = payload.get("num") == actual["num"]

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=? WHERE issue=?",
                    ("WIN" if won_side else "LOSS", actual["num"], row["issue"])
                )

                for name, val in payload["experts"].items():
                    self.db.learn(f"expert:{name}:{payload['regime']}", val, actual["size"])

                for pat in payload["patterns"]:
                    self.db.learn(pat["key"], pat["p"], actual["size"])

                if payload.get("features") and payload["experts"].get("trend"):
                    err = actual["size"] - payload["experts"]["trend"]
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
                        grad = 2.0 * (actual["size"] - mp) * mp * (1.0 - mp)
                        self.db.set("meta_weights", [
                            clip(w * 0.999 + CFG.meta_lr * grad * f, -4, 4)
                            for w, f in zip(mw, feats)
                        ])
                        self.db.set("meta_n", self.db.get("meta_n", 0) + 1)

                bucket = min(19, int(payload.get("p_raw", payload["p"]) * 20))
                n, s = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
                self.db.set(f"cal20:global:{bucket}", [0.995 * n + 1, 0.995 * s + actual["size"]])

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
            rec_tag = "⚠️ <b>ULTRA RECOVERY L1</b> ⚠️\n"
        else:
            rec_tag = "🚨 <b>ULTRA RECOVERY L2</b> 🚨\n"

    wf = f"\n📈 <b>WF900:</b> {p.get('walk_forward', {}).get('steps', 0)} steps" if p.get("walk_forward") else ""
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
        f"🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}{wf}{shift}{pat}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db):
        self.db, self.engine = db, Engine(db)
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
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", json.dumps(pred))
                            )
                            self.db.queue("prediction", issue, format_prediction(pred), time.time() + CFG.delivery_ttl)
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


# ============================================================
# MAIN
# ============================================================

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
