import asyncio
import contextlib
import html
import json
import logging
import math
import os
import random
import signal
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import aiohttp
from aiohttp import web


# ─── CONFIG ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    token: str = "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c"
    chat: str = "-1004485961768"
    urls: tuple = (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "fixed_predictor.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 4000
    min_history: int = int(os.getenv("MIN_HISTORY", "120"))
    name: str = "Ultra Predictor"

    decay: float = 0.975
    tie_band: float = 0.012

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config: token, chat, or urls")


CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor")


# ─── HELPERS ───────────────────────────────────────────────────
def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.5


def variance(xs):
    if not xs:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def stddev(xs):
    return math.sqrt(variance(xs))


def numeric(value):
    if isinstance(value, bool):
        return None
    s = str(value).strip()
    return int(s) if s.isascii() and s.isdigit() else None


def escape(x):
    return html.escape(str(x))


def weighted_probability(obs):
    if not obs:
        return 0.5, 0.0
    sw = sum(w for _, w in obs)
    sw2 = sum(w * w for _, w in obs)
    effective_n = sw * sw / sw2 if sw2 else 0.0
    probability = (4 + sum(y * w for y, w in obs)) / (8 + sw)
    return clip(probability), effective_n


def run_length(seq):
    if not seq:
        return 0
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x == last:
            n += 1
        else:
            break
    return n


def entropy(seq):
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


def get_regime(seq):
    if not seq:
        return "COLD"
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"
    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    rate = flips / max(1, len(tail) - 1)
    if run_length(tail) >= 4:
        return "RUN"
    if rate >= 0.72:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"


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
            None,
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
        if key in records and records[key]["num"] != number:
            raise ValueError(f"Conflicting duplicate: {key}")
        records[key] = {
            "issue": key,
            "num": number,
            "size": int(number >= 5),
        }
    return sorted(records.values(), key=lambda x: int(x["issue"]))


# ─── DATABASE ─────────────────────────────────────────────────
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
            CREATE TABLE IF NOT EXISTS outbox(id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, issue TEXT, text TEXT NOT NULL, created REAL NOT NULL, expires REAL, sent INTEGER NOT NULL DEFAULT 0);
        """)

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.c.execute("COMMIT")
        except BaseException:
            self.c.execute("ROLLBACK")
            raise

    def history(self):
        rows = self.c.execute(
            "SELECT issue, num FROM history ORDER BY CAST(issue AS INTEGER)"
        ).fetchall()
        return [{"issue": r["issue"], "num": r["num"], "size": int(r["num"] >= 5)} for r in rows]

    def ingest(self, records):
        with self.transaction():
            for r in records:
                old = self.c.execute("SELECT num FROM history WHERE issue=?", (r["issue"],)).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute(
                    "INSERT OR IGNORE INTO history VALUES (?, ?)",
                    (r["issue"], r["num"]),
                )
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN "
                "(SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,),
            )

    def memory(self, key):
        row = self.c.execute("SELECT * FROM models WHERE key=?", (key,)).fetchone()
        return dict(row) if row else {"n": 0.0, "loss": 0.0, "inverse_loss": 0.0, "failures": 0}

    def learn(self, key, probability, actual):
        # actual = 1 for BIG, 0 for SMALL
        m = self.memory(key)
        n = CFG.decay * m["n"] + 1.0
        err = probability - actual
        err_inv = (1.0 - probability) - actual
        loss = CFG.decay * m["loss"] + (err ** 2)
        inv = CFG.decay * m["inverse_loss"] + (err_inv ** 2)
        predicted_side = 1 if probability >= 0.5 else 0
        failures = m["failures"] + 1 if predicted_side != int(actual) else 0

        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n, loss=excluded.loss, inverse_loss=excluded.inverse_loss, failures=excluded.failures",
            (key, n, loss, inv, failures),
        )

    def get_pending_predictions(self):
        return self.c.execute(
            "SELECT issue, payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)"
        ).fetchall()

    def expert_weights(self):
        rows = self.c.execute("SELECT key, n, loss, inverse_loss, failures FROM models").fetchall()
        weights = {}
        for row in rows:
            key = row["key"]
            if not key.startswith("expert:"):
                continue
            base = float(row["n"])
            if base <= 0:
                continue
            quality = 1.0 / (1.0 + float(row["loss"]))
            quality2 = 1.0 / (1.0 + float(row["inverse_loss"]))
            recency = min(1.0, base / 30.0)
            fail_penalty = 0.85 ** max(0, int(row["failures"]))
            weights[key] = max(0.0, quality * quality2 * recency * fail_penalty)
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in weights.items()}

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind, issue, text, created, expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires),
        )

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.c.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )


# ─── PREDICTOR ─────────────────────────────────────────────────
class SignalPredictor:
    def __init__(self, db):
        self.db = db

    # ---------- Core signals ----------
    def exact_pattern(self, seq):
        if len(seq) < 8:
            return 0.5
        hits = []
        for length in range(3, min(12, len(seq) // 2) + 1):
            suffix = tuple(seq[-length:])
            obs = []
            for i in range(length, len(seq) - length):
                if tuple(seq[i - length:i]) == suffix:
                    obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 320.0)))
            if len(obs) >= 3:
                p, _ = weighted_probability(obs)
                hits.append(p)
        return mean(hits) if hits else 0.5

    def markov_pattern(self, seq):
        if len(seq) < 8:
            return 0.5
        vals = []
        for order in (1, 2, 3, 4, 5):
            if len(seq) <= order:
                continue
            suffix = tuple(seq[-order:])
            obs = []
            for i in range(order, len(seq) - order):
                if tuple(seq[i - order:i]) == suffix:
                    obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 360.0)))
            if len(obs) >= 2:
                p, _ = weighted_probability(obs)
                vals.append(p)
        return mean(vals) if vals else 0.5

    def regime_signal(self, seq):
        reg = get_regime(seq)
        if len(seq) < 20:
            return 0.5
        obs = []
        for i in range(20, len(seq)):
            if get_regime(seq[max(0, i - 96):i]) == reg:
                obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 400.0)))
        if len(obs) >= 3:
            p, _ = weighted_probability(obs)
            return p
        return 0.5

    def trend_signal(self, seq):
        if not seq:
            return 0.5
        recent = seq[-35:]
        s1 = mean(recent[-5:])
        s2 = mean(recent[-10:])
        s3 = mean(recent[-20:])
        slope = (s1 - s3) + 0.5 * (s2 - s3)
        pressure = 2.0 * (s1 - 0.5) + 1.5 * (s2 - 0.5) + 1.0 * slope
        return clip(1.0 / (1.0 + math.exp(-pressure * 2.5)))

    def momentum_signal(self, seq):
        if len(seq) < 2:
            return 0.5
        short = mean(seq[-5:])
        mid = mean(seq[-12:])
        cur = seq[-1]
        val = 0.35 * (short - 0.5) + 0.35 * (mid - 0.5) + 0.30 * (cur - 0.5)
        return clip(0.5 + val)

    def alternation_signal(self, seq):
        if len(seq) < 4:
            return 0.5
        tail = seq[-20:]
        flips = sum(a != b for a, b in zip(tail, tail[1:]))
        alt_rate = flips / max(1, len(tail) - 1)
        if alt_rate > 0.75:
            return 0.35
        if alt_rate < 0.22:
            return 0.65
        return 0.5

    def distribution_signal(self, seq):
        if not seq:
            return 0.5
        tail = seq[-80:]
        big_frac = sum(1 for x in tail if x == 1) / len(tail)
        return clip(big_frac)

    def entropy_signal(self, seq):
        if not seq:
            return 0.5
        tail = seq[-30:]
        ent = entropy(tail)
        return clip(0.5 + (0.5 - ent / 4.0))

    def recent_bias_signal(self, seq):
        if not seq:
            return 0.5
        tail = seq[-15:]
        return clip(mean(tail))

    def big_share_signal(self, seq):
        if not seq:
            return 0.5
        return clip(mean(seq[-40:]))

    def volatility_signal(self, seq):
        if len(seq) < 5:
            return 0.5
        tail = seq[-20:]
        sigma = stddev(tail)
        # More volatility usually means less certain, so closer to 0.5
        if sigma < 0.12:
            return 0.65
        if sigma > 0.38:
            return 0.45
        return 0.5

    def autocorrelation_signal(self, seq):
        if len(seq) < 6:
            return 0.5
        tail = seq[-12:]
        if len(tail) < 2:
            return 0.5
        mean_val = mean(tail)
        cov = sum((a - mean_val) * (b - mean_val) for a, b in zip(tail, tail[1:]))
        var = sum((x - mean_val) ** 2 for x in tail)
        if var == 0:
            return 0.5
        corr = cov / max(1e-9, var)
        return clip(0.5 + corr * 0.7)

    def cycle_signal(self, seq):
        if len(seq) < 10:
            return 0.5
        tail = seq[-20:]
        # local pulse detection
        period_hints = []
        for k in range(2, min(8, len(tail) // 2) + 1):
            aa = tail[::k]
            if len(aa) < 3:
                continue
            period_hints.append(mean(aa))
        if not period_hints:
            return 0.5
        pulse = mean(period_hints)
        return clip(0.5 + (pulse - 0.5) * 1.2)

    # ---------- adaptive fusion ----------
    def adapt_probability(self, signal_map):
        base = {
            "exact": 1.30,
            "markov": 1.20,
            "regime": 1.12,
            "trend": 1.18,
            "momentum": 1.06,
            "alternation": 0.80,
            "distribution": 0.95,
            "entropy": 0.70,
            "recent_bias": 1.00,
            "big_share": 1.00,
            "volatility": 0.75,
            "autocorrelation": 0.85,
            "cycle": 0.88,
        }
        weights = self.db.expert_weights()
        final = {}
        for key in signal_map:
            derived = weights.get(f"expert:{key}", 1.0)
            final[key] = base.get(key, 1.0) * (0.55 + derived)

        total = sum(final.values())
        if total <= 0:
            return 0.5

        res = 0.5
        for key, val in signal_map.items():
            res += (val - 0.5) * final[key] / total
        return clip(res)

    # ---------- settle prediction ----------
    def settle(self, history):
        lookup = {str(r["issue"]): r for r in history}
        pending_rows = self.db.get_pending_predictions()

        for row in pending_rows:
            if row is None:
                continue
            issue = str(row["issue"])
            if issue not in lookup:
                continue

            payload = row["payload"]
            if not payload:
                continue

            try:
                pred = json.loads(payload)
            except Exception:
                continue

            actual = lookup[issue]
            actual_side = "BIG" if actual["size"] == 1 else "SMALL"

            pred_side = pred.get("side", "BIG")
            won_side = pred_side == actual_side

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=? WHERE issue=?",
                    ("WIN" if won_side else "LOSS", actual["num"], issue),
                )

                for name, val in pred.get("experts", {}).items():
                    expert_key = f"expert:{name}:{pred.get('regime', 'N/A')}"
                    self.db.learn(expert_key, clip(float(val)), float(actual["size"]))

                msg = (
                    f"<b>{'✅ WIN' if won_side else '❌ LOSS'}</b>\n"
                    f"<b>Issue:</b> {issue}\n"
                    f"<b>Actual:</b> {actual['num']} ({actual_side})\n"
                    f"<b>Prediction:</b> {pred_side} -> {pred.get('num', '?')}\n"
                    f"<b>Confidence:</b> {max(pred.get('p', 0.5), 1-pred.get('p', 0.5)) * 100:.1f}%"
                )
                self.db.queue("result", issue, msg, time.time() + 60.0)

    # ---------- exact number chooser ----------
    def choose_exact_number(self, side, nums, regime):
        valid = list(range(5, 10)) if side == "BIG" else list(range(0, 5))
        freq = Counter(nums)
        recent = nums[-12:]
        scores = {n: 0.0 for n in range(10)}

        total = max(1, len(nums))
        for n in range(10):
            scores[n] += 1.0 + 5.0 * (freq.get(n, 0) / total)

        if recent:
            for n in range(10):
                scores[n] += 2.5 * (recent.count(n) / len(recent))

        for n in valid:
            scores[n] += 1.8

        if len(nums) >= 3:
            last3 = nums[-3:]
            for n in valid:
                scores[n] += 2.0 * last3.count(n)

        if regime == "RUN":
            for n in valid:
                scores[n] += 1.2
        elif regime == "ALT":
            for n in valid:
                scores[n] += 0.5

        pool = []
        for n in valid:
            weight = int(max(1, round(scores[n] * 100)))
            pool.extend([n] * weight)

        if not pool:
            return random.choice(valid)
        return random.choice(pool)

    # ---------- final prediction ----------
    def predict(self, history, issue):
        if not history:
            side = random.choice(["BIG", "SMALL"])
            num = random.choice([2, 7])
            return {
                "issue": issue,
                "p": 0.5,
                "side": side,
                "num": num,
                "regime": "COLD",
                "source": "fallback-random",
                "experts": {
                    "exact": 0.5,
                    "markov": 0.5,
                    "regime": 0.5,
                    "trend": 0.5,
                    "momentum": 0.5,
                    "alternation": 0.5,
                    "distribution": 0.5,
                    "entropy": 0.5,
                    "recent_bias": 0.5,
                    "big_share": 0.5,
                    "volatility": 0.5,
                    "autocorrelation": 0.5,
                    "cycle": 0.5,
                },
            }

        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]
        regime = get_regime(seq)

        signal_map = {
            "exact": self.exact_pattern(seq),
            "markov": self.markov_pattern(seq),
            "regime": self.regime_signal(seq),
            "trend": self.trend_signal(seq),
            "momentum": self.momentum_signal(seq),
            "alternation": self.alternation_signal(seq),
            "distribution": self.distribution_signal(seq),
            "entropy": self.entropy_signal(seq),
            "recent_bias": self.recent_bias_signal(seq),
            "big_share": self.big_share_signal(seq),
            "volatility": self.volatility_signal(seq),
            "autocorrelation": self.autocorrelation_signal(seq),
            "cycle": self.cycle_signal(seq),
        }

        p = self.adapt_probability(signal_map)

        if regime == "RUN":
            p = clip(0.5 + (p - 0.5) * 1.28)
        elif regime == "ALT":
            p = clip(0.5 + (p - 0.5) * 0.72)
        elif regime == "IMBAL":
            p = clip(0.5 + (p - 0.5) * 1.16)
        elif regime == "COLD":
            p = clip(0.5 + (p - 0.5) * 0.9)

        # unbiased 50/50 tie breaker
        if abs(p - 0.5) <= CFG.tie_band:
            side = random.choice(["BIG", "SMALL"])
            p = 0.5
        else:
            side = "BIG" if p > 0.5 else "SMALL"

        num = self.choose_exact_number(side, nums, regime)

        return {
            "issue": issue,
            "p": p,
            "side": side,
            "num": num,
            "regime": regime,
            "source": "ultra-deep-ensemble",
            "experts": {
                "exact": signal_map["exact"],
                "markov": signal_map["markov"],
                "regime": signal_map["regime"],
                "trend": signal_map["trend"],
                "momentum": signal_map["momentum"],
                "alternation": signal_map["alternation"],
                "distribution": signal_map["distribution"],
                "entropy": signal_map["entropy"],
                "recent_bias": signal_map["recent_bias"],
                "big_share": signal_map["big_share"],
                "volatility": signal_map["volatility"],
                "autocorrelation": signal_map["autocorrelation"],
                "cycle": signal_map["cycle"],
            },
        }


# ─── FORMATTER ──────────────────────────────────────────────
def format_prediction(p):
    conf = max(p["p"], 1 - p["p"]) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = p["side"]
    color = "🔴" if side == "BIG" else "🔵"

    if conf >= 60:
        quality = "🔥 HOT"
        stake = "2x"
    elif conf >= 55:
        quality = "⭐ STRONG"
        stake = "1.5x"
    elif conf >= 52:
        quality = "✅ BALANCED"
        stake = "1x"
    else:
        quality = "📉 CAUTIOUS"
        stake = "0.5x"

    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {color} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake})\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime', 'N/A'))}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ─── RUNTIME ─────────────────────────────────────────────────
class Runtime:
    def __init__(self, db):
        self.db = db
        self.engine = SignalPredictor(db)
        self.stop = asyncio.Event()
        self.last_fetch = 0.0

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
                failures = 0

                self.engine.settle(history)

                latest = int(history[-1]["issue"])
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)

                    exists = self.db.c.execute(
                        "SELECT 1 FROM predictions WHERE issue=?", (issue,)
                    ).fetchone()

                    already_queued = self.db.c.execute(
                        "SELECT 1 FROM outbox WHERE issue=? AND kind='prediction'", (issue,)
                    ).fetchone()

                    if not exists and not already_queued:
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", json.dumps(pred)),
                            )
                            self.db.queue(
                                "prediction",
                                issue,
                                format_prediction(pred),
                                time.time() + 60.0,
                            )

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                LOG.error("Worker failure: %s", type(exc).__name__)

            await self.pause(min(60, CFG.fetch_every * 2 ** min(failures, 4)))

    async def sender(self, session):
        endpoint_msg = f"https://api.telegram.org/bot{CFG.token}/sendMessage"
        endpoint_sticker = f"https://api.telegram.org/bot{CFG.token}/sendSticker"
        backoff = 1.0
        while not self.stop.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
            if row is None:
                await self.pause(0.5)
                continue

            delay = backoff
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
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    else:
                        delay = 60

            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning("Telegram send failure: %s", e)
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
            "win_rate": f"{acc:.1f}%",
        })


# ─── MAIN ──────────────────────────────────────────────────
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
        connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300, ssl=False)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session), name="worker"),
                asyncio.create_task(runtime.sender(session), name="sender"),
            ]
            stop_task = asyncio.create_task(runtime.stop.wait())
            LOG.info("Ultra predictor started")
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
