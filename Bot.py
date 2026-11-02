# bot.py — FTW VIP PREDICTOR (ADVANCED BEAST ENGINE)
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
    db_path: str = os.getenv("DB_PATH", "predictor_beast.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 5000
    min_history: int = int(os.getenv("MIN_HISTORY", "120"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    learning_rate: float = 0.05
    meta_lr: float = 0.045
    decay: float = 0.988
    max_pattern: int = 20
    delivery_ttl: float = 55.0
    name: str = "FTW VIP PREDICTOR (BEAST)"
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

def ema(seq, k):
    if not seq:
        return 0.5
    a = 2.0 / (k + 1)
    val = seq[0]
    for x in seq[1:]:
        val = a * x + (1.0 - a) * val
    return val

def trend_features(seq):
    if not seq:
        return [0.0] * 16

    def mean(k):
        arr = seq[-k:]
        return sum(arr) / len(arr) if arr else 0.5

    def std(k):
        arr = seq[-k:]
        if len(arr) < 2:
            return 0.0
        m = sum(arr) / len(arr)
        return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

    tail = seq[-12:]
    runs = run_length(seq)
    recent_avg = sum(seq[-6:]) / len(seq[-6:])
    recent_short = sum(seq[-3:]) / len(seq[-3:])
    alternation = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)
    ema12 = ema(seq, 12)
    ema6 = ema(seq, 6)

    return [
        1.0,
        2 * mean(3) - 1,
        2 * mean(5) - 1,
        2 * mean(10) - 1,
        2 * recent_avg - 1,
        2 * recent_short - 1,
        (2 * seq[-1] - 1),
        (2 * alternation - 1),
        (2 * ema12 - 1),
        (2 * ema6 - 1),
        (2 * std(5) - 1),
        (2 * std(10) - 1),
        (min(runs, 10) / 10.0) * (2 * seq[-1] - 1),
        (sum(seq[-10:]) / 10.0) - 0.5,
        (sum(seq[-5:]) / 5.0) - 0.5,
        (ema12 - ema6) * 2,
    ]

def logistic(weights, features):
    return sigmoid(sum(w * x for w, x in zip(weights, features)))

def suffix_count(seq, length):
    if len(seq) < length:
        return {}
    counts = {}
    for i in range(len(seq) - length + 1):
        s = tuple(seq[i:i+length])
        counts[s] = counts.get(s, 0) + 1
    return counts

def recent_suffix_scores(history, max_len=8):
    seq = [x["size"] for x in history]
    if not seq:
        return {}

    scores = {}
    for L in range(1, min(max_len, len(seq)) + 1):
        suffix = tuple(seq[-L:])
        cnt_big = 0
        cnt_small = 0
        total = 0
        for i in range(len(seq) - L):
            cand = tuple(seq[i:i+L])
            if cand == suffix:
                total += 1
                if seq[i+L] == 1:
                    cnt_big += 1
                else:
                    cnt_small += 1
        if total:
            scores[L] = {
                "suffix": suffix,
                "big": cnt_big / total,
                "small": cnt_small / total,
                "total": total,
                "bias": (cnt_big - cnt_small) / total
            }
    return scores

def regime_counts(history):
    seq = [x["size"] for x in history]
    out = {
        "RUN": {"big": 0, "small": 0, "total": 0},
        "ALT": {"big": 0, "small": 0, "total": 0},
        "IMBAL": {"big": 0, "small": 0, "total": 0},
        "MIXED": {"big": 0, "small": 0, "total": 0},
        "COLD": {"big": 0, "small": 0, "total": 0},
    }
    if not seq:
        return out

    for i in range(1, len(seq)):
        reg = get_regime(seq[max(0, i-24):i])
        out.setdefault(reg, {"big": 0, "small": 0, "total": 0})
        out[reg]["total"] += 1
        if seq[i] == 1:
            out[reg]["big"] += 1
        else:
            out[reg]["small"] += 1

    return out

def strong_reversal_allowed(history, current_regime):
    if len(history) < 12:
        return False
    seq = [x["size"] for x in history]
    tail = seq[-12:]
    recent_big_ratio = sum(tail) / len(tail)
    recent_run = run_length(tail)

    if current_regime in ("ALT", "RUN", "IMBAL"):
        if recent_big_ratio <= 0.35 or recent_big_ratio >= 0.65:
            if recent_run >= 3 or abs(recent_big_ratio - 0.5) >= 0.15:
                return True
    return False

# -----------------------------
# PARSE HISTORY
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
        self.set_value("last_issue", '"NONE"')
        self.set_value("last_side", '"NONE"')
        self.set_value("last_num", "0")

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
            return probability, 0.75, False
        brier = m["loss"] / n
        reliability = clip((0.26 - brier) / 0.12, 0.0, 1.0)
        maturity = min(n / 50.0, 1.0)
        shrink = (1 - maturity) * 0.65 + maturity * reliability
        adjusted = 0.5 + (probability - 0.5) * shrink
        weight = clip(math.exp(-6 * (brier - 0.22)), 0.18, 2.8)
        if m["failures"] >= 4:
            weight *= 0.7
        return clip(adjusted), weight, False

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind, issue, text, created, expires) VALUES (?, ?, ?, ?, ?)",
            (kind, issue, text, time.time(), expires)
        )

    def update_recovery_state(self, settled_result):
        consecutive_losses = int(self.get_value("consecutive_losses", 0))
        if settled_result == "LOSS":
            consecutive_losses += 1
        elif settled_result in ("WIN", "JACKPOT"):
            consecutive_losses = 0

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

    def build_suffix_model(self, seq):
        model = []
        if not seq:
            return model

        for L in range(1, min(8, len(seq)) + 1):
            suffix = tuple(seq[-L:])
            count = 0
            big_count = 0
            for i in range(len(seq) - L):
                cand = tuple(seq[i:i+L])
                if cand == suffix:
                    count += 1
                    if seq[i+L] == 1:
                        big_count += 1
            if count > 0:
                model.append((L, suffix, count, big_count / count))
        return sorted(model, key=lambda x: (x[2], abs(x[3]-0.5)), reverse=True)[:5]

    def build_markov_model(self, seq):
        model = []
        for L in range(1, 5):
            if len(seq) <= L:
                continue
            total = 0
            big_after = 0
            for i in range(len(seq) - L):
                state = tuple(seq[i:i+L])
                nxt = seq[i+L]
                total += 1
                if nxt == 1:
                    big_after += 1
            if total:
                model.append((L, big_after / total, total))
        return model

    def predict(self, history, issue):
        seq = [x["size"] for x in history]
        nums = [x["num"] for x in history]

        if not history:
            return {
                "issue": issue,
                "p": 0.5,
                "side": "BIG",
                "num": 5,
                "edge": 0.0,
                "confidence": 0.5,
                "regime": "MIXED",
                "source": "📊 Baseline",
                "patterns": [],
                "recovery_mode": False,
                "strong_reversal": False,
            }

        reg = get_regime(seq)
        reg_map = regime_counts(history)
        reg_counts = reg_map.get(reg, {"big": 1, "small": 1, "total": 2})
        reg_prob = (reg_counts["big"] / reg_counts["total"]) if reg_counts["total"] else 0.5

        suffix_model = self.build_suffix_model(seq)
        suffix_weight_total = 0.0
        suffix_weighted_prob = 0.0
        suffix_patterns = []
        for L, suffix, count, p in suffix_model:
            w = min(1.0, count / 20.0) * (1.0 + (L * 0.15))
            suffix_weight_total += w
            suffix_weighted_prob += p * w
            suffix_patterns.append({"key": f"suffix:{L}:{suffix}", "p": p, "support": count})

        suffix_prob = suffix_weighted_prob / suffix_weight_total if suffix_weight_total else 0.5

        markov_model = self.build_markov_model(seq)
        markov_prob = 0.5
        if markov_model:
            total_w = sum(w for _, _, w in markov_model)
            markov_prob = sum(p * w for _, p, w in markov_model) / total_w if total_w else 0.5

        tail = seq[-8:]
        tail_prob = sum(tail) / len(tail)

        recent_suffix = recent_suffix_scores(history, max_len=8)
        recent_best_prob = 0.5
        if recent_suffix:
            best = max(recent_suffix.items(), key=lambda kv: kv[1]["total"])
            _, data = best
            recent_best_prob = data["big"]

        # exact final blended probability
        p_big = 0.35 * suffix_prob + 0.25 * reg_prob + 0.20 * markov_prob + 0.12 * tail_prob + 0.08 * recent_best_prob
        p_big = clip(p_big, 0.02, 0.98)

        # recovery logic
        recovery = self.db.get_recovery_state()
        loss_streak = max(0, int(recovery["consecutive_losses"]))
        recovery_mode = bool(recovery["recovery_mode"]) or loss_streak > 0
        strong_reversal = strong_reversal_allowed(history, reg)

        if recovery_mode:
            if strong_reversal:
                p_big = clip(p_big - 0.10, 0.02, 0.98)
            else:
                p_big = clip(p_big + 0.05 * min(loss_streak + 1, 5), 0.02, 0.98)

        # choose side
        side = "BIG" if p_big >= 0.5 else "SMALL"

        # exact number selection
        num_dist = [1.0] * 10
        if nums:
            for L in range(1, min(8, len(nums)) + 1):
                suffix = tuple(nums[-L:])
                count = 0
                dist = {}
                for i in range(len(nums) - L):
                    cand = tuple(nums[i:i+L])
                    if cand == suffix:
                        count += 1
                        nxt = nums[i+L]
                        dist[nxt] = dist.get(nxt, 0) + 1
                if count > 0:
                    for n in range(10):
                        num_dist[n] *= 1.0 + 2.0 * dist.get(n, 0) / max(1, count)

        valid_nums = list(range(5, 10)) if side == "BIG" else list(range(0, 5))
        for n in range(10):
            if n not in valid_nums:
                num_dist[n] *= 0.35

        total_num = sum(num_dist)
        probs = [x / total_num for x in num_dist]
        num_pred = max(valid_nums, key=lambda n: probs[n])

        edge = abs(p_big - 0.5)
        confidence = max(p_big, 1.0 - p_big)
        if recovery_mode:
            confidence = min(0.99, confidence + 0.06)
            edge = max(edge, 0.12)

        return {
            "issue": issue,
            "p": p_big,
            "p_raw": p_big,
            "side": side,
            "num": num_pred,
            "edge": edge,
            "confidence": confidence,
            "regime": reg,
            "source": "📊 Advanced State Engine",
            "patterns": suffix_patterns,
            "recovery_mode": recovery_mode,
            "strong_reversal": strong_reversal,
        }

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

    pattern_text = ""
    if p.get("patterns"):
        first = p["patterns"][0]
        pattern_text = f"\n🧩 <b>PATTERN:</b> <code>{escape(first['key'])}</code>"

    rec_txt = ""
    if p.get("recovery_mode"):
        if p.get("strong_reversal"):
            rec_txt = "\n⚡ <b>STRONG REVERSAL:</b> ACTIVE"
        else:
            rec_txt = "\n⚡ <b>RECOVERY:</b> ACTIVE"

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
# SETTLEMENT
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

                settle_history(self.db, history)

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

                            self.db.set_value("last_issue", json.dumps(issue))
                            self.db.set_value("last_side", json.dumps(p["side"]))
                            self.db.set_value("last_num", str(p["num"]))

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

            LOG.info("%s Started", CFG.name)

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
