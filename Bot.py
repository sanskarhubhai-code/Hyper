# bot.py
# FULLY FIXED: No Bias (Random Tie-breaker), Win/Loss Bug Fixed.

import asyncio
import contextlib
import html
import json
import logging
import math
import os
import random  # 🔥 FIX: Added random module for unbiased 50/50 tie-breaking
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
    name: str = "Fixed Predictor"
    
    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"
    
    decay: float = 0.95 

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

def numeric(value):
    if isinstance(value, bool):
        return None
    s = str(value).strip()
    return int(s) if s.isascii() and s.isdigit() else None

def escape(x):
    return html.escape(str(x))

def run_length(seq):
    if not seq:
        return 0
    n = 0
    for x in reversed(seq):
        if x != seq[-1]:
            break
        n += 1
    return n

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

def weighted_probability(obs):
    sw = sum(w for _, w in obs)
    sw2 = sum(w * w for _, w in obs)
    effective_n = sw * sw / sw2 if sw2 else 0.0
    probability = (4 + sum(y * w for y, w in obs)) / (8 + sw)
    return probability, effective_n

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
        records[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda x: int(x["issue"]))

def trend_features(seq):
    if not seq:
        return [0.0] * 12

    def tail_mean(k):
        arr = seq[-k:]
        return mean(arr) if arr else 0.5

    tail = seq[-24:]
    alternation = sum(a != b for a, b in zip(tail, tail[1:])) / max(1, len(tail) - 1)

    fast = tail_mean(5)
    mid = tail_mean(10)
    slow = tail_mean(24)
    reg = get_regime(seq)
    reg_map = {"COLD": 0.0, "MIXED": 0.25, "ALT": 0.5, "IMBAL": 0.75, "RUN": 1.0}
    regime_value = reg_map.get(reg, 0.5)

    return [
        1.0,
        2 * fast - 1,
        2 * mid - 1,
        2 * slow - 1,
        2 * seq[-1] - 1,
        2 * alternation - 1,
        (fast - slow),
        (mid - slow),
        (run_length(seq) / 10.0),
        regime_value,
    ]


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

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.c.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def history(self):
        rows = self.c.execute("SELECT issue, num FROM history ORDER BY CAST(issue AS INTEGER)").fetchall()
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
        m = self.memory(key)
        n = CFG.decay * m["n"] + 1
        loss = CFG.decay * m["loss"] + (probability - actual) ** 2
        inv = CFG.decay * m["inverse_loss"] + (1 - probability - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        failures = m["failures"] + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n, loss=excluded.loss, inverse_loss=excluded.inverse_loss, failures=excluded.failures",
            (key, n, loss, inv, failures),
        )

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind, issue, text, created, expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires),
        )

    def get_pending_predictions(self):
        return self.c.execute(
            "SELECT issue, payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)"
        ).fetchall()


# ─── PREDICTOR ─────────────────────────────────────────────────
class SignalPredictor:
    def __init__(self, db):
        self.db = db

    def exact_pattern(self, seq):
        if len(seq) < 8:
            return 0.5
        candidates = []
        for length in range(4, min(12, len(seq) // 2) + 1):
            suffix = tuple(seq[-length:])
            obs = []
            for i in range(length, len(seq) - length):
                if tuple(seq[i - length:i]) == suffix:
                    obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 300.0)))
            if len(obs) >= 3:
                p, _ = weighted_probability(obs)
                candidates.append(p)
        if not candidates:
            return 0.5
        return sum(candidates) / len(candidates)

    def markov(self, seq):
        if len(seq) < 8:
            return 0.5
        vals = []
        for order in (1, 2, 3, 4):
            suffix = tuple(seq[-order:])
            obs = []
            for i in range(order, len(seq) - order):
                if tuple(seq[i - order:i]) == suffix:
                    obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 300.0)))
            if len(obs) >= 3:
                p, _ = weighted_probability(obs)
                vals.append(p)
        return sum(vals) / len(vals) if vals else 0.5

    def regime_signal(self, seq):
        reg = get_regime(seq)
        if len(seq) < 20:
            return 0.5
        obs = []
        for i in range(20, len(seq)):
            if get_regime(seq[max(0, i - 96):i]) == reg:
                obs.append((seq[i], 0.5 ** ((len(seq) - 1 - i) / 300.0)))
        if len(obs) >= 3:
            p, _ = weighted_probability(obs)
            return p
        return 0.5

    def trend_signal(self, seq):
        if not seq:
            return 0.5
        feats = trend_features(seq)
        z = sum(feats)
        return clip(1.0 / (1.0 + math.exp(-z / 8.0)))

    def big_share(self, seq):
        if not seq:
            return 0.5
        return mean(seq[-30:])

    def predict(self, history, issue):
        if not history:
            # 🔥 FIX: Random fallback instead of hardcoded SMALL
            side = random.choice(["BIG", "SMALL"])
            num = random.choice([2, 7])
            return {
                "issue": issue, "p": 0.5, "side": side, "num": num,
                "regime": "COLD", "source": "default", "experts": {}
            }

        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]
        reg = get_regime(seq)

        p_exact = self.exact_pattern(seq)
        p_markov = self.markov(seq)
        p_regime = self.regime_signal(seq)
        p_trend = self.trend_signal(seq)
        p_bigshare = self.big_share(seq)

        p = (
            0.28 * p_exact +
            0.24 * p_markov +
            0.20 * p_regime +
            0.18 * p_trend +
            0.10 * p_bigshare
        )
        p = clip(p)

        if reg == "RUN":
            p = clip(0.5 + (p - 0.5) * 1.2)
        elif reg == "ALT":
            p = clip(0.5 + (p - 0.5) * 0.7)

        # 🔥 FIX: If p is exactly 50% (or extremely close), pick RANDOMLY to avoid bias
        if abs(p - 0.5) < 0.01:
            side = random.choice(["BIG", "SMALL"])
            p = 0.5
        else:
            side = "BIG" if p > 0.5 else "SMALL"

        valid_range = list(range(5, 10)) if side == "BIG" else list(range(0, 5))

        freq = Counter(nums)
        dist = [1.0] * 10
        total = len(nums)
        if total > 0:
            for i in range(10):
                dist[i] += 6.0 * (freq.get(i, 0) / total)

        recent = nums[-10:]
        if recent:
            for i in range(10):
                dist[i] += 2.0 * (recent.count(i) / len(recent))

        score = {n: dist[n] for n in valid_range}
        best_num = max(valid_range, key=lambda n: score[n])

        return {
            "issue": issue,
            "p": p,
            "side": side,
            "num": best_num,
            "regime": reg,
            "source": "signal-only",
            "experts": {
                "exact": p_exact,
                "markov": p_markov,
                "regime": p_regime,
                "trend": p_trend,
                "big_share": p_bigshare,
            },
        }

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
            won_side = (pred_side == actual_side)

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=? WHERE issue=?",
                    ("WIN" if won_side else "LOSS", actual["num"], issue),
                )

                for name, val in pred.get("experts", {}).items():
                    self.db.learn(f"expert:{name}:{pred.get('regime', 'N/A')}", val, actual["size"])

                msg = (
                    f"<b>{'✅ WIN' if won_side else '❌ LOSS'}</b>\n"
                    f"<b>Issue:</b> {issue}\n"
                    f"<b>Actual:</b> {actual['num']} ({actual_side})\n"
                    f"<b>Prediction:</b> {pred_side} -> {pred.get('num', '?')}\n"
                    f"<b>Confidence:</b> {max(pred.get('p', 0.5), 1-pred.get('p', 0.5)) * 100:.1f}%"
                )
                self.db.queue("result", issue, msg, time.time() + 60.0)


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
            LOG.info("Fixed predictor started")
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
