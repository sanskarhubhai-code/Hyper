# FTW PREDICTOR — SAFE FINAL VERSION (Fixed & Ready)
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
    # 🔥 Tera original Token aur Chat ID set kar diya hai
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls: tuple = (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "predictor_safe_final_v2.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "2.0"))
    timeout: float = 18.0
    max_history: int = 3500 # 🔥 12000 se 3500 kar diya taaki Render crash na ho
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (SAFE FINAL)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ftw_safe_final")


# ============================================================
# HELPERS
# ============================================================

def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))

def sigmoid(z):
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))

def escape(x):
    return html.escape(str(x))

def numeric(value):
    if isinstance(value, bool):
        return None
    s = str(value).strip()
    if s.isascii() and s.isdigit():
        return int(s)
    return None

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

def normalize(values):
    total = sum(values)
    if total <= 0:
        return [0.1] * 10
    return [v / total for v in values]

def recent_big_ratio(seq):
    if not seq:
        return 0.5
    window = seq[-50:]
    return sum(window) / len(window) if window else 0.5

def move_avg(seq, k):
    if not seq:
        return 0.5
    window = seq[-k:]
    return sum(window) / len(window) if window else 0.5

def recent_frequency(nums, window=200):
    if not nums:
        return [0.1] * 10
    recent = nums[-window:]
    counts = Counter(recent)
    return normalize([counts.get(i, 0) + 1.0 for i in range(10)])

def markov_predict(nums, order=2):
    if not nums:
        return [0.1] * 10
    if len(nums) <= order:
        return recent_frequency(nums)
    context = tuple(nums[-order:])
    counts = [0.0] * 10
    total = 0.0
    for i in range(order, len(nums)):
        if tuple(nums[i-order:i]) == context:
            counts[nums[i]] += 1.0
            total += 1.0
    if total <= 0:
        return recent_frequency(nums)
    return normalize([counts[k] + 1.0 for k in range(10)])

def weighted_distribution(history):
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

    return normalize(final)

def exact_number_model(history):
    nums = [int(r["num"]) for r in history]
    if not nums:
        return [0.1] * 10

    freq = recent_frequency(nums, 200)
    markov = markov_predict(nums, 2)
    short = recent_frequency(nums, 40)

    final = [0.0] * 10
    for i in range(10):
        final[i] = 0.45 * freq[i] + 0.30 * markov[i] + 0.25 * short[i]

    return normalize(final)

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

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
        finally:
            self.c.execute("COMMIT" if self.c.in_transaction else "ROLLBACK")

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return default

    def set(self, key, value):
        self.c.execute(
            "INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
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

    def queue(self, kind, issue, text, expires=None):
        self.c.execute(
            "INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires)
        )

    def close(self):
        self.c.close()


# ============================================================
# RECOVERY
# ============================================================

class RecoveryController:
    def __init__(self):
        self.mode = 0

    def update(self, loss_streak):
        self.mode = 1 if loss_streak >= 1 else 0

    def adjust(self, side_prob, loss_streak):
        if loss_streak <= 0:
            return side_prob, False

        # gentle correction only, no blind reversal
        if loss_streak == 1:
            return 0.5 + (side_prob - 0.5) * 0.55, True

        return 0.5 + (side_prob - 0.5) * 0.40, True


# ============================================================
# MODEL LOGIC
# ============================================================

class Engine:
    def __init__(self, db):
        self.db = db
        self.recovery = RecoveryController()

    def predict(self, history, issue):
        nums = [int(r["num"]) for r in history]
        recent_nums = nums[-200:]

        # side probability using recent big ratio and weighted distribution
        side_prob = 0.5
        if recent_nums:
            ratio = sum(1 for x in recent_nums if x >= 5) / len(recent_nums)
            side_prob = clip(ratio, 0.02, 0.98)

        # number distribution from weighted exact model
        exact = exact_number_model(history)
        weighted = weighted_distribution(history)
        final_dist = [0.0] * 10
        for i in range(10):
            final_dist[i] = 0.55 * exact[i] + 0.45 * weighted[i]

        final_dist = normalize(final_dist)

        p_big = sum(final_dist[5:10])
        p_small = sum(final_dist[:5])

        # blend with side probability
        side_prob = 0.60 * side_prob + 0.40 * (p_big if p_big >= p_small else p_small)

        # recovery after 1 loss
        loss_streak = self.db.get_loss_streak()
        self.recovery.update(loss_streak)
        if loss_streak >= 1:
            side_prob, recovery_used = self.recovery.adjust(side_prob, loss_streak)
        else:
            recovery_used = False

        # maintain a neutral bias when trend is weak
        if not nums:
            side_prob = 0.5
        elif len(nums) > 20:
            recent_big = sum(1 for x in nums[-20:] if x >= 5) / 20
            if abs(recent_big - 0.5) < 0.08:
                side_prob = 0.5 + (side_prob - 0.5) * 0.5

        # side and exact number must come from the same distribution
        if p_big >= p_small:
            final_side = "BIG"
            allowed = range(5, 10)
        else:
            final_side = "SMALL"
            allowed = range(0, 5)

        # if distribution is weak, keep low confidence and neutral side
        if abs(p_big - p_small) < 0.04:
            final_side = "BIG" if side_prob >= 0.5 else "SMALL"
            allowed = range(5, 10) if final_side == "BIG" else range(0, 5)

        side_adj = [0.0] * 10
        for k in range(10):
            if k in allowed:
                side_adj[k] = final_dist[k] * 1.5
            else:
                side_adj[k] = final_dist[k] * 0.5

        total_side = sum(side_adj)
        if total_side <= 0:
            side_adj = [0.1] * 10
            total_side = 10.0
        side_adj = [x / total_side for x in side_adj]

        num = max(allowed, key=lambda k: side_adj[k])

        payload = {
            "issue": issue,
            "side": final_side,
            "num": num,
            "p": side_prob,
            "num_probability": side_adj[num],
            "exact_distribution": side_adj,
            "is_recovery": recovery_used or (loss_streak > 0),
            "loss_streak": loss_streak,
            "source": "SAFE FINAL",
            "confidence": max(side_prob, 1.0 - side_prob),
        }
        return payload

    def settle(self, history):
        lookup = {r["issue"]: r for r in history}

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

                sticker = CFG.jack_sticker if (won_side and won_num) else CFG.win_sticker if won_side else CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)
                self.db.c.execute("UPDATE outbox SET sent=2 WHERE kind='prediction' AND issue=? AND sent=0", (row["issue"],))

def format_prediction(p):
    conf = max(p["p"], 1.0 - p["p"]) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if p["side"] == "BIG" else "SMALL"

    rec_tag = ""
    if p.get("is_recovery") and p.get("loss_streak", 0) >= 1:
        rec_tag = "⚠️ <b>RECOVERY</b> ⚠️\n"

    return (
        f"{rec_tag}⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {'🔴' if side == 'BIG' else '🔵'} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db):
        self.db = db
        self.engine = Engine(db)
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
                self.engine.settle(history)

                latest = int(history[-1]["issue"])
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    if not self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone():
                        pred = self.engine.predict(history, issue)
                        pred["issue"] = issue
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", json.dumps(pred))
                            )
                            self.db.queue("prediction", issue, format_prediction(pred), time.time() + CFG.delivery_ttl)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.error("Worker failure: %s", exc)
                failures += 1

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
                    if code == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404):
                        delay = 60
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning("Telegram network fail: %s", e)
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
