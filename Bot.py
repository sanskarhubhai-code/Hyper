# FTW PREDICTOR — FULL FIXED VERSION WITH PATTERN + TREND + SEQUENCE
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
    db_path: str = os.getenv("DB_PATH", "predictor_fixed_pattern_final.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "2.0"))
    timeout: float = 18.0
    max_history: int = 12000
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (FIXED PATTERN FINAL)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("ftw_fixed_pattern_final")


# ============================================================
# HELPERS
# ============================================================

def clip(x, low=0.02, high=0.98):
    return max(low, min(high, float(x)))

def normalize(values):
    total = sum(values)
    if total <= 0:
        return [0.1] * 10
    return [v / total for v in values]

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

def run_pattern(bits):
    if not bits:
        return "—"
    parts = []
    current = bits[0]
    count = 1
    for value in bits[1:]:
        if value == current:
            count += 1
        else:
            parts.append(f"{'B' if current else 'S'}{count}")
            current = value
            count = 1
    parts.append(f"{'B' if current else 'S'}{count}")
    return "-".join(parts)

def size_patterns(history, max_len=12):
    if len(history) < 4:
        return []

    seq = [int(r["size"]) for r in history]
    found = []

    for length in range(min(max_len, len(seq) - 1), 2, -1):
        context = seq[-length:]
        matches = 0
        big_count = 0
        for i in range(len(seq) - length):
            if seq[i:i + length] == context:
                matches += 1
                big_count += int(seq[i + length] >= 5) if i + length < len(seq) else 0

        if matches < 2:
            continue

        p_big = (2.0 + big_count) / (4.0 + matches)
        found.append({
            "key": run_pattern(context),
            "length": length,
            "support": matches,
            "p_big": p_big,
            "predicted_side": "BIG" if p_big >= 0.5 else "SMALL",
            "lift": p_big - 0.5,
        })

    return found

def trend_side(history):
    if len(history) < 10:
        return 0.5, "neutral"

    recent = [int(r["size"]) for r in history][-20:]
    ratio = sum(recent) / len(recent)

    if ratio > 0.65:
        return 0.7, "big_uptrend"
    if ratio < 0.35:
        return 0.3, "small_uptrend"
    return 0.5, "neutral"

def exact_number_distribution(history):
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
# RECOVERY CONTROLLER
# ============================================================

class RecoveryController:
    def __init__(self):
        self.mode = 0

    def update(self, loss_streak):
        self.mode = 1 if loss_streak >= 1 else 0

    def adjust(self, side_prob, loss_streak):
        if loss_streak <= 0:
            return side_prob, False

        if loss_streak == 1:
            return 0.5 + (side_prob - 0.5) * 0.55, True

        return 0.5 + (side_prob - 0.5) * 0.42, True


# ============================================================
# ENGINE
# ============================================================

class Engine:
    def __init__(self, db):
        self.db = db
        self.recovery = RecoveryController()

    def predict(self, history, issue):
        nums = [int(r["num"]) for r in history]
        if not nums:
            final_dist = [0.1] * 10
            return {
                "issue": issue,
                "side": "BIG",
                "num": 7,
                "p": 0.5,
                "num_probability": 0.1,
                "exact_distribution": final_dist,
                "is_recovery": False,
                "loss_streak": 0,
                "source": "SAFE",
                "pattern_trace": [],
                "pattern_display": "—",
                "trend_reason": "neutral",
            }

        exact = exact_number_distribution(history)
        pattern_candidates = size_patterns(history, max_len=12)

        # build pattern display from strongest supported pattern
        pattern_display = "—"
        pattern_trace = []
        if pattern_candidates:
            for p in pattern_candidates[:6]:
                pattern_trace.append({
                    "key": p["key"],
                    "length": p["length"],
                    "support": p["support"],
                    "p_big": p["p_big"],
                    "predicted_side": p["predicted_side"],
                })
            pattern_display = pattern_candidates[0]["key"]

        # side probability from exact number + pattern + trend
        p_big = sum(exact[5:10])
        p_small = sum(exact[:5])

        pattern_bias = 0.5
        if pattern_candidates:
            strongest = max(pattern_candidates, key=lambda x: abs(x["p_big"] - 0.5))
            pattern_bias = strongest["p_big"]

        trend_prob, trend_reason = trend_side(history)

        side_prob = 0.5
        side_prob = 0.55 * side_prob + 0.45 * (p_big if p_big >= p_small else p_small)
        side_prob = 0.70 * side_prob + 0.30 * pattern_bias
        side_prob = 0.80 * side_prob + 0.20 * trend_prob

        loss_streak = self.db.get_loss_streak()
        self.recovery.update(loss_streak)
        if loss_streak >= 1:
            side_prob, recovery_used = self.recovery.adjust(side_prob, loss_streak)
        else:
            recovery_used = False

        # neutralise weak side evidence
        if len(nums) >= 20:
            recent_ratio = sum(1 for x in nums[-20:] if x >= 5) / 20.0
            if abs(recent_ratio - 0.5) < 0.08:
                side_prob = 0.5 + (side_prob - 0.5) * 0.45

        final_side = "BIG" if side_prob >= 0.5 else "SMALL"
        allowed = range(5, 10) if final_side == "BIG" else range(0, 5)

        # exact-number final distribution must match side
        final_dist = [0.0] * 10
        for k in range(10):
            final_dist[k] = exact[k]

        for k in range(10):
            if k in allowed:
                final_dist[k] *= 1.6
            else:
                final_dist[k] *= 0.35

        final_dist = normalize(final_dist)

        # recalc side from same final distribution -> consistency
        final_p_big = sum(final_dist[5:10])
        final_p_small = sum(final_dist[:5])
        if final_p_big >= final_p_small:
            final_side = "BIG"
            allowed = range(5, 10)
        else:
            final_side = "SMALL"
            allowed = range(0, 5)

        side_adjusted = [0.0] * 10
        for k in range(10):
            if k in allowed:
                side_adjusted[k] = final_dist[k] * 1.8
            else:
                side_adjusted[k] = final_dist[k] * 0.3

        total_side = sum(side_adjusted)
        if total_side <= 0:
            side_adjusted = [0.1] * 10
            total_side = 10.0
        side_adjusted = [x / total_side for x in side_adjusted]

        num = max(allowed, key=lambda k: side_adjusted[k])

        payload = {
            "issue": issue,
            "side": final_side,
            "num": num,
            "p": max(side_prob, 1.0 - side_prob),
            "num_probability": side_adjusted[num],
            "exact_distribution": side_adjusted,
            "is_recovery": recovery_used or (loss_streak > 0),
            "loss_streak": loss_streak,
            "source": "PATTERN + TREND + SEQUENCE",
            "pattern_trace": pattern_trace,
            "pattern_display": pattern_display,
            "trend_reason": trend_reason,
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
            predicted_side = payload.get("side", "BIG" if payload.get("p", 0.5) >= 0.5 else "SMALL")

            # record every pattern snapshot against result
            for item in payload.get("pattern_trace", []):
                key = "pattern:" + item["key"]
                memory = self.db.get(key, {"hits": 0, "misses": 0, "score": 0.5})
                hit = predicted_side == actual_side
                memory["hits"] = int(memory.get("hits", 0)) + int(hit)
                memory["misses"] = int(memory.get("misses", 0)) + int(not hit)
                if hit:
                    memory["score"] = 0.8 * float(memory.get("score", 0.5)) + 0.2 * 1.0
                else:
                    memory["score"] = 0.8 * float(memory.get("score", 0.5)) + 0.2 * 0.0
                self.db.set(key, memory)

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

    pattern_display = escape(p.get("pattern_display", "—"))
    trend_reason = escape(p.get("trend_reason", "neutral"))

    return (
        f"{rec_tag}⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {'🔴' if side == 'BIG' else '🔵'} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🧩 <b>PATTERN:</b> <code>{pattern_display}</code>\n"
        f"📈 <b>TREND:</b> {trend_reason}\n"
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
