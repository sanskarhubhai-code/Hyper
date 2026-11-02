
import asyncio
import contextlib
import html
import json
import logging
import math
import os
import random
import signal
import socket
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass

import aiohttp
from aiohttp import web

@dataclass(frozen=True)
class Config:
    token: str = "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c"
    chat: str = "-1004368817214"
    urls: tuple = (
        "https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",
    )
    db_path: str = os.getenv("DB_PATH", "predictor_hyper_v21.db") # Naya DB
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 4000
    min_history: int = 120
    min_support: int = 8
    learning_rate: float = 0.035
    meta_lr: float = 0.03
    decay: float = 0.985
    max_pattern: int = 15
    delivery_ttl: float = 120.0
    name: str = "FTW VIP PREDICTOR (HYPER)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor")

# ... [Tere code ke saare functions: clip, logit, sigmoid, family_run, AdvancedFamilyTracker, trend_features, EXPERT_NAMES, Database, Engine] yahan same rahenge ...

# ─── FORMATTER ──────────────────────────────────────────────
def format_prediction(p):
    conf = max(p["p"], 1 - p["p"]) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if p["side"] == "BIG" else "SMALL"
    color = "🔴" if side == "BIG" else "🔵"
    quality = "🔥 HIGH VIP" if conf >= 65 else "⭐ VIP" if conf >= 60 else "✅ NORMAL"
    stake = "1.5x" if conf >= 65 else "1x" if conf >= 60 else "0.5x"
    shift_text = f"\n🔄 <b>TREND SHIFT:</b> {escape(p['trend_shift'])}" if p.get("trend_shift") else ""
    pat_text = f"\n🧩 <b>PATTERN:</b> {p.get('pattern_str', 'None')}" if p.get("pattern_str") and p.get("pattern_str") != "None" else ""
    source_text = f"\n🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}"
    edge_text = f"\n📐 <b>EDGE:</b> {p.get('edge', 0) * 100:.1f}% | <b>CI:</b> {conf:.1f}%" if p.get("edge") else ""

    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {color} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake}){edge_text}\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime', 'N/A'))}{source_text}{shift_text}{pat_text}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )

# ─── RUNTIME ─────────────────────────────────────────────────
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
                failures = 0

                self.engine.settle(history)

                latest = int(history[-1]["issue"])
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    exists = self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone()

                    if not exists:
                        p = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions VALUES (?,?,?,?,NULL)",
                                (issue, time.time(), "PENDING", json.dumps(p)),
                            )
                            self.db.queue(
                                "prediction",
                                issue,
                                format_prediction(p),
                                time.time() + CFG.delivery_ttl,
                            )

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
                    # ✅ YAHAN LOG ADD KIYA HAI ERROR PAKADNE KE LIYE
                    LOG.error(f"Telegram API Error: {data}")
                    
                    if code == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404):
                        delay = 60
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning(f"Telegram Network Fail: {e}")
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
            "error": self.error
        })

    # ✅ NAYA TEST ENDPOINT
    async def test_send(self, request):
        async with aiohttp.ClientSession() as s:
            async with s.post(
                f"https://api.telegram.org/bot{CFG.token}/sendMessage",
                json={
                    "chat_id": CFG.chat,
                    "text": "🧪 <b>TEST MSG</b>\nAgar yeh dikh raha hai matlab bot sahi kaam kar raha hai!",
                    "parse_mode": "HTML",
                },
            ) as r:
                return web.json_response(await r.json(content_type=None))

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
    app.router.add_get("/test", runtime.test_send) # ✅ TEST ROUTE ADD KIYA
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
            LOG.info(f"{CFG.name} Started...")
            await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
            runtime.stop.set()
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc: raise exc
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        runtime.stop.set()
        for task in tasks: task.cancel()
        if stop_task: stop_task.cancel()
        await asyncio.gather(*tasks, *([stop_task] if stop_task else []), return_exceptions=True)
        await runner.cleanup()
        db.c.close()

if __name__ == "__main__":
    asyncio.run(main())
