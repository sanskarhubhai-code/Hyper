"""
FTW Predictor V46 — ULTIMATE QUANT (Sub-Pattern Voting + CUSUM Override)
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
    db_path: str = os.getenv("DB_PATH", "predictor_v46.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500 
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (V46 ULTIMATE)"

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
LOG = logging.getLogger("predictor_v46")


# ============================================================
# HELPERS & MATH
# ============================================================

def clip(value: float, low: float = 0.02, high: float = 0.98) -> float:
    return max(low, min(high, float(value)))

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

def cusum_change(seq: list[int], threshold=2.5) -> int:
    if len(seq) < 20: return 0
    mean = sum(seq[-20:]) / 20.0
    s_pos, s_neg = 0.0, 0.0
    for val in seq[-20:]:
        s_pos = max(0, s_pos + (val - mean) - 0.5)
        s_neg = min(0, s_neg + (val - mean) + 0.5)
    if s_pos > threshold: return 1
    if s_neg < -threshold: return -1
    return 0

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
# SMART PATTERN MINER (Bayesian Sub-pattern Voting)
# ============================================================

class SmartPatternMiner:
    def __init__(self, db: "Database"):
        self.db = db

    def find_best_size_pattern(self, seq: list[int], regime: str) -> tuple[float, str, int, int]:
        n = len(seq)
        if n < 15: return 0.5, "—", 0, 0
        
        # 🔥 We collect all patterns from 8 down to 5, then vote
        candidates = []
        
        for length in range(8, 4, -1):
            if n < length * 2: continue
            
            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs = []
            
            for i in range(n - length):
                if family_signature(seq[i:i + length]) == sig:
                    obs.append((seq[i + length], recency_weight(i + length, n, 420.0)))
            
            if len(obs) >= CFG.min_support:
                p, eff = weighted_probability(obs)
                big_c = sum(1 for val, _ in obs if val == 1)
                small_c = len(obs) - big_c
                
                pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in sig)
                key = "size_family:" + pattern_str
                
                adjusted_p, rel = self.db.adapt_pattern(key, regime, p, is_number=False)
                
                if rel < 0.15: continue # Skip terrible patterns
                
                # Score based on reliability and sample size
                score = rel * math.sqrt(max(1.0, eff))
                candidates.append((score, adjusted_p, pattern_str, big_c, small_c))

        if not candidates:
            return 0.5, "—", 0, 0
            
        # Sort by highest score (most reliable)
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 🔥 Bayesian Voting: If the top pattern is really strong, use it. 
        # Otherwise, blend the top 2-3 patterns.
        best_score, best_p, best_name, best_big, best_small = candidates[0]
        
        # Blend the next best if it has a similar score
        if len(candidates) > 1:
            score2, p2, name2, big2, small2 = candidates[1]
            if score2 > best_score * 0.8: # If second pattern is close in reliability
                best_p = (best_p * 0.6) + (p2 * 0.4)
                best_big += big2
                best_small += small2
                best_name += f" & {name2}"

        return best_p, best_name, best_big, best_small


# ============================================================
# NUMBER TRACKER
# ============================================================

def track_numbers(nums: list[int]) -> tuple[list[float], str]:
    if not nums: return [0.1] * 10, "No Data"
    
    n = len(nums)
    
    def freq(window: int) -> list[float]:
        recent = nums[-window:]
        counts = Counter(recent)
        total = len(recent)
        return [(counts.get(k, 0) + 1.0) / (total + 10.0) for k in range(10)]

    f30 = freq(30)
    f100 = freq(100)
    
    trans = [0.1] * 10
    if n >= 3:
        ctx = tuple(nums[-3:])
        obs = {}
        total = 0
        for i in range(3, n):
            if tuple(nums[i-3:i]) == ctx:
                nxt = nums[i]
                obs[nxt] = obs.get(nxt, 0) + 1
                total += 1
        if total >= 3:
            trans = [(obs.get(k, 0) + 1.0) / (total + 10.0) for k in range(10)]
    
    exact = [0.1] * 10
    if n >= 10:
        ctx = tuple(nums[-5:])
        obs = {}
        total = 0
        for i in range(5, n):
            if tuple(nums[i-5:i]) == ctx:
                nxt = nums[i]
                obs[nxt] = obs.get(nxt, 0) + 1
                total += 1
        if total >= 2:
            exact = [(obs.get(k, 0) + 1.0) / (total + 10.0) for k in range(10)]
    
    final = [0.0] * 10
    for k in range(10):
        final[k] = 0.15 * f30[k] + 0.25 * f100[k] + 0.30 * trans[k] + 0.30 * exact[k]
    
    total = sum(final)
    if total > 0: final = [x / total for x in final]
    
    return final, "Freq+Markov+Exact"


# ============================================================
# DATABASE
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
            CREATE TABLE IF NOT EXISTS pattern_models(name TEXT PRIMARY KEY, n REAL NOT NULL, loss REAL NOT NULL, failures INTEGER NOT NULL);
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

    def learn_pattern(self, key: str, probability: float, actual: int) -> None:
        row = self.c.execute("SELECT n, loss, failures FROM pattern_models WHERE name=?", (key,)).fetchone()
        old_n, old_loss, old_fails = (float(row["n"]), float(row["loss"]), int(row["failures"])) if row else (0.0, 0.0, 0)
        n = 0.985 * old_n + 1.0
        loss = 0.985 * old_loss + (clip(probability) - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        fails = old_fails + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO pattern_models(name,n,loss,failures) VALUES (?,?,?,?) ON CONFLICT(name) DO UPDATE SET n=excluded.n, loss=excluded.loss, failures=excluded.failures",
            (key, n, loss, fails)
        )

    def adapt_pattern(self, key: str, regime: str, probability: float, is_number: bool = False) -> tuple[float, float]:
        row = self.c.execute("SELECT n, loss, failures FROM pattern_models WHERE name=?", (key,)).fetchone()
        if not row or float(row["n"]) < 2.0: return probability, 1.0
        n = float(row["n"])
        mean_loss = float(row["loss"]) / n
        fails = int(row["failures"])
        reliability = math.exp(-4.0 * max(-0.10, mean_loss - 0.25))
        if fails >= 2: reliability *= 0.2
        if fails >= 4: reliability *= 0.05
        if fails >= 3 and mean_loss > 0.26: reliability *= 0.01
        return clip(probability), max(0.05, min(2.0, reliability))

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)", (kind, issue, text, time.time(), expires))

    def health_stats(self) -> dict[str, Any]:
        settled = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0])
        wins = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0])
        return {"history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]), "settled": settled, "side_wins": wins, "side_win_rate": round(100.0 * wins / settled, 2) if settled else None}

    def close(self) -> None: self.c.close()


# ============================================================
# ENGINE
# ============================================================

class Engine:
    def __init__(self, db: Database):
        self.db = db
        self.miner = SmartPatternMiner(db)

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(row["size"]) for row in history]
        nums = [int(row["num"]) for row in history]
        regime = get_regime(seq)

        pat_p, pat_name, pat_big, pat_small = self.miner.find_best_size_pattern(seq, regime)
        num_dist, num_info = track_numbers(nums)
        
        # 🔥 CUSUM TREND OVERRIDE
        cusum_dir = cusum_change(seq)
        
        if pat_name != "—":
            side = "BIG" if pat_big >= pat_small else "SMALL"
            p = 0.5 + abs(pat_big - pat_small) / max(1, pat_big + pat_small) * 0.5
            
            # If CUSUM says trend has switched, and pattern is betting on the old trend, override
            if cusum_dir == 1 and side == "SMALL":
                side = "BIG"
                p = 0.55
                pat_name += " (CUSUM Reversal)"
            elif cusum_dir == -1 and side == "BIG":
                side = "SMALL"
                p = 0.55
                pat_name += " (CUSUM Reversal)"
        else:
            recent_big = sum(seq[-20:]) / 20.0
            side = "BIG" if recent_big >= 0.5 else "SMALL"
            p = 0.5 + abs(recent_big - 0.5) * 0.4

        allowed_nums = range(5, 10) if side == "BIG" else range(0, 5)
        
        num_dist_final = [0.0] * 10
        for k in range(10):
            if k in allowed_nums:
                num_dist_final[k] = num_dist[k] * 1.5
            else:
                num_dist_final[k] = num_dist[k] * 0.1
        
        total_num = sum(num_dist_final)
        if total_num > 0:
            num_dist_final = [x / total_num for x in num_dist_final]
        
        recent_counts = Counter(nums[-50:])
        num_pred = max(allowed_nums, key=lambda k: (num_dist_final[k], recent_counts.get(k, 0)))
        num_conf = num_dist_final[num_pred]

        return {
            "issue": issue,
            "p": clip(p),
            "side": side,
            "number": num_pred,
            "num_conf": num_conf,
            "regime": regime,
            "pattern": pat_name,
            "pat_big": pat_big,
            "pat_small": pat_small,
            "num_tracking": num_info,
            "source": "V46 Ultimate",
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        for row in self.db.pending_predictions():
            actual = lookup.get(row["issue"])
            if not actual: continue

            payload = json.loads(row["payload"])
            actual_size = int(actual["size"])
            actual_num = int(actual["num"])
            won_side = int(float(payload["p"]) >= 0.5) == actual_size
            
            num1 = payload.get("number")
            won_num = actual_num == num1

            with self.db.transaction():
                self.db.c.execute("UPDATE predictions SET status=?, actual=?, won_number=? WHERE issue=? AND status='PENDING'", ("WIN" if won_side else "LOSS", actual_num, int(won_num), row["issue"]))
                
                pat_name = payload.get("pattern")
                if pat_name and pat_name != "—":
                    # We clean the "CUSUM" tag to learn the base pattern
                    clean_pat = pat_name.replace(" (CUSUM Reversal)", "")
                    self.db.learn_pattern("size_family:" + clean_pat, float(payload["p"]), actual_size)

                sticker = CFG.jack_sticker if (won_side and won_num) else CFG.win_sticker if won_side else CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)


def format_prediction(payload: dict[str, Any]) -> str:
    prob = float(payload["p"])
    conf = max(prob, 1.0 - prob) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = "BIG" if payload["side"] == "BIG" else "SMALL"
    marker = "🔴" if side == "BIG" else "🔵"
    
    pat = payload.get("pattern", "—")
    pat_line = ""
    if pat != "—":
        big_c = payload.get("pat_big", 0)
        small_c = payload.get("pat_small", 0)
        pat_line = f"\n🧩 <b>SIZE PATTERN:</b> <code>{escape(pat)}</code>"
        pat_line += f"\n📊 <b>PATTERN STATS:</b> BIG[{big_c}] | SMALL[{small_c}]"

    num_tracking = payload.get("num_tracking", "—")
    num_line = f"\n🔢 <b>NUM TRACKING:</b> <code>{escape(num_tracking)}</code>"
    
    n1 = int(payload["number"])
    nconf = payload.get("num_conf", 0) * 100

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"🔢 <b>TOP NUMBER:</b> <code>{n1}</code> ({nconf:.1f}%)\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])}{pat_line}{num_line}\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# RUNTIME
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
