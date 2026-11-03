"""
FTW Predictor V34 — Hybrid (Pattern + Safe Adaptive)
Python 3.10+
Dependencies: aiohttp
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
    # 🔥 Default Token aur Chat ID
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_hybrid_v34.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500 # Render safe limit
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (HYBRID V34)"

    # 🔥 Stickers
    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN environment variable is required")
        if not self.chat:
            raise ValueError("TELEGRAM_CHAT environment variable is required")
        if not self.urls:
            raise ValueError("Set at least one URL in HISTORY_URLS")

CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v34")


# ============================================================
# HELPERS & PATTERN MATCHING
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

class QuantumTracker:
    def __init__(self, min_len=3, max_len=8, min_support=12):
        self.min_len = min_len
        self.max_len = max_len
        self.min_support = min_support

    def match_tail_patterns(self, seq: list[int]) -> list[dict]:
        n = len(seq)
        if n < 18: return []
        results = []
        for length in range(self.max_len, self.min_len - 1, -1):
            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs = []
            for i in range(n - length):
                if family_signature(seq[i:i + length]) == sig:
                    w = recency_weight(i + length, n, 420.0)
                    obs.append((seq[i + length], w))
            if len(obs) < self.min_support: continue
            
            p, eff_n = weighted_probability(obs)
            score = abs(p - 0.5) * math.sqrt(max(1.0, eff_n))
            if score > 0.0:
                pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in sig)
                results.append({
                    "sig": sig, "p": p, "support": len(obs), "eff_n": eff_n, 
                    "score": score, "pattern_str": pattern_str
                })
            break
        return results

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
        existing = records.get(issue_key)
        if existing and existing["num"] != number:
            raise ValueError(f"Conflicting duplicate issue in source: {issue_key}")
        records[issue_key] = record
    return sorted(records.values(), key=lambda row: int(row["issue"]))


# ============================================================
# MODEL
# ============================================================

EXPERT_NAMES = (
    "patterns", "base", "recent", "markov1", "markov2", "markov3", "run", "regime", "momentum", "reversal"
)

class Engine:
    def __init__(self, db: "Database"):
        self.db = db

    def _mine_experts(self, seq: list[int]) -> tuple[dict[str, float], str]:
        n = len(seq)
        if not n: return {name: 0.5 for name in EXPERT_NAMES}, "—"

        experts = {}
        pattern_display = "—"

        # 1. Patterns (Hybrid Addition)
        tracker = QuantumTracker(min_len=3, max_len=8, min_support=CFG.min_support)
        pats = tracker.match_tail_patterns(seq)
        if pats:
            pattern_display = pats[0]["pattern_str"]
            candidates = []
            for item in pats:
                key = "pattern:" + item["pattern_str"]
                # Note: adapt_expert takes (name, regime, probability)
                adjusted, rel = self.db.adapt_expert(key, get_regime(seq), item["p"])
                candidates.append((adjusted, rel * math.sqrt(max(1.0, item["eff_n"]))))
            if candidates:
                num = sum(p * w for p, w in candidates)
                den = sum(w for _, w in candidates)
                if den > 0: experts["patterns"] = clip(num / den)

        # 2. Statistical Baseline
        experts["base"] = weighted_probability([(val, recency_weight(i, n)) for i, val in enumerate(seq)])[0]
        recent = seq[-100:]
        experts["recent"] = (4.0 + sum(recent)) / (8.0 + len(recent))

        # 3. Markov Chains
        for order in (1, 2, 3):
            if n <= order: continue
            context = tuple(seq[-order:])
            obs = []
            for i in range(order, n - order + 1):
                if tuple(seq[i - order:i]) == context:
                    obs.append((seq[i], recency_weight(i, n)))
            if len(obs) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(obs)[0]

        # 4. Run continuation
        if n >= 2:
            last_value = seq[-1]
            current_run = 1
            for value in reversed(seq[:-1]):
                if value != last_value: break
                current_run += 1
            run_length_at = [1] * n
            for i in range(1, n):
                run_length_at[i] = run_length_at[i - 1] + 1 if seq[i] == seq[i - 1] else 1
            obs = []
            for i in range(1, n):
                if seq[i - 1] == last_value and min(run_length_at[i - 1], 8) == min(current_run, 8):
                    obs.append((seq[i], recency_weight(i, n)))
            if len(obs) >= CFG.min_support:
                experts["run"] = weighted_probability(obs)[0]

        # 5. Regime
        regime = get_regime(seq)
        obs = []
        for i in range(24, n):
            if get_regime(seq[max(0, i - 96):i]) == regime:
                obs.append((seq[i], recency_weight(i, n)))
        if len(obs) >= CFG.min_support:
            experts["regime"] = weighted_probability(obs)[0]

        # 6. Momentum & Reversal
        momentum_tail = seq[-8:]
        momentum = (4.0 + sum(momentum_tail)) / (8.0 + len(momentum_tail))
        experts["momentum"] = momentum
        experts["reversal"] = 1.0 - momentum

        for name in EXPERT_NAMES:
            experts[name] = clip(experts.get(name, 0.5), 0.05, 0.95)

        return experts, pattern_display

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(row["size"]) for row in history]
        experts, pattern_display = self._mine_experts(seq)
        regime = get_regime(seq)

        weighted_sum = 0.0
        total_weight = 0.0
        for name, probability in experts.items():
            if name == "patterns":
                adjusted, reliability = self.db.adapt_expert("patterns", regime, probability)
                reliability *= 3.0 # Boost pattern weight
            else:
                adjusted, reliability = self.db.adapt_expert(name, regime, probability)
            weighted_sum += adjusted * reliability
            total_weight += reliability

        p = weighted_sum / total_weight if total_weight else 0.5
        p = clip(p)

        side = "BIG" if p >= 0.5 else "SMALL"
        allowed_numbers = range(5, 10) if side == "BIG" else range(0, 5)
        number_distribution = exact_number_distribution(history)
        ranked = sorted(allowed_numbers, key=lambda number: number_distribution[number], reverse=True)
        number_prediction = ranked[0]

        return {
            "issue": issue,
            "p": p,
            "side": side,
            "number": number_prediction,
            "number_distribution": number_distribution,
            "regime": regime,
            "pattern": pattern_display,
            "experts": experts,
            "source": "Hybrid V34",
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {row["issue"]: row for row in history}
        pending = self.db.pending_predictions()

        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None: continue
            try: payload = json.loads(row["payload"])
            except: continue

            actual_size = int(actual["size"])
            predicted_side = int(float(payload.get("p", 0.5)) >= 0.5)
            side_won = predicted_side == actual_size
            number_won = payload.get("number") == actual["num"]

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=?, won_number=? WHERE issue=? AND status='PENDING'",
                    ("WIN" if side_won else "LOSS", actual["num"], int(number_won), row["issue"]),
                )
                regime = str(payload.get("regime", "MIXED"))
                saved_experts = payload.get("experts", {})
                
                if isinstance(saved_experts, dict):
                    for name, probability in saved_experts.items():
                        try:
                            self.db.learn_expert(str(name), regime, float(probability), actual_size)
                        except: pass

                # Learn pattern specifically
                pat_str = payload.get("pattern")
                if pat_str and pat_str != "—" and "patterns" in saved_experts:
                    self.db.learn_expert("pattern:" + pat_str, regime, float(saved_experts["patterns"]), actual_size)

                sticker = CFG.win_sticker if side_won else CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)


def exact_number_distribution(history: list[dict[str, Any]]) -> list[float]:
    nums = [int(row["num"]) for row in history]
    if not nums: return [0.1] * 10
    def frequency(window: int) -> list[float]:
        recent = nums[-window:]
        counts = Counter(recent)
        return [(counts.get(n, 0) + 1.0) / (len(recent) + 10.0) for n in range(10)]
    def markov(order: int) -> list[float]:
        if len(nums) <= order: return frequency(100)
        context = tuple(nums[-order:])
        counts = [0.0] * 10
        for i in range(order, len(nums) - order + 1):
            if tuple(nums[i - order:i]) == context: counts[nums[i]] += 1.0
        total = sum(counts)
        return [(c + 1.0) / (total + 10.0) for c in counts] if total > 0 else frequency(100)

    components = [(0.45, frequency(80)), (0.30, frequency(200)), (0.15, markov(1)), (0.10, markov(2))]
    distribution = [sum(w * comp[n] for w, comp in components) for n in range(10)]
    total = sum(distribution)
    return [v / total for v in distribution]


def format_prediction(payload: dict[str, Any]) -> str:
    probability = float(payload["p"])
    confidence = max(probability, 1.0 - probability) * 100
    filled = max(0, min(10, int(confidence // 10)))
    bar = "█" * filled + "░" * (10 - filled)
    side = escape(payload["side"])
    marker = "🔴" if payload["side"] == "BIG" else "🔵"
    
    pat_str = payload.get("pattern", "—")
    pat_line = f"\n🧩 <b>PATTERN:</b> <code>{escape(pat_str)}</code>" if pat_str != "—" else ""

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"🔢 <b>TOP NUMBER:</b> <code>{int(payload['number'])}</code>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>MODEL CONFIDENCE:</b> {confidence:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])}{pat_line}\n"
        "ℹ️ Experimental estimate; outcomes are not guaranteed.\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# DATABASE & RUNTIME
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
            CREATE TABLE IF NOT EXISTS expert_models(name TEXT NOT NULL, regime TEXT NOT NULL, n REAL NOT NULL, loss REAL NOT NULL, PRIMARY KEY(name, regime));
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

    def learn_expert(self, name: str, regime: str, probability: float, actual: int) -> None:
        row = self.c.execute("SELECT n,loss FROM expert_models WHERE name=? AND regime=?", (name, regime)).fetchone()
        old_n, old_loss = (float(row["n"]), float(row["loss"])) if row else (0.0, 0.0)
        n = 0.985 * old_n + 1.0
        loss = 0.985 * old_loss + (clip(probability) - actual) ** 2
        self.c.execute("INSERT INTO expert_models(name,regime,n,loss) VALUES (?,?,?,?) ON CONFLICT(name,regime) DO UPDATE SET n=excluded.n, loss=excluded.loss", (name, regime, n, loss))

    def adapt_expert(self, name: str, regime: str, probability: float) -> tuple[float, float]:
        row = self.c.execute("SELECT n,loss FROM expert_models WHERE name=? AND regime=?", (name, regime)).fetchone()
        if not row or float(row["n"]) < 1.0: return 0.5 + (probability - 0.5) * 0.5, 1.0
        n, mean_loss = float(row["n"]), float(row["loss"]) / float(row["n"])
        evidence = min(n / 40.0, 1.0)
        adjusted = 0.5 + (probability - 0.5) * (0.35 + 0.65 * evidence)
        reliability = math.exp(-3.0 * max(-0.10, mean_loss - 0.25))
        return clip(adjusted), max(0.35, min(1.8, reliability))

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)", (kind, issue, text, time.time(), expires))

    def health_stats(self) -> dict[str, Any]:
        settled = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0])
        wins = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0])
        return {
            "history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]),
            "settled": settled,
            "side_wins": wins,
            "side_win_rate": round(100.0 * wins / settled, 2) if settled else None,
        }

    def close(self) -> None:
        self.c.close()


class Runtime:
    def __init__(self, db: Database):
        self.db, self.engine = db, Engine(db)
        self.stop_event = asyncio.Event()
        self.last_fetch = 0.0

    async def pause(self, seconds: float) -> None:
        try: await asyncio.wait_for(self.stop_event.wait(), timeout=max(0.05, seconds))
        except asyncio.TimeoutError: pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def fetch_one(url: str):
            try:
                async with session.get(url) as res:
                    res.raise_for_status()
                    return parse_history(await res.json(content_type=None))
            except Exception: return []
        
        results = await asyncio.gather(*(fetch_one(u) for u in CFG.urls))
        merged = {}
        for rows in results:
            for row in rows: merged[row["issue"]] = row
        return sorted(merged.values(), key=lambda r: int(r["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        while not self.stop_event.is_set():
            try:
                records = await self.fetch(session)
                self.db.ingest(records)
                history = self.db.history()
                self.engine.settle(history)
                self.last_fetch = time.time()
                failures = 0

                if len(history) >= CFG.min_history:
                    next_issue = str(int(history[-1]["issue"]) + 1)
                    if not self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (next_issue,)).fetchone():
                        pred = self.engine.predict(history, next_issue)
                        msg = format_prediction(pred)
                        with self.db.transaction():
                            self.db.c.execute("INSERT INTO predictions(issue,created,status,payload,actual,won_number) VALUES (?,?,?, ?,NULL,0)", (next_issue, time.time(), "PENDING", json.dumps(pred)))
                            self.db.queue("prediction", next_issue, msg, time.time() + CFG.delivery_ttl)
            except asyncio.CancelledError: raise
            except Exception as e: 
                failures += 1
                LOG.error("Worker error: %s", e)
            await self.pause(min(60.0, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        backoff = 1.0
        while not self.stop_event.is_set():
            self.db.c.execute("UPDATE outbox SET sent=2 WHERE sent=0 AND expires IS NOT NULL AND expires <= ?", (time.time(),))
            row = self.db.c.execute("SELECT id,kind,text FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1").fetchone()
            if not row: 
                await self.pause(0.5)
                continue

            is_sticker = row["kind"] == "sticker"
            url = f"https://api.telegram.org/bot{CFG.token}/{'sendSticker' if is_sticker else 'sendMessage'}"
            payload = {"chat_id": CFG.chat, "sticker": row["text"]} if is_sticker else {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}
            
            delay = backoff
            try:
                async with session.post(url, json=payload) as res:
                    data = await res.json(content_type=None)
                    if res.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue
                    code = data.get("error_code", res.status)
                    if code == 429: delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400, 401, 403, 404): delay = 60
            except Exception as e:
                LOG.warning("TG error: %s", e)
                delay = 10
            await self.pause(delay)
            backoff = min(60.0, backoff * 2.0)

    async def health(self, request: web.Request) -> web.Response:
        fresh = self.last_fetch > 0 and time.time() - self.last_fetch < 90
        return web.json_response({"status": "ok" if fresh else "degraded", **self.db.health_stats()})


async def main() -> None:
    CFG.validate()
    db, runtime = Database(CFG.db_path), Runtime(Database(CFG.db_path))
    app = web.Application()
    app.router.add_get("/", runtime.health)
    runner = web.AppRunner(app)
    
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError): loop.add_signal_handler(sig, runtime.stop_event.set)

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
