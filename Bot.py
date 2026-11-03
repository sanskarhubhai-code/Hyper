"""
FTW Predictor V33 — safer adaptive baseline
Python 3.10+
Dependencies: aiohttp

Environment variables:
  TELEGRAM_TOKEN      Required: newly rotated Telegram bot token
  TELEGRAM_CHAT       Required: destination chat ID
  HISTORY_URLS        Optional: comma-separated history API URLs
  DB_PATH             Optional: SQLite path (default: predictor_v33.db)
  PORT                Optional: health endpoint port (default: 8080)
  FETCH_EVERY         Optional: seconds between successful polls (default: 3)
  MIN_HISTORY         Optional: minimum history before predicting (default: 180)

This is an experimental statistical model, not a guarantee of future outcomes.
It intentionally does not recommend stakes or increase stakes after losses.
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
    # 🔥 Default token aur chat ID set kar diya hai
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_v33.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "12"))
    delivery_ttl: float = 600.0
    name: str = "FTW PREDICTOR (V33)"

    # 🔥 Stickers add kar diye hain warna bot crash hota
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
        if self.port < 1 or self.port > 65535:
            raise ValueError("PORT must be between 1 and 65535")
        if self.fetch_every <= 0:
            raise ValueError("FETCH_EVERY must be greater than zero")
        if self.min_history < 1:
            raise ValueError("MIN_HISTORY must be at least 1")


CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v33")


# ============================================================
# HELPERS
# ============================================================

def clip(value: float, low: float = 0.02, high: float = 0.98) -> float:
    return max(low, min(high, float(value)))


def escape(value: Any) -> str:
    return html.escape(str(value))


def numeric(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value).strip()
    if not text.isascii() or not text.isdigit():
        return None
    try:
        return int(text)
    except ValueError:
        return None


def weighted_probability(observations: list[tuple[int, float]]) -> tuple[float, float]:
    """Smoothed probability and effective weighted sample size."""
    if not observations:
        return 0.5, 0.0

    weight_sum = sum(weight for _, weight in observations)
    squared_weight_sum = sum(weight * weight for _, weight in observations)
    effective_n = (
        weight_sum * weight_sum / squared_weight_sum
        if squared_weight_sum > 0 else 0.0
    )
    probability = (
        4.0 + sum(value * weight for value, weight in observations)
    ) / (8.0 + weight_sum)

    return clip(probability), effective_n


def recency_weight(index: int, n: int, half_life: float = 330.0) -> float:
    age = max(0, n - 1 - index)
    return 0.5 ** (age / half_life)


def get_regime(seq: list[int]) -> str:
    tail = seq[-24:]
    if len(tail) < 8:
        return "COLD"

    flips = sum(a != b for a, b in zip(tail, tail[1:]))
    flip_rate = flips / max(1, len(tail) - 1)

    run = 1
    for value in reversed(tail[:-1]):
        if value != tail[-1]:
            break
        run += 1

    if run >= 5:
        return "RUN"
    if flip_rate >= 0.72:
        return "ALT"
    if abs(sum(tail) / len(tail) - 0.5) > 0.20:
        return "IMBAL"
    return "MIXED"


def parse_history(payload: Any) -> list[dict[str, Any]]:
    """Extract issue/number rows from common nested API response shapes."""
    obj = payload

    for _ in range(8):
        if isinstance(obj, list):
            break
        if not isinstance(obj, dict):
            return []

        next_obj = None
        for key in ("data", "list", "result", "items", "history", "records"):
            candidate = obj.get(key)
            if isinstance(candidate, (list, dict)):
                next_obj = candidate
                break
        if next_obj is None:
            return []
        obj = next_obj

    if not isinstance(obj, list):
        return []

    records: dict[str, dict[str, Any]] = {}
    for item in obj:
        if not isinstance(item, dict):
            continue

        issue_value = item.get("issueNumber", item.get("issue"))
        number_value = item.get("number", item.get("value"))
        issue = numeric(issue_value)
        number = numeric(number_value)

        if issue is None or number is None or not 0 <= number <= 9:
            continue

        issue_key = str(issue)
        record = {
            "issue": issue_key,
            "num": number,
            "size": int(number >= 5),
        }

        existing = records.get(issue_key)
        if existing and existing["num"] != number:
            raise ValueError(f"Conflicting duplicate issue in source: {issue_key}")
        records[issue_key] = record

    return sorted(records.values(), key=lambda row: int(row["issue"]))


# ============================================================
# MODEL
# ============================================================

EXPERT_NAMES = (
    "base",
    "recent",
    "markov1",
    "markov2",
    "markov3",
    "run",
    "regime",
    "momentum",
    "reversal",
)


class Engine:
    def __init__(self, db: "Database"):
        self.db = db

    def _mine_experts(self, seq: list[int]) -> dict[str, float]:
        n = len(seq)
        if not n:
            return {name: 0.5 for name in EXPERT_NAMES}

        experts: dict[str, float] = {}

        # Recency-weighted overall baseline.
        base_obs = [
            (value, recency_weight(i, n))
            for i, value in enumerate(seq)
        ]
        experts["base"] = weighted_probability(base_obs)[0]

        # Shorter-window estimate; modest smoothing prevents extreme probabilities.
        recent = seq[-100:]
        experts["recent"] = (
            4.0 + sum(recent)
        ) / (8.0 + len(recent))

        # Markov experts: only use contexts observed enough times.
        for order in (1, 2, 3):
            if n <= order:
                continue

            context = tuple(seq[-order:])
            observations: list[tuple[int, float]] = []

            # Exclude the current suffix itself: it has no known next result yet.
            for i in range(order, n - order + 1):
                if tuple(seq[i - order:i]) == context:
                    observations.append((seq[i], recency_weight(i, n)))

            if len(observations) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(observations)[0]

        # Conditional continuation estimate for the current run length.
        if n >= 2:
            last_value = seq[-1]
            current_run = 1
            for value in reversed(seq[:-1]):
                if value != last_value:
                    break
                current_run += 1

            run_length_at = [1] * n
            for i in range(1, n):
                run_length_at[i] = (
                    run_length_at[i - 1] + 1
                    if seq[i] == seq[i - 1] else 1
                )

            observations = []
            for i in range(1, n):
                if (
                    seq[i - 1] == last_value
                    and min(run_length_at[i - 1], 8) == min(current_run, 8)
                ):
                    observations.append((seq[i], recency_weight(i, n)))

            if len(observations) >= CFG.min_support:
                experts["run"] = weighted_probability(observations)[0]

        # Historical estimate for the current coarse regime.
        regime = get_regime(seq)
        observations = []
        for i in range(24, n):
            prior_regime = get_regime(seq[max(0, i - 96):i])
            if prior_regime == regime:
                observations.append((seq[i], recency_weight(i, n)))

        if len(observations) >= CFG.min_support:
            experts["regime"] = weighted_probability(observations)[0]

        # Simple recent momentum and its complementary reversal hypothesis.
        momentum_tail = seq[-8:]
        momentum = (4.0 + sum(momentum_tail)) / (8.0 + len(momentum_tail))
        experts["momentum"] = momentum
        experts["reversal"] = 1.0 - momentum

        # Do not silently omit experts: use neutral estimates when evidence is thin.
        for name in EXPERT_NAMES:
            experts[name] = clip(experts.get(name, 0.5), 0.05, 0.95)

        return experts

    def predict(self, history: list[dict[str, Any]], issue: str) -> dict[str, Any]:
        seq = [int(row["size"]) for row in history]
        experts = self._mine_experts(seq)
        regime = get_regime(seq)

        weighted_sum = 0.0
        total_weight = 0.0
        for name, probability in experts.items():
            adjusted, reliability = self.db.adapt_expert(
                name=name,
                regime=regime,
                probability=probability,
            )
            weighted_sum += adjusted * reliability
            total_weight += reliability

        p = weighted_sum / total_weight if total_weight else 0.5
        p = clip(p)

        side = "BIG" if p >= 0.5 else "SMALL"
        allowed_numbers = range(5, 10) if side == "BIG" else range(0, 5)

        number_distribution = exact_number_distribution(history)
        # This is a display/ranking heuristic, not a calibrated exact-number chance.
        ranked = sorted(
            allowed_numbers,
            key=lambda number: number_distribution[number],
            reverse=True,
        )
        number_prediction = ranked[0]

        return {
            "issue": issue,
            "p": p,
            "side": side,
            "number": number_prediction,
            "number_distribution": number_distribution,
            "regime": regime,
            "experts": experts,
            "source": "Adaptive statistical baseline",
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {row["issue"]: row for row in history}
        pending = self.db.pending_predictions()

        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None:
                continue

            try:
                payload = json.loads(row["payload"])
            except (TypeError, json.JSONDecodeError):
                LOG.exception("Invalid prediction payload for issue %s", row["issue"])
                continue

            actual_size = int(actual["size"])
            predicted_side = int(float(payload.get("p", 0.5)) >= 0.5)
            side_won = predicted_side == actual_size
            number_won = payload.get("number") == actual["num"]

            with self.db.transaction():
                self.db.c.execute(
                    """
                    UPDATE predictions
                    SET status=?, actual=?, won_number=?
                    WHERE issue=? AND status='PENDING'
                    """,
                    (
                        "WIN" if side_won else "LOSS",
                        actual["num"],
                        int(number_won),
                        row["issue"],
                    ),
                )

                regime = str(payload.get("regime", "MIXED"))
                saved_experts = payload.get("experts", {})
                if isinstance(saved_experts, dict):
                    for name, probability in saved_experts.items():
                        try:
                            self.db.learn_expert(
                                str(name),
                                regime,
                                float(probability),
                                actual_size,
                            )
                        except (TypeError, ValueError):
                            LOG.warning(
                                "Skipping invalid expert value for issue %s",
                                row["issue"],
                            )

                sticker = (
                    CFG.win_sticker
                    if side_won
                    else CFG.loss_sticker
                )
                self.db.queue("sticker", row["issue"], sticker)


def exact_number_distribution(
    history: list[dict[str, Any]],
) -> list[float]:
    """Smoothed blended frequency/Markov distribution for digits 0–9."""
    nums = [int(row["num"]) for row in history]
    if not nums:
        return [0.1] * 10

    def frequency(window: int) -> list[float]:
        recent = nums[-window:]
        counts = Counter(recent)
        total = len(recent)
        return [
            (counts.get(number, 0) + 1.0) / (total + 10.0)
            for number in range(10)
        ]

    def markov(order: int) -> list[float]:
        if len(nums) <= order:
            return frequency(100)

        context = tuple(nums[-order:])
        counts = [0.0] * 10

        for i in range(order, len(nums) - order + 1):
            if tuple(nums[i - order:i]) == context:
                counts[nums[i]] += 1.0

        total = sum(counts)
        if total <= 0:
            return frequency(100)

        return [(count + 1.0) / (total + 10.0) for count in counts]

    components = [
        (0.45, frequency(80)),
        (0.30, frequency(200)),
        (0.15, markov(1)),
        (0.10, markov(2)),
    ]

    distribution = [
        sum(weight * component[number] for weight, component in components)
        for number in range(10)
    ]
    total = sum(distribution)
    return [value / total for value in distribution]


def format_prediction(payload: dict[str, Any]) -> str:
    probability = float(payload["p"])
    confidence = max(probability, 1.0 - probability) * 100
    filled = max(0, min(10, int(confidence // 10)))
    bar = "█" * filled + "░" * (10 - filled)

    side = escape(payload["side"])
    marker = "🔴" if payload["side"] == "BIG" else "🔵"

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"🔢 <b>TOP NUMBER:</b> <code>{int(payload['number'])}</code>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>MODEL CONFIDENCE:</b> {confidence:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])}\n"
        "ℹ️ Experimental estimate; outcomes are not guaranteed.\n"
        "━━━━━━━━━━━━━━━━━━"
    )


# ============================================================
# DATABASE
# ============================================================

class Database:
    def __init__(self, path: str):
        self.c = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=20,
        )
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=NORMAL")
        self.c.execute("PRAGMA foreign_keys=ON")
        self.c.execute("PRAGMA busy_timeout=20000")

        self.c.executescript(
            """
            CREATE TABLE IF NOT EXISTS history(
                issue TEXT PRIMARY KEY,
                num INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS predictions(
                issue TEXT PRIMARY KEY,
                created REAL NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                actual INTEGER,
                won_number INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS expert_models(
                name TEXT NOT NULL,
                regime TEXT NOT NULL,
                n REAL NOT NULL,
                loss REAL NOT NULL,
                PRIMARY KEY(name, regime)
            );

            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                issue TEXT,
                text TEXT NOT NULL,
                created REAL NOT NULL,
                expires REAL,
                sent INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._migrate()

    def _migrate(self) -> None:
        columns = {
            row["name"]
            for row in self.c.execute("PRAGMA table_info(predictions)").fetchall()
        }
        if "won_number" not in columns:
            self.c.execute(
                "ALTER TABLE predictions "
                "ADD COLUMN won_number INTEGER NOT NULL DEFAULT 0"
            )

    @contextmanager
    def transaction(self):
        self.c.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            if self.c.in_transaction:
                self.c.execute("ROLLBACK")
            raise
        else:
            if self.c.in_transaction:
                self.c.execute("COMMIT")

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for record in records:
                old = self.c.execute(
                    "SELECT num FROM history WHERE issue=?",
                    (record["issue"],),
                ).fetchone()

                if old and int(old["num"]) != int(record["num"]):
                    raise ValueError(
                        f"Source changed settled issue {record['issue']}"
                    )

                self.c.execute(
                    "INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)",
                    (record["issue"], record["num"]),
                )

            self.c.execute(
                """
                DELETE FROM history
                WHERE issue NOT IN (
                    SELECT issue FROM history
                    ORDER BY CAST(issue AS INTEGER) DESC
                    LIMIT ?
                )
                """,
                (CFG.max_history,),
            )

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute(
            """
            SELECT issue,num
            FROM history
            ORDER BY CAST(issue AS INTEGER) ASC
            """
        ).fetchall()

        return [
            {
                "issue": row["issue"],
                "num": int(row["num"]),
                "size": int(int(row["num"]) >= 5),
            }
            for row in rows
        ]

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute(
            """
            SELECT issue,payload
            FROM predictions
            WHERE status='PENDING'
            ORDER BY CAST(issue AS INTEGER) ASC
            """
        ).fetchall()

    def learn_expert(
        self,
        name: str,
        regime: str,
        probability: float,
        actual: int,
    ) -> None:
        row = self.c.execute(
            """
            SELECT n,loss FROM expert_models
            WHERE name=? AND regime=?
            """,
            (name, regime),
        ).fetchone()

        old_n = float(row["n"]) if row else 0.0
        old_loss = float(row["loss"]) if row else 0.0

        decay = 0.985
        n = decay * old_n + 1.0
        loss = decay * old_loss + (clip(probability) - actual) ** 2

        self.c.execute(
            """
            INSERT INTO expert_models(name,regime,n,loss)
            VALUES (?,?,?,?)
            ON CONFLICT(name,regime) DO UPDATE SET
                n=excluded.n,
                loss=excluded.loss
            """,
            (name, regime, n, loss),
        )

    def adapt_expert(
        self,
        name: str,
        regime: str,
        probability: float,
    ) -> tuple[float, float]:
        row = self.c.execute(
            """
            SELECT n,loss FROM expert_models
            WHERE name=? AND regime=?
            """,
            (name, regime),
        ).fetchone()

        if not row or float(row["n"]) < 1.0:
            # Unknown experts receive neutral shrinkage and ordinary weight.
            return 0.5 + (probability - 0.5) * 0.5, 1.0

        n = float(row["n"])
        mean_loss = float(row["loss"]) / n

        # Shrink early estimates toward 0.5; reliability grows with evidence.
        evidence = min(n / 40.0, 1.0)
        adjusted = 0.5 + (probability - 0.5) * (0.35 + 0.65 * evidence)

        # Brier loss of 0.25 is the neutral reference for binary outcomes.
        reliability = math.exp(-3.0 * max(-0.10, mean_loss - 0.25))
        reliability = max(0.35, min(1.8, reliability))

        return clip(adjusted), reliability

    def queue(
        self,
        kind: str,
        issue: str | None,
        text: str,
        expires: float | None = None,
    ) -> None:
        self.c.execute(
            """
            INSERT INTO outbox(kind,issue,text,created,expires)
            VALUES (?,?,?,?,?)
            """,
            (kind, issue, text, time.time(), expires),
        )

    def health_stats(self) -> dict[str, Any]:
        settled = int(
            self.c.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE status IN ('WIN','LOSS')
                """
            ).fetchone()[0]
        )
        wins = int(
            self.c.execute(
                "SELECT COUNT(*) FROM predictions WHERE status='WIN'"
            ).fetchone()[0]
        )
        number_wins = int(
            self.c.execute(
                """
                SELECT COUNT(*) FROM predictions
                WHERE status IN ('WIN','LOSS') AND won_number=1
                """
            ).fetchone()[0]
        )
        history_count = int(
            self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        )

        return {
            "history": history_count,
            "settled": settled,
            "side_wins": wins,
            "side_win_rate": round(100.0 * wins / settled, 2) if settled else None,
            "number_wins": number_wins,
            "number_hit_rate": (
                round(100.0 * number_wins / settled, 2) if settled else None
            ),
        }

    def close(self) -> None:
        self.c.close()


# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db: Database):
        self.db = db
        self.engine = Engine(db)
        self.stop_event = asyncio.Event()
        self.last_fetch = 0.0

    async def pause(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(
                self.stop_event.wait(),
                timeout=max(0.05, seconds),
            )
        except asyncio.TimeoutError:
            pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def fetch_one(url: str) -> list[dict[str, Any]]:
            try:
                async with session.get(url) as response:
                    response.raise_for_status()
                    payload = await response.json(content_type=None)
                    return parse_history(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOG.warning("History source failed (%s): %s", url, exc)
                return []

        results = await asyncio.gather(
            *(fetch_one(url) for url in CFG.urls)
        )
        usable_sources = [rows for rows in results if rows]

        if not usable_sources:
            raise RuntimeError("No history source returned usable records")

        merged: dict[str, dict[str, Any]] = {}
        for rows in usable_sources:
            for row in rows:
                issue = row["issue"]
                existing = merged.get(issue)
                if existing and existing["num"] != row["num"]:
                    raise ValueError(f"History source conflict for issue {issue}")
                merged[issue] = row

        return sorted(
            merged.values(),
            key=lambda row: int(row["issue"]),
        )[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0

        while not self.stop_event.is_set():
            try:
                records = await self.fetch(session)
                self.db.ingest(records)
                history = self.db.history()

                if not history:
                    raise RuntimeError("History database is empty")

                self.engine.settle(history)
                self.last_fetch = time.time()
                failures = 0

                if len(history) >= CFG.min_history:
                    latest_issue = int(history[-1]["issue"])
                    next_issue = str(latest_issue + 1)

                    exists = self.db.c.execute(
                        "SELECT 1 FROM predictions WHERE issue=?",
                        (next_issue,),
                    ).fetchone()

                    if not exists:
                        prediction = self.engine.predict(history, next_issue)
                        message = format_prediction(prediction)

                        with self.db.transaction():
                            self.db.c.execute(
                                """
                                INSERT INTO predictions(
                                    issue,created,status,payload,actual,won_number
                                )
                                VALUES (?,?,?, ?,NULL,0)
                                """,
                                (
                                    next_issue,
                                    time.time(),
                                    "PENDING",
                                    json.dumps(prediction, separators=(",", ":")),
                                ),
                            )
                            self.db.queue(
                                "prediction",
                                next_issue,
                                message,
                                time.time() + CFG.delivery_ttl,
                            )

            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                LOG.exception("Worker iteration failed")

            backoff = min(
                60.0,
                CFG.fetch_every * (2 ** min(failures, 4)),
            )
            await self.pause(backoff)

    async def sender(self, session: aiohttp.ClientSession) -> None:
        endpoint_message = (
            f"https://api.telegram.org/bot{CFG.token}/sendMessage"
        )
        endpoint_sticker = (
            f"https://api.telegram.org/bot{CFG.token}/sendSticker"
        )
        backoff = 1.0

        while not self.stop_event.is_set():
            now = time.time()
            self.db.c.execute(
                """
                UPDATE outbox SET sent=2
                WHERE sent=0 AND expires IS NOT NULL AND expires <= ?
                """,
                (now,),
            )

            row = self.db.c.execute(
                """
                SELECT id,kind,text
                FROM outbox
                WHERE sent=0
                ORDER BY id ASC
                LIMIT 1
                """
            ).fetchone()

            if not row:
                await self.pause(0.5)
                continue

            is_sticker = row["kind"] == "sticker"
            url = endpoint_sticker if is_sticker else endpoint_message
            payload = (
                {"chat_id": CFG.chat, "sticker": row["text"]}
                if is_sticker
                else {
                    "chat_id": CFG.chat,
                    "text": row["text"],
                    "parse_mode": "HTML",
                }
            )

            delay = backoff
            try:
                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)

                    if response.status == 200 and data.get("ok"):
                        self.db.c.execute(
                            "UPDATE outbox SET sent=1 WHERE id=?",
                            (row["id"],),
                        )
                        backoff = 1.0
                        continue

                    error_code = data.get("error_code", response.status)
                    if error_code == 429:
                        retry_after = data.get("parameters", {}).get(
                            "retry_after", 10
                        )
                        delay = max(1.0, float(retry_after))
                    elif error_code in (400, 401, 403, 404):
                        LOG.error(
                            "Telegram rejected outbox item (%s): %s",
                            error_code,
                            data.get("description", "no description"),
                        )
                        delay = 60.0
                    else:
                        LOG.warning(
                            "Telegram response %s: %s",
                            response.status,
                            data,
                        )

            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Telegram request failed")
                delay = 10.0

            await self.pause(delay)
            backoff = min(60.0, backoff * 2.0)

    async def health(self, request: web.Request) -> web.Response:
        stats = self.db.health_stats()
        fresh = self.last_fetch > 0 and time.time() - self.last_fetch < 90

        return web.json_response(
            {
                "status": "ok" if fresh else "degraded",
                "last_successful_fetch": self.last_fetch or None,
                **stats,
            }
        )


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    CFG.validate()

    db = Database(CFG.db_path)
    runtime = Runtime(db)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop_event.set)

    app = web.Application()
    app.router.add_get("/", runtime.health)

    runner = web.AppRunner(app)
    tasks: list[asyncio.Task] = []
    stop_task: asyncio.Task | None = None

    timeout = aiohttp.ClientTimeout(total=CFG.timeout)
    # Default aiohttp connector verifies TLS certificates.
    connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300)

    try:
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", CFG.port)
        await site.start()

        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        ) as session:
            tasks = [
                asyncio.create_task(
                    runtime.worker(session),
                    name="history-worker",
                ),
                asyncio.create_task(
                    runtime.sender(session),
                    name="telegram-sender",
                ),
            ]
            stop_task = asyncio.create_task(
                runtime.stop_event.wait(),
                name="shutdown-waiter",
            )

            LOG.info(
                "%s started on port %s; %d history source(s)",
                CFG.name,
                CFG.port,
                len(CFG.urls),
            )

            done, _ = await asyncio.wait(
                [*tasks, stop_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            runtime.stop_event.set()

            for task in done:
                if task is stop_task or task.cancelled():
                    continue
                exc = task.exception()
                if exc:
                    raise exc

            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    finally:
        runtime.stop_event.set()

        for task in tasks:
            task.cancel()
        if stop_task:
            stop_task.cancel()

        await asyncio.gather(
            *tasks,
            *([stop_task] if stop_task else []),
            return_exceptions=True,
        )
        await runner.cleanup()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
