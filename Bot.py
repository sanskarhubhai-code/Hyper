"""
FTW Predictor V48 — APEX INTELLIGENCE (Bayesian + Entropy + Adaptive Meta-Learner)
Python 3.10+
Deps: aiohttp

⚠️ HONEST DISCLOSURE:
This is a statistical model over a random game. "Bayesian", "Meta-Learner",
"Adaptive" — these describe how probabilities are combined, NOT that the
underlying game is beatable. Long-run expected hit rate ≈ 50%. Use only
for entertainment. Output is not a guarantee.
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

DISCLOSURE = "<i>Statistical output — not a guarantee.</i>"


@dataclass(frozen=True)
class Config:
    token: str = os.getenv("TELEGRAM_TOKEN", "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c")
    chat: str = os.getenv("TELEGRAM_CHAT", "-1004485961768")
    urls_raw: str = os.getenv("HISTORY_URLS", DEFAULT_HISTORY_URL)
    db_path: str = os.getenv("DB_PATH", "predictor_v48.4_apex.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 18.0
    max_history: int = 3500
    min_history: int = int(os.getenv("MIN_HISTORY", "180"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    delivery_ttl: float = 600.0
    sticker_ttl: float = 3600.0
    max_outbox_attempts: int = 3
    prediction_retention_days: int = 30
    name: str = "FTW PREDICTOR (V48 APEX)"

    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(x.strip() for x in self.urls_raw.split(",") if x.strip())

    def validate(self) -> None:
        if not self.token:
            raise ValueError("TELEGRAM_TOKEN is required")
        if not self.chat:
            raise ValueError("TELEGRAM_CHAT is required")


CFG = Config()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOG = logging.getLogger("predictor_v48")


# ============================================================
# HELPERS & MATH
# ============================================================

def clip(value: float, low: float = 0.02, high: float = 0.98) -> float:
    return max(low, min(high, float(value)))


def sigmoid(z: float) -> float:
    z = max(-25.0, min(25.0, float(z)))
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float = 0.02) -> float:
    p = max(eps, min(1.0 - eps, float(p)))
    return math.log(p / (1.0 - p))


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
    """Kish effective sample size + Laplace-smoothed weighted mean."""
    if not observations:
        return 0.5, 0.0
    weight_sum = sum(w for _, w in observations)
    squared_weight_sum = sum(w * w for _, w in observations)
    effective_n = (weight_sum * weight_sum / squared_weight_sum) if squared_weight_sum > 0 else 0.0
    probability = (4.0 + sum(v * w for v, w in observations)) / (8.0 + weight_sum)
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


def entropy_bits(seq: list[int]) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    return -sum((f / total) * math.log2(f / total) for f in counts.values())


def cusum_change(
    seq: list[int],
    threshold: float = 3.0,
    reference: float = 0.5,
    allowance: float = 0.1,
    window: int = 40,
) -> int:
    """Return 1 for upward shift, -1 for downward shift, otherwise 0."""
    tail = seq[-window:]
    if len(tail) < 20:
        return 0
    s_pos = 0.0
    s_neg = 0.0
    for value in tail:
        s_pos = max(0.0, s_pos + value - reference - allowance)
        s_neg = max(0.0, s_neg + reference - value - allowance)
    if s_pos > threshold:
        return 1
    if s_neg > threshold:
        return -1
    return 0


def family_run(seq: list[int]) -> list[tuple[int, int]]:
    if not seq:
        return []
    out, cur, count = [], seq[0], 1
    for x in seq[1:]:
        if x == cur:
            count += 1
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

    records = {}
    for item in obj:
        if not isinstance(item, dict):
            continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))
        if issue is None or number is None or not 0 <= number <= 9:
            continue
        issue_key = str(issue)
        records[issue_key] = {"issue": issue_key, "num": number, "size": int(number >= 5)}
    return sorted(records.values(), key=lambda row: int(row["issue"]))


# ============================================================
# SMART PATTERN MINER — ensemble weighted vote
# ============================================================

class SmartPatternMiner:
    """
    Note on naming: `family_signature` == exact RLE of the sequence.
    On a binary alphabet, this is mathematically equivalent to exact match.
    Kept for readability; no information is gained from the "family" abstraction.
    """

    def __init__(self, db: "Database"):
        self.db = db

    def find_best_size_pattern(self, seq: list[int], regime: str) -> tuple[str, str, int, int, float]:
        """
        Returns (side, pattern_name, big_count, small_count, p_big).
        p_big is a reliability-weighted ensemble probability, not a single-pattern stat.
        """
        n = len(seq)
        if n < 15:
            return "BIG", "—", 0, 0, 0.5

        candidates: list[tuple[float, str, str, int, int]] = []

        for length in range(9, 4, -1):
            if n < length * 2:
                continue

            suffix = seq[-length:]
            sig = family_signature(suffix)
            obs: list[tuple[int, float]] = []

            for i in range(n - length):
                if family_signature(seq[i:i + length]) == sig:
                    obs.append((seq[i + length], recency_weight(i + length, n, 420.0)))

            if len(obs) < CFG.min_support:
                continue

            p, eff = weighted_probability(obs)
            big_c = sum(1 for val, _ in obs if val == 1)
            small_c = len(obs) - big_c

            pattern_str = "-".join(f"{'B' if v else 'S'}{c}" for v, c in sig)
            # Regime-scoped pattern key — patterns can behave differently per regime.
            key = f"size_family[{regime}]:{pattern_str}"

            _, rel = self.db.adapt_pattern(key, p)
            if rel < 0.15:
                continue

            side = "BIG" if p >= 0.5 else "SMALL"
            score = rel * math.sqrt(max(1.0, eff))
            candidates.append((score, side, pattern_str, big_c, small_c))

        if not candidates:
            return "BIG", "—", 0, 0, 0.5

        big_score = sum(c[0] for c in candidates if c[1] == "BIG")
        small_score = sum(c[0] for c in candidates if c[1] == "SMALL")
        total = big_score + small_score

        if total <= 0:
            return "BIG", "—", 0, 0, 0.5

        p_big = big_score / total
        final_side = "BIG" if p_big >= 0.5 else "SMALL"

        best = max((c for c in candidates if c[1] == final_side), key=lambda x: x[0])
        _, _, best_name, best_big, best_small = best

        return final_side, best_name, best_big, best_small, clip(p_big)


# ============================================================
# DEEP NUMBER TRACKER — Bayesian N-Gram with recency weighting
# ============================================================

def track_numbers(nums: list[int]) -> tuple[list[float], str]:
    if not nums:
        return [0.1] * 10, "No Data"
    n = len(nums)

    base = [1.0] * 10

    for window, weight in ((30, 1.0), (100, 0.5)):
        recent = nums[-window:]
        for idx, k in enumerate(recent):
            pos = len(recent) - idx
            rw = 0.5 ** (pos / 30.0)
            base[k] += rw * weight

    for order in (2, 3, 4):
        if n <= order:
            continue
        ctx = tuple(nums[-order:])
        obs = [0.0] * 10
        total_w = 0.0
        for i in range(order, n):
            if tuple(nums[i - order:i]) == ctx:
                age = n - i
                rw = 0.5 ** (age / 200.0)
                obs[nums[i]] += rw
                total_w += rw
        if total_w >= 0.5:
            for k in range(10):
                base[k] += (obs[k] / total_w) * 5.0

    total = sum(base)
    if total <= 0:
        return [0.1] * 10, "No Data"
    dist = [x / total for x in base]
    return dist, "Bayesian N-Gram (Recency)"


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
            CREATE TABLE IF NOT EXISTS predictions(
                issue TEXT PRIMARY KEY, created REAL NOT NULL, status TEXT NOT NULL,
                payload TEXT NOT NULL, actual INTEGER,
                won_number INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS pattern_models(
                name TEXT PRIMARY KEY, n REAL NOT NULL, loss REAL NOT NULL,
                failures INTEGER NOT NULL DEFAULT 0,
                wins INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
                issue TEXT, text TEXT NOT NULL, created REAL NOT NULL,
                expires REAL, sent INTEGER NOT NULL DEFAULT 0,
                attempts INTEGER NOT NULL DEFAULT 0
            );
        """)
        self._migrate()

    def _migrate(self) -> None:
        """Add columns missing in older DBs. Safe to run repeatedly."""
        def cols(table: str) -> set[str]:
            return {r["name"] for r in self.c.execute(f"PRAGMA table_info({table})")}

        pm = cols("pattern_models")
        for name, spec in (("wins", "INTEGER NOT NULL DEFAULT 0"),
                           ("total", "INTEGER NOT NULL DEFAULT 0")):
            if name not in pm:
                self.c.execute(f"ALTER TABLE pattern_models ADD COLUMN {name} {spec}")

        ob = cols("outbox")
        if "attempts" not in ob:
            self.c.execute("ALTER TABLE outbox ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")

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

    def get(self, key: str, default: Any = None) -> Any:
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )

    def ingest(self, records: list[dict[str, Any]]) -> None:
        with self.transaction():
            for r in records:
                self.c.execute("INSERT OR IGNORE INTO history(issue,num) VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute(
                "DELETE FROM history WHERE issue NOT IN "
                "(SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)",
                (CFG.max_history,),
            )

    def history(self) -> list[dict[str, Any]]:
        rows = self.c.execute(
            "SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER) ASC"
        ).fetchall()
        return [{"issue": r["issue"], "num": int(r["num"]), "size": int(int(r["num"]) >= 5)} for r in rows]

    def pending_predictions(self) -> list[sqlite3.Row]:
        return self.c.execute(
            "SELECT issue,payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER) ASC"
        ).fetchall()

    def learn_pattern(self, key: str, probability: float, actual: int) -> None:
        """Update pattern stats with sliding wins/total window."""
        row = self.c.execute(
            "SELECT n, loss, wins, total FROM pattern_models WHERE name=?", (key,)
        ).fetchone()
        if row:
            old_n = float(row["n"] or 0.0)
            old_loss = float(row["loss"] or 0.0)
            old_wins = int(row["wins"] or 0)
            old_total = int(row["total"] or 0)
        else:
            old_n, old_loss, old_wins, old_total = 0.0, 0.0, 0, 0

        n = 0.985 * old_n + 1.0
        loss = 0.985 * old_loss + (clip(probability) - actual) ** 2
        correct = int((probability >= 0.5) == (actual >= 0.5))
        wins = old_wins + correct
        total = old_total + 1

        # Cap window to prevent unbounded growth (halve when full).
        if total > 300:
            wins //= 2
            total //= 2

        self.c.execute(
            "INSERT INTO pattern_models(name,n,loss,failures,wins,total) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "n=excluded.n, loss=excluded.loss, wins=excluded.wins, total=excluded.total",
            (key, n, loss, 0, wins, total),
        )

    def adapt_pattern(self, key: str, probability: float) -> tuple[float, float]:
        """
        Return (probability, reliability). Reliability scales how much to trust this
        pattern's directional signal, based on its Beta-posterior win rate.
        Probability itself is not shifted — that would be double-counting with p_big.
        """
        row = self.c.execute(
            "SELECT wins, total FROM pattern_models WHERE name=?", (key,)
        ).fetchone()
        if not row:
            return clip(probability), 1.0

        total = int(row["total"] or 0)
        wins = int(row["wins"] or 0)
        if total < 3:
            return clip(probability), 1.0

        # Beta posterior with weak prior centered at 0.5 (α0 = β0 = 2).
        alpha = wins + 2
        beta = (total - wins) + 2
        win_rate = alpha / (alpha + beta)

        evidence = total / (total + 15.0)       # 0..1, saturates ~30-50 samples
        skill = max(0.0, (win_rate - 0.5) * 2.0)  # 0 at chance, 1 at 100%
        reliability = 0.2 + 0.8 * skill * evidence

        return clip(probability), max(0.05, min(2.0, reliability))

    def queue(self, kind: str, issue: str | None, text: str, expires: float | None = None) -> None:
        self.c.execute(
            "INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)",
            (kind, issue, text, time.time(), expires),
        )

    def prune(self) -> None:
        """Housekeeping — remove stale settled rows and sent outbox entries."""
        cutoff = time.time() - CFG.prediction_retention_days * 86400
        with self.transaction():
            self.c.execute(
                "DELETE FROM predictions WHERE status != 'PENDING' AND created < ?",
                (cutoff,),
            )
            self.c.execute("DELETE FROM outbox WHERE sent = 1 AND created < ?", (cutoff,))
            self.c.execute("DELETE FROM outbox WHERE sent = 2 AND created < ?", (cutoff,))

    def health_stats(self) -> dict[str, Any]:
        settled = int(
            self.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        )
        wins = int(self.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0])
        number_hits = int(
            self.c.execute("SELECT COUNT(*) FROM predictions WHERE won_number=1").fetchone()[0]
        )
        outbox_pending = int(
            self.c.execute("SELECT COUNT(*) FROM outbox WHERE sent=0").fetchone()[0]
        )
        return {
            "history": int(self.c.execute("SELECT COUNT(*) FROM history").fetchone()[0]),
            "settled": settled,
            "side_wins": wins,
            "side_win_rate": round(100.0 * wins / settled, 2) if settled else None,
            "number_hits": number_hits,
            "outbox_pending": outbox_pending,
        }

    def close(self) -> None:
        self.c.close()


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
        if not seq:
            raise ValueError("Cannot predict without history")

        regime = get_regime(seq)
        entropy = entropy_bits(seq[-24:])

        _, pat_name, pat_big, pat_small, pat_p = self.miner.find_best_size_pattern(seq, regime)
        num_dist, num_info = track_numbers(nums)

        if pat_name != "—":
            p_big = pat_p
            reason = "Pattern Ensemble"
        else:
            recent = seq[-20:]
            p_big = (sum(recent) + 2.0) / (len(recent) + 4.0)
            reason = "Recent Trend"

        if entropy > 0.92:
            p_big = 0.5 + (p_big - 0.5) * 0.5
            reason += " · High Entropy"

        cusum_dir = cusum_change(seq)
        if cusum_dir:
            recent = seq[-20:]
            recent_rate = (sum(recent) + 2.0) / (len(recent) + 4.0)
            p_big = 0.85 * p_big + 0.15 * recent_rate
            reason += " · CUSUM"

        p_big = clip(p_big)
        side = "BIG" if p_big >= 0.5 else "SMALL"

        allowed = list(range(5, 10)) if side == "BIG" else list(range(0, 5))

        num_dist_final = [
            num_dist[k] * (1.5 if k in allowed else 0.1) for k in range(10)
        ]
        total = sum(num_dist_final)
        if total > 0:
            num_dist_final = [x / total for x in num_dist_final]

        recent_counts = Counter(nums[-50:])
        num_pred = max(allowed, key=lambda k: (num_dist_final[k], recent_counts.get(k, 0)))
        num_conf = num_dist_final[num_pred]

        return {
            "issue": issue,
            "p": p_big,
            "side": side,
            "number": num_pred,
            "num_conf": num_conf,
            "regime": regime,
            "pattern": pat_name,
            "pat_big": pat_big,
            "pat_small": pat_small,
            "num_tracking": num_info,
            "override_reason": reason,
            "source": "V48 Apex",
            "created": time.time(),
        }

    def settle(self, history: list[dict[str, Any]]) -> None:
        lookup = {r["issue"]: r for r in history}
        for row in self.db.pending_predictions():
            actual = lookup.get(row["issue"])
            if not actual:
                continue

            payload = json.loads(row["payload"])
            actual_size = int(actual["size"])
            actual_num = int(actual["num"])

            predicted_side = payload.get("side", "BIG")
            actual_side = "BIG" if actual_size == 1 else "SMALL"
            won_side = predicted_side == actual_side
            won_num = actual_num == payload.get("number")

            with self.db.transaction():
                self.db.c.execute(
                    "UPDATE predictions SET status=?, actual=?, won_number=? "
                    "WHERE issue=? AND status='PENDING'",
                    ("WIN" if won_side else "LOSS", actual_num, int(won_num), row["issue"]),
                )

                pat_name = payload.get("pattern")
                regime = payload.get("regime")
                if pat_name and pat_name != "—" and regime:
                    self.db.learn_pattern(
                        f"size_family[{regime}]:{pat_name}",
                        float(payload["p"]),
                        actual_size,
                    )

                sticker = (
                    CFG.jack_sticker if (won_side and won_num)
                    else CFG.win_sticker if won_side
                    else CFG.loss_sticker
                )
                self.db.queue(
                    "sticker", row["issue"], sticker,
                    expires=time.time() + CFG.sticker_ttl,
                )


def format_prediction(payload: dict[str, Any]) -> str:
    prob = float(payload["p"])
    conf = max(prob, 1.0 - prob) * 100
    filled = int(conf // 10)
    bar = "█" * filled + "░" * (10 - filled)
    side = payload["side"]
    marker = "🔴" if side == "BIG" else "🔵"

    pat_line = ""
    if payload.get("pattern", "—") != "—":
        pat_line = f"\n🧩 <b>SIZE PATTERN:</b> <code>{escape(payload['pattern'])}</code>"
        pat_line += f"\n📊 <b>PATTERN STATS:</b> BIG[{payload.get('pat_big', 0)}] | SMALL[{payload.get('pat_small', 0)}]"

    n1 = int(payload["number"])
    # Express number confidence as multiple of uniform baseline (10% = 1.0×).
    nconf = payload.get("num_conf", 0.1) * 100
    multiplier = payload.get("num_conf", 0.1) / 0.1
    num_line = (
        f"\n🔢 <b>TOP NUMBER:</b> <code>{n1}</code> "
        f"({multiplier:.2f}× uniform, {nconf:.1f}%)"
        f"\n📈 <b>NUM TRACKER:</b> <code>{escape(payload.get('num_tracking', '—'))}</code>"
    )

    override = payload.get("override_reason", "")
    override_line = f"\n⚠️ <b>OVERRIDE:</b> <i>{escape(override)}</i>" if override else ""

    return (
        f"⚡ <b>{escape(CFG.name)}</b> ⚡\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>SIDE:</b> {marker} <b>{side}</b>\n"
        f"📅 <b>ISSUE:</b> <code>{escape(payload['issue'])}</code>\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🌊 <b>REGIME:</b> {escape(payload['regime'])}"
        f"{pat_line}{num_line}{override_line}\n"
        "━━━━━━━━━━━━━━━━━━\n"
        f"{DISCLOSURE}"
    )


# ============================================================
# RUNTIME
# ============================================================

class Runtime:
    def __init__(self, db: Database):
        self.db = db
        self.engine = Engine(db)
        self.stop = asyncio.Event()
        self.last_fetch = 0.0
        self.sources_ok = 0
        self.sources_total = 0
        self.last_source_error: str | None = None

    async def pause(self, sec: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=max(0.05, sec))
        except asyncio.TimeoutError:
            pass

    async def fetch(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        async def one(url: str) -> list[dict[str, Any]] | None:
            try:
                async with session.get(url) as r:
                    r.raise_for_status()
                    return parse_history(await r.json(content_type=None))
            except Exception as e:
                LOG.warning("Source fetch failed: %s — %s", url, e)
                return None

        results = await asyncio.gather(*(one(u) for u in CFG.urls))
        sources = [r for r in results if r]
        self.sources_total = len(CFG.urls)
        self.sources_ok = len(sources)

        if not sources:
            self.last_source_error = "all_sources_failed"
            raise RuntimeError("HistoryUnavailable")

        # Prefer source with highest max-issue; fill gaps from lower-priority sources.
        def max_issue(src: list[dict[str, Any]]) -> int:
            return max((int(r["issue"]) for r in src), default=0)

        sources.sort(key=max_issue, reverse=True)

        # Warn on high-issue disagreement between sources (data integrity signal).
        if len(sources) > 1:
            top = sources[0]
            top_map = {r["issue"]: r["num"] for r in top[-20:]}
            for other in sources[1:]:
                other_map = {r["issue"]: r["num"] for r in other[-20:]}
                for issue, num in top_map.items():
                    if issue in other_map and other_map[issue] != num:
                        LOG.warning(
                            "Source disagreement issue=%s: %s vs %s",
                            issue, num, other_map[issue],
                        )

        merged: dict[str, dict[str, Any]] = {}
        for src in sources:  # newest first — older sources cannot override
            for row in src:
                merged.setdefault(row["issue"], row)

        self.last_source_error = None
        return sorted(merged.values(), key=lambda x: int(x["issue"]))[-CFG.max_history:]

    async def worker(self, session: aiohttp.ClientSession) -> None:
        failures = 0
        last_prune = 0.0
        while not self.stop.is_set():
            try:
                self.db.ingest(await self.fetch(session))
                history = self.db.history()
                self.engine.settle(history)
                self.last_fetch = time.time()

                if time.time() - last_prune > 3600:
                    self.db.prune()
                    last_prune = time.time()

                if len(history) >= CFG.min_history:
                    issue = str(int(history[-1]["issue"]) + 1)
                    exists = self.db.c.execute(
                        "SELECT 1 FROM predictions WHERE issue=?", (issue,)
                    ).fetchone()
                    if not exists:
                        pred = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute(
                                "INSERT INTO predictions(issue,created,status,payload) "
                                "VALUES (?,?,?,?)",
                                (issue, time.time(), "PENDING", json.dumps(pred)),
                            )
                            self.db.queue(
                                "prediction", issue, format_prediction(pred),
                                time.time() + CFG.delivery_ttl,
                            )
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                LOG.exception(
                    "Worker error (failures=%d sources=%d/%d)",
                    failures, self.sources_ok, self.sources_total,
                )
            await self.pause(min(60, CFG.fetch_every * (2 ** min(failures, 4))))

    async def sender(self, session: aiohttp.ClientSession) -> None:
        backoff = 1.0
        while not self.stop.is_set():
            # Sweep expired unsent rows before doing anything else.
            self.db.c.execute(
                "UPDATE outbox SET sent=2 WHERE sent=0 AND expires IS NOT NULL AND expires <= ?",
                (time.time(),),
            )

            row = self.db.c.execute(
                "SELECT * FROM outbox WHERE sent=0 ORDER BY id ASC LIMIT 1"
            ).fetchone()
            if not row:
                await self.pause(0.5)
                continue

            try:
                url = (
                    f"https://api.telegram.org/bot{CFG.token}/"
                    + ("sendSticker" if row["kind"] == "sticker" else "sendMessage")
                )
                body = (
                    {"chat_id": CFG.chat, "sticker": row["text"]}
                    if row["kind"] == "sticker"
                    else {"chat_id": CFG.chat, "text": row["text"], "parse_mode": "HTML"}
                )

                async with session.post(url, json=body) as res:
                    data = await res.json(content_type=None)
                    if res.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0
                        continue

                    code = data.get("error_code")

                    # 401 = invalid token. Whole bot is dead; stop cleanly.
                    if code == 401:
                        LOG.critical("Telegram 401 — token invalid. Stopping sender.")
                        self.stop.set()
                        return

                    # 429 = rate-limited. Respect retry_after.
                    if code == 429:
                        retry = float(data.get("parameters", {}).get("retry_after", 10))
                        await self.pause(retry)
                        continue

                    # 400/403/404 = likely permanent (bad HTML, forbidden, deleted).
                    if code in (400, 403, 404):
                        attempts = int(row["attempts"] or 0) + 1
                        if attempts >= CFG.max_outbox_attempts:
                            LOG.error(
                                "Dropping outbox id=%s after %d attempts: %s",
                                row["id"], attempts, data.get("description"),
                            )
                            self.db.c.execute(
                                "UPDATE outbox SET sent=2, attempts=? WHERE id=?",
                                (attempts, row["id"]),
                            )
                        else:
                            self.db.c.execute(
                                "UPDATE outbox SET attempts=? WHERE id=?",
                                (attempts, row["id"]),
                            )
                            await self.pause(5.0)
                        continue

                    LOG.warning("Telegram error %s: %s", code, data.get("description"))
                    await self.pause(backoff)
                    backoff = min(60.0, backoff * 2)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("Sender error")
                await self.pause(backoff)
                backoff = min(60.0, backoff * 2)

    async def health(self, request: web.Request) -> web.Response:
        now = time.time()
        data = {
            "status": "ok" if self.last_fetch and now - self.last_fetch < 90 else "degraded",
            "last_fetch_age": round(now - self.last_fetch, 1) if self.last_fetch else None,
            "sources_ok": self.sources_ok,
            "sources_total": self.sources_total,
            "last_source_error": self.last_source_error,
            **self.db.health_stats(),
        }
        return web.json_response(data)


async def main() -> None:
    CFG.validate()
    db = Database(CFG.db_path)
    runtime = Runtime(db)

    app = web.Application()
    app.router.add_get("/", runtime.health)
    runner = web.AppRunner(app)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, runtime.stop.set)

    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()

        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [
                asyncio.create_task(runtime.worker(session)),
                asyncio.create_task(runtime.sender(session)),
            ]
            await asyncio.gather(*tasks)
    finally:
        await runner.cleanup()
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
