# bot.py — FTW VIP PREDICTOR (HYPER V22 — SERIOUS REGIME-ADAPTIVE ENGINE)
# Python 3.10+ | Deps: aiohttp
#
# Stronger than V21 because:
# - separate weights per regime
# - true online learning of ensemble coefficients
# - richer state features
# - dynamic behavior mixing
# - better 10-digit number model
# - no skip logic
#
# NOTE:
# This is still statistical only. If the underlying source is pseudo-random,
# no code can create guaranteed edge. But this design is much more realistic.

import asyncio, contextlib, html, json, logging, math, os, random, signal, socket, sqlite3, time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass

import aiohttp
from aiohttp import web

# ─── CONFIG ─────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    token: str = "8611789455:AAGjXkDekFYjUGgWRat_8OySdsRZANjdc9c"
    chat: str = "-1004485961768"
    urls: tuple = ("https://sky-predictor-1012593186417.asia-southeast1.run.app/api/wingo-history-1m-1000",)
    db_path: str = os.getenv("DB_PATH", "predictor_hyper_v22.db")
    port: int = int(os.getenv("PORT", "8080"))
    fetch_every: float = float(os.getenv("FETCH_EVERY", "3"))
    timeout: float = 15.0
    max_history: int = 4000
    min_history: int = int(os.getenv("MIN_HISTORY", "120"))
    min_support: int = int(os.getenv("MIN_PATTERN_SUPPORT", "8"))
    learning_rate: float = 0.07
    meta_lr: float = 0.06
    decay: float = 0.985
    max_pattern: int = 30
    delivery_ttl: float = 60.0
    name: str = "FTW VIP PREDICTOR (HYPER V22)"
    win_sticker: str = "CAACAgUAAxkBAAFW0Ixq4QlioiYyx39xRe3akLzGvqXoSgACjg8AAg_PgFc259YBd5tjBj0E"
    loss_sticker: str = "CAACAgUAAxkBAAFW0JZq4Qr4QPStqiQjySoP1DQMvsFZawACVx0AAptrIVX6uzts6bbocD0E"
    jack_sticker: str = "CAACAgUAAxkBAAFW0Ipq4QlLbGgRu5R3TIIiSY_9z1uNuwACSxQAAu1zgVchvQvX59KsHD0E"

    def validate(self):
        if not self.token or not self.chat or not self.urls:
            raise ValueError("Missing required config: token, chat, or urls")

CFG = Config()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("predictor")

# ─── UTILITIES ──────────────────────────────────────────────
def clip(x, low=0.02, high=0.98): return max(low, min(high, float(x)))
def sigmoid(z): z = max(-20.0, min(20.0, float(z))); return 1.0 / (1.0 + math.exp(-z))
def logit(p, eps=0.02): p = max(eps, min(1-eps, float(p))); return math.log(p/(1-p))
def escape(x): return html.escape(str(x))

def mean(xs): return sum(xs)/len(xs) if xs else 0.5
def stddev(xs): 
    if len(xs) < 2: return 0.0
    m = mean(xs)
    return math.sqrt(sum((x-m)**2 for x in xs)/len(xs))

def run_length(seq):
    if not seq: return 0
    n = 0
    for x in reversed(seq):
        if x != seq[-1]: break
        n += 1
    return n

def get_regime(seq):
    if not seq: return "COLD"
    tail = seq[-24:]
    if len(tail) < 8: return "COLD"
    flips = sum(a != b for a,b in zip(tail, tail[1:]))
    rate = flips / max(1, len(tail)-1)
    if run_length(tail) >= 4: return "RUN"
    if rate >= 0.72: return "ALT"
    if abs(sum(tail)/len(tail)-0.5) > 0.20: return "IMBAL"
    return "MIXED"

def numeric(value):
    if isinstance(value, bool): return None
    s = str(value).strip()
    return int(s) if s.isascii() and s.isdigit() else None

def parse_history(payload):
    obj = payload
    for _ in range(6):
        if isinstance(obj, list): break
        if not isinstance(obj, dict): return []
        obj = next((obj[k] for k in ("data","list","result","items","history") if isinstance(obj.get(k),(list,dict))), None)
    if not isinstance(obj, list): return []
    recs = {}
    for item in obj:
        if not isinstance(item, dict): continue
        issue = numeric(item.get("issueNumber", item.get("issue")))
        number = numeric(item.get("number", item.get("value")))
        if issue is None or number is None or not 0 <= number <= 9: continue
        key = str(issue)
        if key in recs and recs[key]["num"] != number: raise ValueError(f"Conflicting duplicate: {key}")
        recs[key] = {"issue": key, "num": number, "size": int(number >= 5)}
    return sorted(recs.values(), key=lambda x: int(x["issue"]))

def weighted_probability(obs):
    sw = sum(w for _,w in obs)
    sw2 = sum(w*w for _,w in obs)
    effective_n = sw*sw/sw2 if sw2 else 0.0
    probability = (4 + sum(y*w for y,w in obs)) / (8 + sw)
    return probability, effective_n

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
            self.c.execute("ROLLBACK"); raise

    def get(self, key, default=None):
        row = self.c.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key, value):
        self.c.execute("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, json.dumps(value)))

    def ingest(self, records):
        with self.transaction():
            for r in records:
                old = self.c.execute("SELECT num FROM history WHERE issue=?", (r["issue"],)).fetchone()
                if old and old["num"] != r["num"]:
                    raise ValueError(f"Source changed settled issue {r['issue']}")
                self.c.execute("INSERT OR IGNORE INTO history VALUES (?,?)", (r["issue"], r["num"]))
            self.c.execute("DELETE FROM history WHERE issue NOT IN (SELECT issue FROM history ORDER BY CAST(issue AS INTEGER) DESC LIMIT ?)", (CFG.max_history,))

    def history(self):
        rows = self.c.execute("SELECT issue,num FROM history ORDER BY CAST(issue AS INTEGER)").fetchall()
        return [{"issue": r["issue"], "num": r["num"], "size": int(r["num"] >= 5)} for r in rows]

    def memory(self, key):
        row = self.c.execute("SELECT * FROM models WHERE key=?", (key,)).fetchone()
        return dict(row) if row else {"n":0.0,"loss":0.0,"inverse_loss":0.0,"failures":0}

    def learn(self, key, probability, actual):
        m = self.memory(key)
        n = CFG.decay * m["n"] + 1
        loss = CFG.decay * m["loss"] + (probability - actual) ** 2
        inv = CFG.decay * m["inverse_loss"] + (1 - probability - actual) ** 2
        wrong = int(probability >= 0.5) != actual
        failures = m["failures"] + 1 if wrong else 0
        self.c.execute(
            "INSERT INTO models VALUES (?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
            "n=excluded.n,loss=excluded.loss,inverse_loss=excluded.inverse_loss,failures=excluded.failures",
            (key, n, loss, inv, failures),
        )

    def adapt(self, key, probability):
        m = self.memory(key)
        n = m["n"]
        if n < 1:
            return probability, 0.75, False
        brier = m["loss"] / n
        inv = m["inverse_loss"] / n
        flipped = (inv + 0.025 < brier) and (brier > 0.24) and (m["failures"] >= 3)
        effective_loss = inv if flipped else brier
        p = (1-probability) if flipped else probability
        reliability = clip((0.25 - effective_loss) / 0.12, 0.0, 1.0)
        maturity = min(n / 80.0, 1.0)
        shrink = (1 - maturity) * 0.7 + maturity * reliability
        p = 0.5 + (p - 0.5) * shrink
        weight = clip(math.exp(-6*(effective_loss - 0.22)), 0.18, 2.2)
        if m["failures"] >= 4 and not flipped:
            weight *= 0.55
        return clip(p), weight, flipped

    def queue(self, kind, issue, text, expires=None):
        self.c.execute("INSERT INTO outbox(kind,issue,text,created,expires) VALUES (?,?,?,?,?)", (kind, issue, text, time.time(), expires))

# ─── REGIME-AWARE FEATURE MATH ─────────────────────────────────
def ema_series(seq, period):
    if not seq: return 0.5
    a = 2.0/(period+1)
    v = float(seq[0])
    for x in seq[1:]:
        v = a*x + (1-a)*v
    return v

def stable_entropy(p):
    p = clip(p)
    if p in (0.0, 1.0): return 0.0
    return -(p*math.log2(p) + (1-p)*math.log2(1-p))

def build_state_features(seq):
    # returns rich state features for regime-specific prediction
    if not seq:
        return [0.0]*24

    S = seq
    n = len(S)
    def tail_mean(k):
        arr = S[-k:]; return mean(arr) if arr else 0.5

    # basic stats
    m1 = tail_mean(3)
    m2 = tail_mean(5)
    m3 = tail_mean(10)
    m4 = tail_mean(24)

    # volatility / oscillation
    alt = sum(a != b for a,b in zip(S[-12:], S[-11:])) / max(1, len(S[-12:])-1)
    run_now = run_length(S)
    recent_flip = sum(a != b for a,b in zip(S[-8:], S[-7:])) # quick proxy

    # transitions
    last = S[-1]
    prev = S[-2] if len(S)>=2 else last
    prev2 = S[-3] if len(S)>=3 else prev
    multi_state = 1.0 if S[-1]==1 and S[-2]==1 else 0.0
    triple_state = 1.0 if S[-1]==S[-2]==S[-3] else 0.0

    # directional trend
    trend = (tail_mean(5)-tail_mean(12))
    trend2 = (tail_mean(10)-tail_mean(24))

    # regime
    regime = get_regime(S)
    regime_map = {"COLD":0.0,"MIXED":0.25,"ALT":0.5,"IMBAL":0.75,"RUN":1.0}
    regime_value = regime_map.get(regime, 0.5)

    features = [
        1.0,
        2*m1 - 1,
        2*m2 - 1,
        2*m3 - 1,
        2*m4 - 1,
        2*tail_mean(48)-1,
        (S[-1] - 0.5)*2.0,
        (tail_mean(5)-0.5)*2.0,
        (tail_mean(10)-0.5)*2.0,
        (tail_mean(24)-0.5)*2.0,
        trend,
        trend2,
        run_now/10.0,
        alt*2.0 - 1.0,
        recent_flip/8.0,
        stable_entropy(mean(S[-12:])),
        stable_entropy(mean(S[-24:])),
        (1.0 if last == 1 else 0.0) - 0.5,
        (1.0 if prev == 1 else 0.0) - 0.5,
        (1.0 if prev2 == 1 else 0.0) - 0.5,
        multi_state*2.0 - 1.0,
        triple_state*2.0 - 1.0,
        regime_value,
        (S[-1] == S[-2]) * 1.0
    ]
    return features

# ─── SERIALIZED EXPERTS ───────────────────────────────────────
def logistic_linear(weights, feats):
    return sigmoid(sum(w*x for w,x in zip(weights, feats)))

# ─── MAIN ENGINE ───────────────────────────────────────────────
class Engine:
    def __init__(self, db):
        self.db = db

    def get_runs(self, seq):
        if not seq: return []
        runs = []; curr=seq[0]; length=1
        for x in seq[1:]:
            if x == curr: length +=1
            else: runs.append((curr, length)); curr=x; length=1
        runs.append((curr, length))
        return runs

    def mine_family_patterns(self, seq):
        if not seq: return []
        runs = self.get_runs(seq)
        if len(runs) < 2: return []
        for k in range(min(8, len(runs)), 1, -1):
            target = runs[-k:]
            obs = []
            for i in range(len(runs)-k-1, -1, -1):
                if runs[i:i+k] == target and i+k < len(runs):
                    obs.append((runs[i+k][0], 1.0))
            if obs:
                p, eff_n = weighted_probability(obs)
                bits = "-".join(f"{'B' if v else 'S'}{c}" for v,c in target)
                key = f"family:{bits}"
                rank = abs(p-0.5)*math.sqrt(eff_n) + 12.0
                return [(rank, p, len(obs), eff_n, key)]
        return []

    def mine_patterns(self, seq):
        n = len(seq)
        fam = self.mine_family_patterns(seq)
        if fam: return fam
        candidates = {}
        def weight(idx): return 0.5 ** ((n-1-idx)/300.0)
        for length in range(5, min(CFG.max_pattern, max(5, n//2)) + 1):
            if n < length * 2: continue
            suffix = tuple(seq[-length:])
            obs, last = [], -length - 1
            for i in range(length, n-length):
                if tuple(seq[i-length:i]) == suffix and i-last >= length+1:
                    obs.append((seq[i], weight(i))); last = i
            if len(obs) >= CFG.min_support:
                p, eff_n = weighted_probability(obs)
                if eff_n >= CFG.min_support * 0.5:
                    bits = "".join("B" if x else "S" for x in suffix)
                    candidates[f"exact:{bits}"] = (abs(p-0.5)*math.sqrt(eff_n), p, len(obs), eff_n, f"exact:{bits}")
        return sorted(candidates.values(), key=lambda x: x[0], reverse=True)[:12]

    def _per_regime_weights(self, regime):
        # store a keepalive set of weights per regime, learned online
        key = f"regime_model:{regime}"
        w = self.db.get(key)
        if not isinstance(w, list) or len(w) != 24:
            return [0.0]*24
        return w

    def _save_regime_weights(self, regime, weights):
        self.db.set(f"regime_model:{regime}", weights)

    def _decision_features(self, seq):
        return build_state_features(seq)

    def evaluate_experts(self, seq):
        n = len(seq)
        if n == 0:
            return {}, [], [], "COLD"

        experts = {}
        def weight(idx):
            return 0.5 ** ((n-1-idx)/350.0)

        # patterns
        patterns = self.mine_patterns(seq)
        if patterns:
            num = den = 0.0
            for _, p, _, eff_n, key in patterns:
                correct, w, _ = self.db.adapt(f"pattern:{key}", p)
                num += correct * (w * math.sqrt(max(1.0, eff_n)))
                den += (w * math.sqrt(max(1.0, eff_n)))
            if den > 0:
                experts["patterns"] = num / den

        # markov
        for order in (1,2,3,4):
            suffix = tuple(seq[-order:])
            obs = [(seq[i], weight(i)) for i in range(order, n-order) if tuple(seq[i-order:i]) == suffix]
            if len(obs) >= CFG.min_support:
                experts[f"markov{order}"] = weighted_probability(obs)[0]

        # run
        runat = [1]*n
        for i in range(1,n):
            runat[i] = runat[i-1]+1 if seq[i] == seq[i-1] else 1
        current_run = min(runat[-1], 8)
        last_val = seq[-1]
        run_obs = [(seq[i], weight(i)) for i in range(24, n) if seq[i-1] == last_val and min(runat[i-1],8) == current_run]
        if len(run_obs) >= CFG.min_support:
            experts["run"] = weighted_probability(run_obs)[0]

        # regime
        regime = get_regime(seq)
        regime_obs = [(seq[i], weight(i)) for i in range(24, n) if get_regime(seq[max(0, i-96):i]) == regime]
        if len(regime_obs) >= CFG.min_support:
            experts["regime"] = weighted_probability(regime_obs)[0]

        # base
        experts["base"] = weighted_probability([(y, weight(i)) for i,y in enumerate(seq)])[0]

        # neighbors
        neigh = []
        for i in range(10, n-10):
            diff = sum(a!=b for a,b in zip(seq[i-10:i], seq[-10:]))
            if diff <= 2:
                neigh.append((seq[i], weight(i) * (0.4 ** diff)))
        if len(neigh) >= 20:
            experts["neighbors"] = weighted_probability(neigh)[0]

        # trend features made explicit
        feat = self._decision_features(seq)
        regime_model = self._per_regime_weights(regime)
        experts["trend_model"] = logistic_linear(regime_model, feat)

        # momentum / state / cyclic / jump
        momentum = mean(seq[-5:]) if seq else 0.5
        experts["momentum"] = 0.2 + momentum*0.6
        if len(seq) >= 12:
            cyc = 0.5
            for lag in (3,5,8,12):
                if len(seq) >= lag + 1:
                    cyc += (seq[-1] - seq[-lag - 1]) / (lag + 1)
            experts["cyclic"] = clip(0.5 + 0.2*cyc, 0.02, 0.98)
        if len(seq) >= 4:
            recent = seq[-4:]
            jump = abs(recent[-1]-recent[0]); experts["jump"] = clip(0.5 + (jump-0.5)*0.25, 0.02, 0.98)
        if len(seq) >= 10:
            state_val = 0.5
            if run_length(seq) >= 4 and seq[-1] == 1: state_val = 0.75
            elif run_length(seq) >= 4 and seq[-1] == 0: state_val = 0.25
            experts["state"] = state_val

        return experts, patterns, feat, regime

    def settle(self, history):
        lookup = {r["issue"]: r for r in history}
        if history:
            sig = f"{history[0]['issue']}-{history[-1]['issue']}-{len(history)}"
            if self.db.get("ngram_sig") != sig:
                num_ngrams, num_freq = {}, {}
                for i in range(1, len(history)):
                    curr = history[i]["num"]
                    num_freq[str(curr)] = num_freq.get(str(curr), 0) + 1
                    for length in range(2, 11):
                        if i >= length:
                            seq_key = ",".join(str(history[j]["num"]) for j in range(i-length, i))
                            num_ngrams.setdefault(seq_key, {})
                            num_ngrams[seq_key][str(curr)] = num_ngrams[seq_key].get(str(curr), 0) + 1
                self.db.set("num_ngrams", num_ngrams)
                self.db.set("num_freq", num_freq)
                self.db.set("ngram_sig", sig)

        pending = self.db.c.execute("SELECT issue,payload FROM predictions WHERE status='PENDING' ORDER BY CAST(issue AS INTEGER)").fetchall()
        for row in pending:
            actual = lookup.get(row["issue"])
            if actual is None: continue
            p = json.loads(row["payload"]); y = actual["size"]
            won_side = int(p["p"] >= 0.5) == y
            won_num = p.get("num") == actual["num"]

            with self.db.transaction():
                self.db.c.execute("UPDATE predictions SET status=?,actual=? WHERE issue=?", ("WIN" if won_side else "LOSS", actual["num"], row["issue"]))
                for name, val in p["experts"].items():
                    self.db.learn(f"expert:{name}:{p['regime']}", val, y)
                for pat in p["patterns"]:
                    self.db.learn(pat["key"], pat["p"], y)

                # online update of regime weights
                if p.get("features") and len(p["features"]) == 24:
                    feat = p["features"]
                    regime = p["regime"]
                    w = self._per_regime_weights(regime)
                    # online gradient on p_raw target
                    target = y
                    pred = logistic_linear(w, feat)
                    err = target - pred
                    new_w = [clip(wi + CFG.learning_rate * err * fi, -5.0, 5.0) for wi, fi in zip(w, feat)]
                    self._save_regime_weights(regime, new_w)

                # meta update
                if p.get("meta_feats"):
                    target_meta = len(EXPERT_NAMES) + 2
                    mw = self.db.get("meta_weights")
                    if not isinstance(mw, list) or len(mw) != target_meta:
                        if isinstance(mw, list) and len(mw) == target_meta - 1:
                            mw = mw[:-1] + [0.0, mw[-1]]
                        else:
                            mw = [1.0/len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [0.0]
                    feats = p["meta_feats"]
                    if len(feats) == target_meta - 1:
                        feats = feats[:-1] + [0.0, feats[-1]]
                    if len(feats) == target_meta:
                        pred_m = sigmoid(sum(w * f for w,f in zip(mw, feats)))
                        grad = 2 * (y - pred_m) * pred_m * (1 - pred_m)
                        self.db.set("meta_weights", [clip(w * 0.998 + CFG.meta_lr * grad * f, -5, 5) for w, f in zip(mw, feats)])
                        self.db.set("meta_n", self.db.get("meta_n", 0) + 1)

                # calibration bucket
                bucket = min(19, int(p.get("p_raw", p["p"]) * 20))
                n, s = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
                self.db.set(f"cal20:global:{bucket}", [0.995*n + 1, 0.995*s + y])

                # recent result memory for recovery
                recent = self.db.get("recent_results", [])
                recent.append("WIN" if won_side else "LOSS")
                if len(recent) > 25: recent = recent[-25:]
                self.db.set("recent_results", recent)

                if won_side and won_num:
                    sticker = CFG.jack_sticker
                elif won_side:
                    sticker = CFG.win_sticker
                else:
                    sticker = CFG.loss_sticker
                self.db.queue("sticker", row["issue"], sticker)
                self.db.c.execute("UPDATE outbox SET sent=2 WHERE kind='prediction' AND issue=? AND sent=0", (row["issue"],))

# ─── NUMBER MODEL ─────────────────────────────────────────────
class NumberModel:
    def __init__(self, db):
        self.db = db

    def normalize(self, vals):
        total = sum(vals)
        if total <= 0: return [1.0/len(vals)] * len(vals)
        return [v/total for v in vals]

    def predict(self, nums):
        ngram_db = self.db.get("num_ngrams", {})
        freq_db = self.db.get("num_freq", {})
        dist = [1.0] * 10

        # base counts
        if freq_db:
            total = sum(freq_db.values())
            if total > 0:
                for i in range(10):
                    dist[i] += 10.0 * (freq_db.get(str(i), 0) / total)

        # history pattern transitions
        for length in range(min(10, len(nums)), 1, -1):
            seq_key = ",".join(str(x) for x in nums[-length:])
            if seq_key in ngram_db:
                trans = ngram_db[seq_key]
                total = sum(trans.values())
                if total >= 5:
                    for k,v in trans.items():
                        dist[int(k)] += (v/total) * (length * 1.3)
                    break

        # local repetition bias
        if nums:
            recent = nums[-5:]
            c = Counter(recent)
            for i in range(10):
                dist[i] += 2.0 * (c.get(i, 0) / max(1, len(recent)))

        dist = self.normalize(dist)
        best_idx = max(range(10), key=lambda i: dist[i])
        return best_idx, dist[best_idx]

# ─── FINAL PREDICTION ENGINE ──────────────────────────────────
class AdvancedEngine(Engine):
    def __init__(self, db):
        super().__init__(db)
        self.number_model = NumberModel(db)

    def _meta_features(self, experts, big_prob):
        return [logit(clip(experts.get(n, 0.5), 0.05, 0.95)) for n in EXPERT_NAMES] + [logit(clip(big_prob, 0.05, 0.95)), 1.0]

    def predict(self, history, issue):
        seq = [r["size"] for r in history]
        nums = [r["num"] for r in history]
        experts, patterns, feat, regime = self.evaluate_experts(seq)
        num_best, num_conf = self.number_model.predict(nums)
        big_prob = 0.0
        if nums:
            dist = self.number_model.predict(nums)  # not used as direct distribution, just for best
            # big probability estimate from numeric distribution
            # We do a better approximation by looking at number model
            big_prob = sum(self.number_model.normalize([1.0]*10)[5:]) if False else 0.5
        # Use actual count of 5-9 numbers in recent history as strong proxy:
        recent_big_share = sum(1 for x in nums[-24:] if x >= 5) / max(1, len(nums[-24:]))
        big_prob = clip(recent_big_share)

        experts["num_prior"] = big_prob

        adjusted = []
        for name, p in experts.items():
            corrected, weight, _ = self.db.adapt(f"expert:{name}:{regime}", p)
            if name == "patterns": weight *= 2.6
            if name == "num_prior": weight *= 2.1
            if name == "regime": weight *= 1.5
            if name == "run" and regime in ("RUN","IMBAL"): weight *= 2.0
            adjusted.append((corrected, weight))

        total_w = sum(w for _,w in adjusted)
        p_avg = sum(p*w for p,w in adjusted) / total_w if total_w else 0.5

        feat_meta = self._meta_features(experts, big_prob)
        mw = self.db.get("meta_weights")
        if not isinstance(mw, list) or len(mw) != len(feat_meta):
            mw = [1.0 / len(EXPERT_NAMES)] * len(EXPERT_NAMES) + [1.0 / len(EXPERT_NAMES), 0.0]
        p_meta = sigmoid(sum(w * f for w,f in zip(mw, feat_meta)))
        alpha = min(self.db.get("meta_n", 0) / 80.0, 1.0)
        p_raw = alpha * p_meta + (1-alpha) * p_avg

        # streak continuation
        if len(seq) >= 4 and run_length(seq) >= 4:
            runat = [1]*len(seq)
            for i in range(1, len(seq)):
                runat[i] = runat[i-1]+1 if seq[i] == seq[i-1] else 1
            cont = 0.0; total = 0
            for i in range(4, len(seq)):
                if runat[i-1] >= 4:
                    total += 1
                    if seq[i] == seq[i-1]: cont += 1
            hist = (cont + 1) / (total + 2) if total else 0.5
            if seq[-1] == 1:
                p_raw = 0.7 * hist + 0.3 * p_raw
            else:
                p_raw = 0.7 * (1-hist) + 0.3 * p_raw

        # recovery adjustment
        recent = self.db.get("recent_results", [])
        if recent and recent[-1] == "LOSS":
            p_raw = clip(p_raw + 0.08)
        elif len(recent) >= 2 and recent[-2] == "LOSS" and recent[-1] == "WIN":
            p_raw = clip(p_raw + 0.04)

        # calibration
        bucket = min(19, int(p_raw * 20))
        nval, sval = self.db.get(f"cal20:global:{bucket}", [0.0, 0.0])
        p_cal = clip((30 * p_raw + sval) / (30 + nval))

        # final side
        pred_side = "BIG" if p_cal >= 0.5 else "SMALL"
        valid_nums = range(5,10) if pred_side == "BIG" else range(0,5)

        # number final selection
        best_num = max(valid_nums, key=lambda x: self.number_model.predict(nums)[0] if False else 0)
        # do real numeric distribution
        dist = [1.0]*10
        if nums:
            freq = Counter(nums)
            total = sum(freq.values())
            if total > 0:
                for i in range(10):
                    dist[i] += 3.0 * (freq.get(i, 0) / total)
            for length in range(min(10, len(nums)), 1, -1):
                seq_key = ",".join(str(x) for x in nums[-length:])
                ngram_db = self.db.get("num_ngrams", {})
                if seq_key in ngram_db:
                    trans = ngram_db[seq_key]
                    total = sum(trans.values())
                    if total >= 5:
                        for k,v in trans.items():
                            dist[int(k)] += (v/total) * (length * 1.2)
                        break
        total_dist = sum(dist)
        dist_norm = [d/total_dist for d in dist]
        wanted = [dist_norm[i] for i in valid_nums]
        best_num = max(valid_nums, key=lambda x: dist_norm[x])
        num_prob = dist_norm[best_num]

        # use direct number model if stronger
        nm_best, nm_prob = self.number_model.predict(nums)
        if nm_best in valid_nums and nm_prob >= num_prob:
            best_num = nm_best
            num_prob = nm_prob

        prev_regime = get_regime(seq[:-1]) if len(seq) > 1 else "N/A"
        trend_shift = f"{prev_regime} ➡️ {regime}" if prev_regime != regime else None

        return {
            "issue": issue,
            "p": p_cal,
            "p_raw": p_raw,
            "side": pred_side,
            "num": best_num,
            "num_probability": num_prob,
            "regime": regime,
            "source": "🎯 Regime-Adaptive Advanced",
            "trend_shift": trend_shift,
            "pattern_str": _pattern_str(patterns),
            "experts": experts,
            "features": feat,
            "meta_feats": feat_meta,
            "patterns": [{"key": k, "p": p, "support": s} for _, p, s, _, k in patterns]
        }

# ─── PATTERN STRING ────────────────────────────────────────────
def _pattern_str(patterns):
    if not patterns: return "None"
    top = patterns[0]
    key = top[4]; bits = key.split(":")[-1]
    arrow = "⬆️" if top[1] >= 0.5 else "⬇️"
    return f"<code>{bits}</code> {arrow}"

# ─── FORMATTER ──────────────────────────────────────────────
def format_prediction(p):
    conf = max(p["p"], 1-p["p"]) * 100
    filled = int(conf // 10)
    bar = "█"*filled + "░"*(10-filled)
    side = "BIG" if p["side"] == "BIG" else "SMALL"
    color = "🔴" if side == "BIG" else "🔵"
    quality = "🔥 HOT VIP" if conf >= 70 else "⭐ VIP" if conf >= 60 else "✅ STRONG" if conf >= 55 else "📉 CAUTIOUS"
    stake = "2x" if conf >= 70 else "1.5x" if conf >= 60 else "1x" if conf >= 55 else "0.5x"
    shift_text = f"\n🔄 <b>TREND SHIFT:</b> {escape(p['trend_shift'])}" if p.get("trend_shift") else ""
    pat_text = f"\n🧩 <b>PATTERN:</b> {p.get('pattern_str','None')}" if p.get("pattern_str") and p.get("pattern_str") != "None" else ""
    source_text = f"\n🧠 <b>SOURCE:</b> {p.get('source', 'Unknown')}"
    edge_text = f"\n📐 <b>EDGE:</b> {abs(p['p'] - 0.5)*200:.1f}% | <b>CI:</b> {conf:.1f}%" if p.get("p") else ""
    return (
        f"⚡️ <b>{escape(CFG.name)}</b> ⚡️\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 <b>TARGET:</b> {color} <b>{side}</b> → <code>{p['num']}</code>\n"
        f"📅 <b>PERIOD:</b> <code>{escape(p['issue'])}</code>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>CONFIDENCE:</b> {conf:.1f}% [{bar}]\n"
        f"🏆 <b>QUALITY:</b> {quality} (Stake: {stake}){edge_text}\n"
        f"🌊 <b>REGIME:</b> {escape(p.get('regime','N/A'))}{source_text}{shift_text}{pat_text}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )

# ─── RUNTIME ─────────────────────────────────────────────────
class Runtime:
    def __init__(self, db):
        self.db = db
        self.engine = AdvancedEngine(db)
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
        if not sources: raise RuntimeError("HistoryUnavailable")
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
                self.last_fetch = time.time(); self.error = None; failures = 0
                self.engine.settle(history)
                latest = int(history[-1]["issue"])
                self.db.set("latest_seen", latest)
                if len(history) >= CFG.min_history:
                    issue = str(latest + 1)
                    exists = self.db.c.execute("SELECT 1 FROM predictions WHERE issue=?", (issue,)).fetchone()
                    if not exists:
                        prediction = self.engine.predict(history, issue)
                        with self.db.transaction():
                            self.db.c.execute("INSERT INTO predictions VALUES (?,?,?,?,NULL)", (issue, time.time(), "PENDING", json.dumps(prediction)))
                            self.db.queue("prediction", issue, format_prediction(prediction), time.time() + CFG.delivery_ttl)
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
                await self.pause(0.5); continue
            delay = backoff
            try:
                if row["kind"] == "sticker":
                    url = endpoint_sticker; payload = {"chat_id": CFG.chat, "sticker": row["text"]}
                else:
                    url = endpoint_msg; payload = {"chat_id": CFG.chat, "text": row["text"], "parse_mode":"HTML"}
                async with session.post(url, json=payload) as response:
                    data = await response.json(content_type=None)
                    if response.status == 200 and data.get("ok"):
                        self.db.c.execute("UPDATE outbox SET sent=1 WHERE id=?", (row["id"],))
                        backoff = 1.0; continue
                    code = data.get("error_code", response.status)
                    if code == 429:
                        delay = float(data.get("parameters", {}).get("retry_after", 10))
                    elif code in (400,401,403,404):
                        delay = 60
                        if code == 400 and row["kind"] == "sticker":
                            LOG.error(f"Invalid Sticker ID: {row['text']}")
                            self.db.c.execute("UPDATE outbox SET sent=2 WHERE id=?", (row["id"],)); continue
            except asyncio.CancelledError: raise
            except Exception as e: LOG.warning(f"Telegram fail: {e}")
            await self.pause(delay); backoff = min(60.0, backoff * 2)

    async def health(self, request):
        total = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status IN ('WIN','LOSS')").fetchone()[0]
        wins = self.db.c.execute("SELECT COUNT(*) FROM predictions WHERE status='WIN'").fetchone()[0]
        acc = (wins / total * 100) if total else 0.0
        return web.json_response({
            "status": "ok" if self.last_fetch and time.time() - self.last_fetch < 90 else "degraded",
            "history": self.db.c.execute("SELECT COUNT(*) FROM history").fetchone()[0],
            "settled": total, "wins": wins, "win_rate": f"{acc:.1f}%"
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
    app = web.Application(); app.router.add_get("/", runtime.health)
    runner = web.AppRunner(app); tasks = []; stop_task = None
    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", CFG.port).start()
        timeout = aiohttp.ClientTimeout(total=CFG.timeout)
        connector = aiohttp.TCPConnector(limit=8, ttl_dns_cache=300, ssl=False, family=socket.AF_INET)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            tasks = [asyncio.create_task(runtime.worker(session), name="worker"),
                     asyncio.create_task(runtime.sender(session), name="sender")]
            stop_task = asyncio.create_task(runtime.stop.wait())
            LOG.info(f"{CFG.name} V22 Started (Regime-Adaptive Advanced Engine)")
            await asyncio.wait([*tasks, stop_task], return_when=asyncio.FIRST_COMPLETED)
            runtime.stop.set()
            for task in tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc: raise exc
            for task in tasks: task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        runtime.stop.set()
        for task in tasks: task.cancel()
        if stop_task: stop_task.cancel()
        await asyncio.gather(*tasks, *([stop_task] if stop_task else []), return_exceptions=True)
        await runner.cleanup(); db.c.close()

if __name__ == "__main__":
    asyncio.run(main())
