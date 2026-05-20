import json, math, threading, time, os, requests
from collections import defaultdict, Counter, deque
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.220
SKIP_ENTROPY_THRESHOLD = 2.8
BRAKE_TRIGGER          = 3
BRAKE_PAUSE            = 3
POLL_INTERVAL          = 5

HIGH_MULT_CLASSES      = {2, 3, 4, 7}

HIGH_MULT_EV        = {2: 9, 3: 20, 4: 13, 7: 34}
HIGH_MULT_EV_TARGET = 1.0

TARGET_PLAY_MIN  = 0.45
TARGET_PLAY_MAX  = 0.65
ENT_THRESH_MIN   = 2.65
ENT_THRESH_MAX   = 3.00

TOP1_THRESH_MIN  = 0.190
TOP1_THRESH_MAX  = 0.260
TOP1_THRESH_STEP = 0.005

RESET_HOUR_IST   = 5
RESET_MINUTE_IST = 30
IST              = pytz.timezone("Asia/Kolkata")

CLASS_NAMES  = {1:"Purple",2:"10x",3:"25x",4:"15x",5:"Yellow",6:"Lt. Green",7:"50x",8:"Dk. Green"}
CLASS_COLORS = {1:"#a855f7",2:"#ef4444",3:"#f97316",4:"#eab308",5:"#facc15",6:"#22c55e",7:"#06b6d4",8:"#16a34a"}

FETCH_BASE    = "https://m.starmakerstudios.com/go-v1/ssc/2711/records?start_round="
FETCH_HEADERS = {
    'User-Agent': "sm/9.9.4/Android/13/google play/d48399ffafa2d343/wifi/en-IN/SM-M325F/10977524107285207///India",
    'Accept':     "application/json, text/plain, */*",
    'Cookie':     "PHPSESSID=pd6mapbqfhbk3e7argj51uh1ts; oauth_token=94le54aFnKy5CrbNzo7s903FOWniysVT"
}

# ── PATTERN DETECTION CONFIG ──────────────────────────────────────────────────
PATTERN_BASE_WEIGHTS = {
    7: 3.0, 3: 2.0, 4: 1.5, 2: 1.0,
    1: 1.0, 5: 1.0, 6: 1.0, 8: 1.0,
}

PATTERN_GROUPS = {
    "high_mult": {"classes": {2, 3, 4, 7}, "default_threshold": 5.0},
    "cls1":      {"classes": {1},           "default_threshold": 4.0},
    "cls5":      {"classes": {5},           "default_threshold": 4.0},
    "cls6":      {"classes": {6},           "default_threshold": 4.0},
    "cls8":      {"classes": {8},           "default_threshold": 4.0},
}

PATTERN_LOOKBACK_DEFAULT = 10
PATTERN_LOOKBACK_MIN     = 6
PATTERN_LOOKBACK_MAX     = 16

PATTERN_DECAY_DEFAULT    = 0.5
PATTERN_DECAY_MIN        = 0.25
PATTERN_DECAY_MAX        = 0.75

PATTERN_HIT_BUFFER       = 100

PATTERN_BOOST_MIN_DEFAULT = 0.15
PATTERN_BOOST_MAX_DEFAULT = 0.35
PATTERN_BOOST_SCALE       = 0.04
PATTERN_BOOST_EVAL_WINDOW = 50
PATTERN_BOOST_STEP        = 0.02

# ── DYNAMIC BRAKE CONFIG ──────────────────────────────────────────────────────
BRAKE_TRIGGER_MIN      = 2
BRAKE_TRIGGER_MAX      = 5
BRAKE_TRIGGER_DEFAULT  = 3

BRAKE_PAUSE_MIN        = 2
BRAKE_PAUSE_MAX        = 6
BRAKE_PAUSE_DEFAULT    = 3

BRAKE_HITRATE_WINDOW   = 30
BRAKE_HITRATE_LOW      = 0.50
BRAKE_HITRATE_HIGH     = 0.68

BRAKE_CONF_BUFFER      = 40
BRAKE_CONF_HIGH        = 0.30
BRAKE_CONF_LOW         = 0.24

BRAKE_TRIGGER_STEP     = 1
BRAKE_PAUSE_STEP       = 1

# ── DYNAMIC TRAIN_ROUNDS CONFIG ───────────────────────────────────────────────
TRAIN_ROUNDS_MIN     = 30
TRAIN_ROUNDS_MAX     = 80
TRAIN_ROUNDS_DEFAULT = 50

# ── DYNAMIC MARKOV WEIGHT CONFIG ──────────────────────────────────────────────
MARKOV_WEIGHT_BUFFER = 60
MARKOV_WEIGHT_MIN    = 0.02
MARKOV_WEIGHT_MAX    = 0.40
MARKOV_WEIGHT_STEP   = 0.015

# ── DYNAMIC BONUS THRESHOLD CONFIG ───────────────────────────────────────────
BONUS_CONF_THRESH_DEFAULT = 0.10
BONUS_CONF_THRESH_MIN     = 0.06
BONUS_CONF_THRESH_MAX     = 0.18
BONUS_CONF_THRESH_STEP    = 0.005
BONUS_EVAL_WINDOW         = 40

# ── CLASS HIT RATE CONFIG ─────────────────────────────────────────────────────
CLASS_HIT_BUF_MAX     = 60
CLASS_HIT_MIN_SAMPLES = 8
CLASS_HIT_BOOST_SCALE = 0.30

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 1: REGIME DETECTION
# ══════════════════════════════════════════════════════════════════════════════
REGIMES = {"stable", "chaotic", "high_mult", "repeat_heavy", "alternating"}

REGIME_WINDOW = 30   # rounds to look back for regime detection

# Per-regime weight multipliers applied on top of base dynamic weights
REGIME_WEIGHT_MODS = {
    "stable":       {"wb": 1.0, "wm1": 1.0, "wm2": 1.1, "wm3": 1.2, "wm4": 1.3, "wr": 1.0, "wv": 1.0, "wo": 1.0},
    "chaotic":      {"wb": 1.5, "wm1": 1.2, "wm2": 0.9, "wm3": 0.7, "wm4": 0.5, "wr": 0.8, "wv": 1.2, "wo": 1.3},
    "high_mult":    {"wb": 0.8, "wm1": 1.0, "wm2": 1.1, "wm3": 1.2, "wm4": 1.2, "wr": 1.3, "wv": 1.1, "wo": 0.9},
    "repeat_heavy": {"wb": 0.9, "wm1": 1.3, "wm2": 1.4, "wm3": 1.2, "wm4": 1.0, "wr": 1.2, "wv": 1.3, "wo": 0.8},
    "alternating":  {"wb": 1.2, "wm1": 1.4, "wm2": 1.5, "wm3": 1.1, "wm4": 0.8, "wr": 0.9, "wv": 1.0, "wo": 1.1},
}

REGIME_ENTROPY_ADJUST = {
    "stable":       0.0,
    "chaotic":     +0.15,
    "high_mult":   -0.05,
    "repeat_heavy":-0.05,
    "alternating":  0.05,
}

def detect_regime(rewards):
    """Classify current market regime using entropy, repeat rate, mult freq, transitions, streaks."""
    if len(rewards) < REGIME_WINDOW:
        return "stable"
    window = rewards[-REGIME_WINDOW:]

    # Entropy of window
    cnt = Counter(window)
    total = len(window)
    ent = -sum((v/total) * math.log2(v/total) for v in cnt.values() if v > 0)

    # Repeat rate: how often consecutive values repeat
    repeats = sum(1 for i in range(1, len(window)) if window[i] == window[i-1])
    repeat_rate = repeats / (len(window) - 1)

    # High-mult frequency
    hm_freq = sum(1 for x in window if x in HIGH_MULT_CLASSES) / total

    # Alternating score: count alternating pairs (a,b,a pattern)
    alt_count = sum(1 for i in range(2, len(window)) if window[i] == window[i-2] and window[i] != window[i-1])
    alt_rate = alt_count / max(1, len(window) - 2)

    # Transition entropy (uniqueness of transitions)
    trans_pairs = [(window[i], window[i+1]) for i in range(len(window)-1)]
    trans_cnt = Counter(trans_pairs)
    trans_ent = -sum((v/len(trans_pairs)) * math.log2(v/len(trans_pairs)) for v in trans_cnt.values() if v > 0)

    # Scoring
    if ent > 2.85 and trans_ent > 4.5:
        return "chaotic"
    if hm_freq > 0.30:
        return "high_mult"
    if repeat_rate > 0.30:
        return "repeat_heavy"
    if alt_rate > 0.25:
        return "alternating"
    return "stable"

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 4: ENTROPY TREND (velocity + acceleration)
# ══════════════════════════════════════════════════════════════════════════════
ENTROPY_TREND_WINDOW = 8   # compute entropy over last N-round windows

def compute_entropy_trend(rewards):
    """Returns (entropy_velocity, entropy_acceleration) using rolling window entropies."""
    if len(rewards) < ENTROPY_TREND_WINDOW * 3:
        return 0.0, 0.0
    def window_entropy(seq):
        cnt = Counter(seq); n = len(seq)
        return -sum((v/n)*math.log2(v/n) for v in cnt.values() if v > 0)
    n = len(rewards)
    e_old   = window_entropy(rewards[n - ENTROPY_TREND_WINDOW*3 : n - ENTROPY_TREND_WINDOW*2])
    e_mid   = window_entropy(rewards[n - ENTROPY_TREND_WINDOW*2 : n - ENTROPY_TREND_WINDOW])
    e_new   = window_entropy(rewards[n - ENTROPY_TREND_WINDOW   : n])
    velocity     = e_new - e_mid
    acceleration = (e_new - e_mid) - (e_mid - e_old)
    return round(velocity, 4), round(acceleration, 4)

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 2: TRANSITION MOMENTUM
# ══════════════════════════════════════════════════════════════════════════════
MOMENTUM_SHORT = 10
MOMENTUM_LONG  = 30

def compute_class_momentum(rewards):
    """Returns dict {class: momentum_score} — positive = rising, negative = falling."""
    if len(rewards) < MOMENTUM_LONG:
        return {i: 0.0 for i in range(1, 9)}
    recent = rewards[-MOMENTUM_SHORT:]
    older  = rewards[-MOMENTUM_LONG : -MOMENTUM_SHORT]
    rc = Counter(recent); oc = Counter(older)
    momentum = {}
    for cls in range(1, 9):
        r_prob = rc.get(cls, 0) / len(recent)
        o_prob = oc.get(cls, 0) / max(1, len(older))
        momentum[cls] = round(r_prob - o_prob, 4)
    return momentum

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 3: ANTI-PATTERN DETECTION
# ══════════════════════════════════════════════════════════════════════════════
ANTI_PATTERN_WINDOW  = 40
ANTI_PATTERN_MIN_OBS = 5
ANTI_BREAK_THRESH    = 0.55   # if a pattern breaks > 55% of the time, penalize

def compute_anti_pattern_penalties(rewards):
    """
    Track patterns of length 2 and 3 that consistently fail to continue.
    Returns {class: penalty_factor} where penalty_factor in [0.0, 1.0].
    1.0 = no penalty, lower = penalize.
    """
    penalties = {i: 1.0 for i in range(1, 9)}
    if len(rewards) < ANTI_PATTERN_WINDOW:
        return penalties
    window = rewards[-ANTI_PATTERN_WINDOW:]
    n = len(window)
    # For each length-2 suffix, track how often the predicted next class is actually wrong
    fail_counts = defaultdict(int)
    total_counts = defaultdict(int)
    for i in range(n - 2):
        key = tuple(window[i:i+2])
        actual_next = window[i+2]
        total_counts[key] += 1
        # What Markov would predict
        # (We just check: does the pattern "continue" to any expected class?)
        # We track which class DIDN'T follow after this pattern
        for cls in range(1, 9):
            if cls != actual_next:
                fail_counts[(key, cls)] += 1
    # Now for current suffix, see if high failure rate exists
    if n >= 2:
        cur_key = tuple(window[-2:])
        for cls in range(1, 9):
            obs = total_counts.get(cur_key, 0)
            if obs >= ANTI_PATTERN_MIN_OBS:
                fails = fail_counts.get((cur_key, cls), 0)
                fail_rate = fails / obs
                if fail_rate > ANTI_BREAK_THRESH:
                    # cls rarely follows this pattern — penalize it
                    penalties[cls] = max(0.5, 1.0 - (fail_rate - ANTI_BREAK_THRESH) * 2.0)
    return penalties

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 5: FAILURE LEARNING LAYER (MISS CONTEXT TRACKING)
# ══════════════════════════════════════════════════════════════════════════════
MISS_CONTEXT_MAX = 200

_miss_contexts = []   # list of dicts with miss context info

def record_miss_context(ctx):
    """Store context of a missed prediction for analysis."""
    global _miss_contexts
    _miss_contexts.append(ctx)
    if len(_miss_contexts) > MISS_CONTEXT_MAX:
        _miss_contexts.pop(0)

def compute_miss_suppression(current_entropy, current_top1, current_regime):
    """
    Analyze miss contexts and return a suppression factor [0.6, 1.0].
    Lower = current conditions historically produce many misses.
    """
    if len(_miss_contexts) < 10:
        return 1.0
    # Find contexts similar to current
    similar = []
    for ctx in _miss_contexts[-80:]:
        ent_diff  = abs(ctx.get("entropy", 3.0) - current_entropy)
        top1_diff = abs(ctx.get("top1", 0.25) - current_top1)
        regime_match = ctx.get("regime", "stable") == current_regime
        if ent_diff < 0.3 and top1_diff < 0.05 and regime_match:
            similar.append(ctx)
    if len(similar) < 5:
        return 1.0
    miss_rate = len(similar) / max(1, len(_miss_contexts[-80:]))
    # suppress confidence when similar conditions historically caused many misses
    suppression = max(0.60, 1.0 - miss_rate * 0.8)
    return round(suppression, 4)

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 6: CYCLE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
CYCLE_PERIODS   = [5, 8, 13, 21, 34]
CYCLE_MIN_REPS  = 3   # need at least this many full cycles to trust
CYCLE_WINDOW    = 120

def compute_cycle_scores(rewards):
    """
    Test autocorrelation at Fibonacci-ish periods.
    Returns {class: cycle_boost} — positive means class is due at this period.
    """
    boosts = {i: 0.0 for i in range(1, 9)}
    if len(rewards) < max(CYCLE_PERIODS) * CYCLE_MIN_REPS:
        return boosts
    window = rewards[-CYCLE_WINDOW:]
    n = len(window)
    for period in CYCLE_PERIODS:
        if n < period * CYCLE_MIN_REPS:
            continue
        # Binary series per class
        for cls in range(1, 9):
            series = [1 if x == cls else 0 for x in window]
            # Autocorrelation at this lag
            lag_series = series[period:]
            base_series = series[:len(lag_series)]
            if len(base_series) < 10:
                continue
            corr_num = sum(a * b for a, b in zip(base_series, lag_series))
            corr_den = math.sqrt(
                sum(a*a for a in base_series) * sum(b*b for b in lag_series) + 1e-9
            )
            autocorr = corr_num / corr_den
            # Check if class appeared `period` rounds ago
            if n >= period and window[n - period] == cls:
                boosts[cls] += autocorr * 0.15
    return {k: round(v, 4) for k, v in boosts.items()}

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 7: CONFIDENCE CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
CALIB_BUCKETS  = 10   # number of buckets
CALIB_BUF_MAX  = 300

_calib_buffer  = []   # list of (raw_conf, hit) tuples

def record_calibration(raw_conf, hit):
    """Record a (confidence, hit/miss) pair for calibration."""
    _calib_buffer.append((raw_conf, int(hit)))
    if len(_calib_buffer) > CALIB_BUF_MAX:
        _calib_buffer.pop(0)

def get_calibrated_conf(raw_conf):
    """Map raw confidence to empirical hit rate from calibration buffer."""
    if len(_calib_buffer) < 30:
        return raw_conf
    # Find entries in same bucket
    bucket_size = 1.0 / CALIB_BUCKETS
    bucket_lo = int(raw_conf / bucket_size) * bucket_size
    bucket_hi = bucket_lo + bucket_size
    bucket_entries = [(c, h) for c, h in _calib_buffer if bucket_lo <= c < bucket_hi]
    if len(bucket_entries) < 5:
        return raw_conf
    actual_rate = sum(h for _, h in bucket_entries) / len(bucket_entries)
    # Blend raw and empirical (50/50 blend to avoid instability)
    return round(0.5 * raw_conf + 0.5 * actual_rate, 4)

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 8: VARIABLE-LENGTH MARKOV (Context Reliability)
# ══════════════════════════════════════════════════════════════════════════════
MARKOV_MIN_SAMPLES = 8   # minimum transitions needed to trust an order

def get_reliable_markov_order(t1, t2, t3, t4, k1, k2, k3, k4):
    """
    Returns the highest reliable Markov order, falling back if context sparse.
    Also returns the corresponding transition table and key.
    """
    candidates = [
        (4, t4, k4),
        (3, t3, k3),
        (2, t2, k2),
        (1, t1, k1),
    ]
    for order, t, k in candidates:
        if k in t and sum(t[k].values()) >= MARKOV_MIN_SAMPLES:
            return order, t, k
    return 1, t1, k1

# ══════════════════════════════════════════════════════════════════════════════
# IMPROVEMENT 9: RECENCY-WEIGHTED MISS PENALTY (Dead Class Suppression)
# ══════════════════════════════════════════════════════════════════════════════
DEAD_CLASS_WINDOW      = 20
DEAD_CLASS_MISS_THRESH = 0.75   # if miss rate >= this, suppress
DEAD_CLASS_PENALTY     = 0.60   # multiply score by this factor

def compute_dead_class_penalties(rewards, class_hit_buf):
    """
    Suppress classes that are predicted frequently but miss consistently in recent rounds.
    Returns {class: penalty_factor}.
    """
    penalties = {i: 1.0 for i in range(1, 9)}
    for cls, buf in class_hit_buf.items():
        if len(buf) < DEAD_CLASS_WINDOW:
            continue
        recent = buf[-DEAD_CLASS_WINDOW:]
        miss_rate = 1.0 - (sum(recent) / len(recent))
        if miss_rate >= DEAD_CLASS_MISS_THRESH:
            # Apply graduated penalty
            severity = (miss_rate - DEAD_CLASS_MISS_THRESH) / (1.0 - DEAD_CLASS_MISS_THRESH)
            penalty = 1.0 - severity * (1.0 - DEAD_CLASS_PENALTY)
            penalties[cls] = round(max(DEAD_CLASS_PENALTY, penalty), 4)
    return penalties

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════════════════════════
_lock              = threading.Lock()
_rewards           = []
_raw_rounds        = []
_sim_stats         = {}
_brake_left        = 0
_live_log          = []
_pending_pred      = None
_skip_pending_pred = None
_entropy_threshold = SKIP_ENTROPY_THRESHOLD
_top1_threshold    = SKIP_TOP1_THRESHOLD
_last_reset_date   = None

_cached_pred = None
_current_regime  = "stable"   # NEW: live regime tracking

_fetch_status = {
    "last_attempt": None, "last_success": None,
    "last_error":   None, "total_fetched": 0,
    "status": "starting", "last_reset": None,
}

# ── PATTERN STATE ─────────────────────────────────────────────────────────────
_hit_pattern_scores  = {g: [] for g in PATTERN_GROUPS}
_dynamic_threshold   = {g: PATTERN_GROUPS[g]["default_threshold"] for g in PATTERN_GROUPS}
_last_pattern_info   = {}
_dynamic_lookback    = PATTERN_LOOKBACK_DEFAULT
_dynamic_decay       = PATTERN_DECAY_DEFAULT
_dynamic_boost_max   = PATTERN_BOOST_MAX_DEFAULT
_dynamic_boost_min   = PATTERN_BOOST_MIN_DEFAULT

_boost_eval_log      = []

# ── DYNAMIC BRAKE STATE ───────────────────────────────────────────────────────
_dynamic_brake_trigger = BRAKE_TRIGGER_DEFAULT
_dynamic_brake_pause   = BRAKE_PAUSE_DEFAULT
_brake_play_results    = []
_brake_loss_confs      = []

# ── DYNAMIC TRAIN_ROUNDS STATE ────────────────────────────────────────────────
_dynamic_train_rounds  = TRAIN_ROUNDS_DEFAULT

# ── DYNAMIC MARKOV WEIGHTS STATE ─────────────────────────────────────────────
_markov_hit_buf = {1: [], 2: [], 3: [], 4: []}
_dynamic_markov_w = {
    "wb":  0.05,
    "wm1": 0.10,
    "wm2": 0.15,
    "wm3": 0.25,
    "wm4": 0.20,
    "wr":  0.10,
    "wv":  0.10,
    "wo":  0.05,
}

# ── DYNAMIC BONUS THRESHOLD STATE ────────────────────────────────────────────
_dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT
_bonus_eval_log        = []

# ── CLASS HIT RATE STATE ─────────────────────────────────────────────────────
_class_hit_buf = {i: [] for i in range(1, 9)}

# ── ENTROPY TREND STATE ───────────────────────────────────────────────────────
_entropy_velocity     = 0.0
_entropy_acceleration = 0.0

# ── DAILY RESET ───────────────────────────────────────────────────────────────
def _should_reset():
    global _last_reset_date
    now_ist  = datetime.now(IST)
    today    = now_ist.date()
    past_530 = (now_ist.hour > RESET_HOUR_IST or
                (now_ist.hour == RESET_HOUR_IST and now_ist.minute >= RESET_MINUTE_IST))
    return past_530 and _last_reset_date != today

def _do_reset():
    global _last_reset_date, _pending_pred, _cached_pred, _skip_pending_pred
    global _dynamic_brake_trigger, _dynamic_brake_pause
    global _dynamic_train_rounds, _top1_threshold
    global _dynamic_lookback, _dynamic_decay, _dynamic_boost_max, _dynamic_boost_min
    global _dynamic_bonus_thresh, _current_regime
    global _entropy_velocity, _entropy_acceleration
    now_ist = datetime.now(IST)
    print(f"[Reset] 5:30 AM IST — wiping data ({now_ist.date()})")
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    with _lock:
        _rewards.clear(); _raw_rounds.clear()
        _sim_stats.clear(); _live_log.clear()
        _pending_pred      = None
        _skip_pending_pred = None
        _cached_pred       = None
        _fetch_status["last_reset"] = now_ist.strftime("%d %b %Y %H:%M IST")
        _fetch_status["status"]     = "reset_done"
        for g in PATTERN_GROUPS:
            _hit_pattern_scores[g].clear()
            _dynamic_threshold[g] = PATTERN_GROUPS[g]["default_threshold"]
        _last_pattern_info.clear()
        _boost_eval_log.clear()
        _brake_play_results.clear()
        _brake_loss_confs.clear()
        _bonus_eval_log.clear()
        for o in _markov_hit_buf: _markov_hit_buf[o].clear()
        for i in range(1, 9): _class_hit_buf[i].clear()
        _miss_contexts.clear()
        _calib_buffer.clear()
    _dynamic_brake_trigger = BRAKE_TRIGGER_DEFAULT
    _dynamic_brake_pause   = BRAKE_PAUSE_DEFAULT
    _dynamic_train_rounds  = TRAIN_ROUNDS_DEFAULT
    _top1_threshold        = SKIP_TOP1_THRESHOLD
    _dynamic_lookback      = PATTERN_LOOKBACK_DEFAULT
    _dynamic_decay         = PATTERN_DECAY_DEFAULT
    _dynamic_boost_max     = PATTERN_BOOST_MAX_DEFAULT
    _dynamic_boost_min     = PATTERN_BOOST_MIN_DEFAULT
    _dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT
    _current_regime        = "stable"
    _entropy_velocity      = 0.0
    _entropy_acceleration  = 0.0
    _last_reset_date = now_ist.date()
    print("[Reset] All data cleared.")

# ── FILE HELPERS ──────────────────────────────────────────────────────────────
def _load_file():
    try:
        with open(DATA_FILE) as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return sorted(data, key=lambda x: x["round"])
    except Exception:
        pass
    return []

def _save_file(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2)

# ── FETCHER ───────────────────────────────────────────────────────────────────
def fetch_new():
    existing = _load_file()
    known    = {r["round"] for r in existing}
    url      = FETCH_BASE
    new_recs = []
    while url:
        try:
            resp = requests.get(url, headers=FETCH_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            err = str(e)
            if "403" in err or "401" in err:
                err = "AUTH FAILED (403) — Cookie expired!"
            print(f"[Fetcher] {err}")
            with _lock:
                _fetch_status["last_error"] = err
                _fetch_status["status"]     = "error"
            break
        records = data.get("list", [])
        if not records:
            break
        fresh = []; overlap = False
        for r in records:
            rv = r.get("round")
            if rv in known: overlap = True; break
            fresh.append(r); known.add(rv)
        new_recs.extend(fresh)
        if overlap or not data.get("has_more") or not data.get("callback"):
            break
        url = data["callback"]
        time.sleep(0.3)
    if new_recs:
        all_recs = sorted(new_recs + existing, key=lambda x: x["round"])
        _save_file(all_recs)
        return len(new_recs), all_recs
    return 0, existing

# ── MARKOV ENGINE ─────────────────────────────────────────────────────────────
def build_trans(seq, order):
    t = defaultdict(lambda: defaultdict(int))
    for i in range(len(seq) - order):
        key = tuple(seq[i:i+order])
        t[key][seq[i+order]] += 1
    tp = {}
    for k, v in t.items():
        s = sum(v.values())
        tp[k] = {kk: vv/s for kk, vv in v.items()}
    return t, tp

def build_global_stats(rw):
    freq = Counter(rw); prob = {k: v/len(rw) for k, v in freq.items()}
    t1,tp1=build_trans(rw,1); t2,tp2=build_trans(rw,2)
    t3,tp3=build_trans(rw,3); t4,tp4=build_trans(rw,4)
    ls={}; gaps=defaultdict(list)
    for i,r in enumerate(rw):
        if r in ls: gaps[r].append(i-ls[r])
        ls[r]=i
    avg_gap={k:sum(v)/len(v) if v else 8 for k,v in gaps.items()}
    rl=defaultdict(list); i=0
    while i<len(rw):
        j=i
        while j<len(rw) and rw[j]==rw[i]: j+=1
        rl[rw[i]].append(j-i); i=j
    avg_run={k:sum(v)/len(v) for k,v in rl.items()}
    return prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,avg_gap,avg_run

# ── DYNAMIC TRAIN_ROUNDS ─────────────────────────────────────────────────────
def compute_dynamic_train_rounds(total_rounds):
    scaled = TRAIN_ROUNDS_MIN + int((total_rounds / 100) * 3)
    return max(TRAIN_ROUNDS_MIN, min(TRAIN_ROUNDS_MAX, scaled))

# ── DYNAMIC MARKOV WEIGHTS ────────────────────────────────────────────────────
def recalibrate_markov_weights(hit_bufs):
    with _lock:
        cur_w = dict(_dynamic_markov_w)
    order_keys = {1: "wm1", 2: "wm2", 3: "wm3", 4: "wm4"}
    new_w = dict(cur_w)
    for order, key in order_keys.items():
        buf = hit_bufs.get(order, [])
        if len(buf) < 10:
            continue
        window   = buf[-MARKOV_WEIGHT_BUFFER:]
        hit_rate = sum(window) / len(window)
        if hit_rate > 0.65:
            new_w[key] = min(MARKOV_WEIGHT_MAX, cur_w[key] + MARKOV_WEIGHT_STEP)
        elif hit_rate < 0.45:
            new_w[key] = max(MARKOV_WEIGHT_MIN, cur_w[key] - MARKOV_WEIGHT_STEP)
        if abs(new_w[key] - cur_w[key]) > 0.001:
            print(f"[MarkovW] Order-{order} ({key}): {cur_w[key]:.3f}→{new_w[key]:.3f} "
                  f"(hit_rate={hit_rate:.2%}, n={len(window)})")
    return new_w

# ── DYNAMIC PATTERN PARAMS ────────────────────────────────────────────────────
def recalibrate_pattern_params(total_rounds, sim_hit_rate):
    global _dynamic_lookback, _dynamic_decay
    new_lookback = max(PATTERN_LOOKBACK_MIN,
                       min(PATTERN_LOOKBACK_MAX,
                           int(6 + (total_rounds / 800) * 10)))
    if sim_hit_rate < 0.50:
        new_decay = max(PATTERN_DECAY_MIN, _dynamic_decay - 0.03)
    elif sim_hit_rate > 0.68:
        new_decay = min(PATTERN_DECAY_MAX, _dynamic_decay + 0.03)
    else:
        new_decay = _dynamic_decay
    if new_lookback != _dynamic_lookback:
        print(f"[PatternParam] Lookback: {_dynamic_lookback}→{new_lookback}")
        _dynamic_lookback = new_lookback
    if abs(new_decay - _dynamic_decay) > 0.001:
        print(f"[PatternParam] Decay: {_dynamic_decay:.3f}→{new_decay:.3f}")
        _dynamic_decay = new_decay

# ── DYNAMIC BOOST EFFECTIVENESS ───────────────────────────────────────────────
def recalibrate_boost_cap(boost_eval_log):
    global _dynamic_boost_max, _dynamic_boost_min
    if len(boost_eval_log) < 20:
        return
    window = boost_eval_log[-PATTERN_BOOST_EVAL_WINDOW:]
    boosted_hits   = [h for b, h in window if b]
    unboosted_hits = [h for b, h in window if not b]
    if len(boosted_hits) < 5 or len(unboosted_hits) < 5:
        return
    boosted_rate   = sum(boosted_hits)   / len(boosted_hits)
    unboosted_rate = sum(unboosted_hits) / len(unboosted_hits)
    cur_max = _dynamic_boost_max
    cur_min = _dynamic_boost_min
    if boosted_rate < unboosted_rate - 0.05:
        new_max = max(PATTERN_BOOST_MIN_DEFAULT, cur_max - PATTERN_BOOST_STEP)
        new_min = max(0.05, cur_min - PATTERN_BOOST_STEP * 0.5)
    elif boosted_rate > unboosted_rate + 0.05:
        new_max = min(0.50, cur_max + PATTERN_BOOST_STEP)
        new_min = min(0.25, cur_min + PATTERN_BOOST_STEP * 0.5)
    else:
        new_max = cur_max
        new_min = cur_min
    if abs(new_max - cur_max) > 0.001:
        print(f"[BoostCap] max: {cur_max:.3f}→{new_max:.3f}")
        _dynamic_boost_max = new_max
        _dynamic_boost_min = new_min

# ── DYNAMIC BONUS THRESHOLD ───────────────────────────────────────────────────
def recalibrate_bonus_thresh(bonus_eval_log):
    global _dynamic_bonus_thresh
    if len(bonus_eval_log) < 15:
        return
    window = bonus_eval_log[-BONUS_EVAL_WINDOW:]
    bonus_rounds    = [(triggered, hit) for triggered, hit in window if triggered]
    no_bonus_rounds = [(triggered, hit) for triggered, hit in window if not triggered]
    if len(bonus_rounds) < 5:
        return
    bonus_hit_rate    = sum(h for _, h in bonus_rounds) / len(bonus_rounds)
    no_bonus_hit_rate = (sum(h for _, h in no_bonus_rounds) / len(no_bonus_rounds)
                         if no_bonus_rounds else 0.60)
    cur = _dynamic_bonus_thresh
    if bonus_hit_rate < no_bonus_hit_rate - 0.08:
        new_thresh = min(BONUS_CONF_THRESH_MAX, cur + BONUS_CONF_THRESH_STEP)
    elif bonus_hit_rate > no_bonus_hit_rate + 0.05:
        new_thresh = max(BONUS_CONF_THRESH_MIN, cur - BONUS_CONF_THRESH_STEP)
    else:
        new_thresh = cur
    if abs(new_thresh - cur) > 0.0001:
        print(f"[BonusThresh] {cur:.4f}→{new_thresh:.4f}")
        _dynamic_bonus_thresh = new_thresh

# ── CLASS HIT RATE RECORDER ───────────────────────────────────────────────────
def _record_class_hits(all_preds, actual_val, source="play"):
    with _lock:
        for cls in all_preds:
            hit_for_cls = 1 if actual_val == cls else 0
            _class_hit_buf[cls].append(hit_for_cls)
            if len(_class_hit_buf[cls]) > CLASS_HIT_BUF_MAX * 3:
                _class_hit_buf[cls].pop(0)
    overall_hit = actual_val in all_preds
    print(f"[ClassHit] ({source}) preds={all_preds} actual={actual_val} "
          f"{'HIT ✓' if overall_hit else 'MISS ✗'}")

# ── PATTERN DETECTOR ─────────────────────────────────────────────────────────
def compute_pattern_scores(rewards):
    with _lock:
        lookback = _dynamic_lookback
        decay    = _dynamic_decay
    if len(rewards) < lookback:
        return {g: 0.0 for g in PATTERN_GROUPS}
    last_n = rewards[-lookback:]
    scores = {g: 0.0 for g in PATTERN_GROUPS}
    for dist_from_end, cls in enumerate(reversed(last_n)):
        distance = dist_from_end + 1
        d        = decay ** (distance - 1)
        weight   = PATTERN_BASE_WEIGHTS.get(cls, 1.0)
        contrib  = weight * d
        for group_name, group_cfg in PATTERN_GROUPS.items():
            if cls in group_cfg["classes"]:
                scores[group_name] += contrib
    return scores

def apply_pattern_boost(scores_dict, pattern_scores, dyn_thresholds):
    with _lock:
        boost_max = _dynamic_boost_max
        boost_min = _dynamic_boost_min
    boosted = dict(scores_dict)
    pattern_info = {}
    any_triggered = False
    for group_name, group_cfg in PATTERN_GROUPS.items():
        pscore    = pattern_scores.get(group_name, 0.0)
        threshold = dyn_thresholds.get(group_name, group_cfg["default_threshold"])
        triggered = pscore >= threshold
        boost_applied = 0.0
        if triggered:
            any_triggered = True
            excess        = pscore - threshold
            boost_applied = min(boost_max,
                                boost_min + excess * PATTERN_BOOST_SCALE)
            for cls in group_cfg["classes"]:
                if cls in boosted:
                    boosted[cls] += boosted[cls] * boost_applied
                else:
                    boosted[cls] = boost_applied * 0.01
        pattern_info[group_name] = {
            "score":         round(pscore, 4),
            "threshold":     round(threshold, 4),
            "triggered":     triggered,
            "boost_applied": round(boost_applied, 4),
        }
    pattern_info["_any_triggered"] = any_triggered
    total = sum(boosted.values()) or 1.0
    boosted = {k: v / total for k, v in boosted.items()}
    return boosted, pattern_info

def update_pattern_hit(group_name, pattern_score):
    buf = _hit_pattern_scores[group_name]
    buf.append(pattern_score)
    if len(buf) > PATTERN_HIT_BUFFER:
        buf.pop(0)
    if buf:
        _dynamic_threshold[group_name] = sum(buf) / len(buf)
    print(f"[Pattern] Group '{group_name}' hit. score={pattern_score:.4f} "
          f"new_threshold={_dynamic_threshold[group_name]:.4f} (buffer={len(buf)})")

# ── DYNAMIC BRAKE CALIBRATION ─────────────────────────────────────────────────
def recalibrate_brake(play_results, loss_confs):
    global _dynamic_brake_trigger, _dynamic_brake_pause
    cur_trigger = _dynamic_brake_trigger
    cur_pause   = _dynamic_brake_pause
    new_trigger = cur_trigger
    hitrate     = None
    if len(play_results) >= 10:
        window  = play_results[-BRAKE_HITRATE_WINDOW:]
        hitrate = sum(window) / len(window)
        if hitrate < BRAKE_HITRATE_LOW:
            new_trigger = max(BRAKE_TRIGGER_MIN, cur_trigger - BRAKE_TRIGGER_STEP)
        elif hitrate > BRAKE_HITRATE_HIGH:
            new_trigger = min(BRAKE_TRIGGER_MAX, cur_trigger + BRAKE_TRIGGER_STEP)
        if new_trigger != cur_trigger:
            print(f"[DynBrake] Trigger: {cur_trigger}→{new_trigger} (hitrate={hitrate:.2%})")
    new_pause = cur_pause
    avg_conf  = None
    if len(loss_confs) >= 5:
        window_confs = loss_confs[-BRAKE_CONF_BUFFER:]
        avg_conf     = sum(window_confs) / len(window_confs)
        if avg_conf > BRAKE_CONF_HIGH:
            new_pause = min(BRAKE_PAUSE_MAX, cur_pause + BRAKE_PAUSE_STEP)
        elif avg_conf < BRAKE_CONF_LOW:
            new_pause = max(BRAKE_PAUSE_MIN, cur_pause - BRAKE_PAUSE_STEP)
        if new_pause != cur_pause:
            print(f"[DynBrake] Pause: {cur_pause}→{new_pause} (avg_loss_conf={avg_conf:.4f})")
    return new_trigger, new_pause, hitrate, avg_conf

# ── TOP1 THRESHOLD ADAPTATION ────────────────────────────────────────────────
def recalibrate_top1_threshold(play_pct):
    global _top1_threshold
    cur = _top1_threshold
    if play_pct < TARGET_PLAY_MIN:
        new = max(TOP1_THRESH_MIN, cur - TOP1_THRESH_STEP)
    elif play_pct > TARGET_PLAY_MAX:
        new = min(TOP1_THRESH_MAX, cur + TOP1_THRESH_STEP)
    else:
        new = cur
    if abs(new - cur) > 0.0001:
        print(f"[Adaptive] Top1Thresh: {cur:.4f}→{new:.4f} (play%={play_pct*100:.1f}%)")
        _top1_threshold = new

# ══════════════════════════════════════════════════════════════════════════════
# CORE SCORING — enhanced with all 9 improvements
# ══════════════════════════════════════════════════════════════════════════════
def score_round(h, prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,ag,ar):
    n = len(h)
    if n < 4:
        ranked = sorted(prob.items(), key=lambda x: -x[1])
        return ([ranked[0][0], ranked[1][0]], {k: 1/8 for k in range(1,9)},
                3.0, 0.125, 0.125, 0.125, ranked[2][0] if len(ranked)>=3 else None, {})

    l1,l2,l3,l4 = h[-1],h[-2],h[-3],h[-4]
    k1=(l1,); k2=(l2,l1); k3=(l3,l2,l1); k4=(l4,l3,l2,l1)

    def rel(t, key): return min(1.0, sum(t[key].values())/30) if key in t else 0
    r2=rel(t2,k2); r3=rel(t3,k3); r4=rel(t4,k4)

    rec = h[-100:] if n>=100 else h
    rs  = h[-20:]  if n>=20  else h
    rc  = Counter(rec); rsc = Counter(rs)

    lp = {}
    for i2, r in enumerate(h): lp[r] = i2
    rv = h[-1]; rl2 = 1
    for i2 in range(n-2, -1, -1):
        if h[i2] == rv: rl2 += 1
        else: break

    gh = defaultdict(list); lsh = {}
    for i2, r in enumerate(h):
        if r in lsh: gh[r].append(i2 - lsh[r])
        lsh[r] = i2
    agh = {r: sum(gh[r])/len(gh[r]) if gh.get(r) else ag.get(r,8) for r in range(1,9)}
    rh = defaultdict(list); i2 = 0
    while i2 < n:
        j = i2
        while j < len(h) and h[j] == h[i2]: j += 1
        rh[h[i2]].append(j - i2); i2 = j
    arh = {r: sum(rh[r])/len(rh[r]) if rh.get(r) else ar.get(r,1.5) for r in range(1,9)}

    # ── IMPROVEMENT 1: Regime-aware weights ───────────────────────────────────
    regime = detect_regime(h)
    regime_mods = REGIME_WEIGHT_MODS.get(regime, REGIME_WEIGHT_MODS["stable"])

    with _lock:
        dw = dict(_dynamic_markov_w)

    # Apply regime multipliers
    dw_regime = {k: dw[k] * regime_mods.get(k, 1.0) for k in dw}

    # ── IMPROVEMENT 8: Variable-length Markov ─────────────────────────────────
    best_order, best_t, best_k = get_reliable_markov_order(t1, t2, t3, t4, k1, k2, k3, k4)

    sc = {}
    for idx in range(1, 9):
        base = prob.get(idx, 0)
        m1 = tp1.get(k1, {}).get(idx, base)
        m2 = tp2.get(k2, {}).get(idx, m1) if r2 > 0 else m1
        m3 = tp3.get(k3, {}).get(idx, m2) if r3 > 0 else m2
        m4 = tp4.get(k4, {}).get(idx, m3) if r4 > 0 else m3

        # Use best reliable order as primary signal
        best_pred = best_t.get(best_k, {}).get(idx, m1)
        # Blend best-order into m4 slot if higher order is reliable
        if best_order >= 3:
            m4 = best_pred

        rp  = rc.get(idx, 0)  / len(rec)
        rsp = rsc.get(idx, 0) / len(rs)
        pos = lp.get(idx, 0)
        agv = agh.get(idx, 8)
        od  = (n - 1 - pos) / agv if agv else 0
        ob  = min(0.5, max(0.0, (od - 1.0) * 0.1))
        rpen = 0.05 if idx == rv and rl2 >= arh.get(idx, 1.5) else 0

        wb  = dw_regime["wb"]
        wm1 = dw_regime["wm1"]
        wm2 = dw_regime["wm2"] * r2
        wm3 = dw_regime["wm3"] * r3
        wm4 = dw_regime["wm4"] * r4
        wr  = dw_regime["wr"]
        wv  = dw_regime["wv"]
        wo  = dw_regime["wo"]
        tw  = wb+wm1+wm2+wm3+wm4+wr+wv+wo or 1
        raw = (wb*base + wm1*m1 + wm2*m2 + wm3*m3 + wm4*m4 + wr*rp + wv*rsp + wo*ob)
        sc[idx] = max(0.0, raw/tw - rpen)

    markov_preds = {}
    for order, tp, key in [(1,tp1,k1),(2,tp2,k2),(3,tp3,k3),(4,tp4,k4)]:
        if key in tp:
            top_cls = max(tp[key], key=tp[key].get)
            markov_preds[order] = top_cls

    ts = sum(sc.values()) or 1
    sc = {k: v/ts for k, v in sc.items()}

    with _lock:
        dyn_thresh = dict(_dynamic_threshold)

    pattern_scores = compute_pattern_scores(h)
    sc, pattern_info = apply_pattern_boost(sc, pattern_scores, dyn_thresh)
    with _lock:
        _last_pattern_info.clear()
        _last_pattern_info.update(pattern_info)
        _last_pattern_info["raw_scores"] = {g: round(v, 4) for g, v in pattern_scores.items()}

    # ── IMPROVEMENT 2: Transition Momentum Boost ──────────────────────────────
    momentum = compute_class_momentum(h)
    for cls in list(sc.keys()):
        m = momentum.get(cls, 0.0)
        if m > 0.02:    # rising class → boost
            sc[cls] *= (1.0 + min(0.30, m * 4.0))
        elif m < -0.02: # falling class → penalize
            sc[cls] *= (1.0 + max(-0.25, m * 3.0))

    # ── IMPROVEMENT 3: Anti-Pattern Penalties ─────────────────────────────────
    anti_penalties = compute_anti_pattern_penalties(h)
    for cls in list(sc.keys()):
        sc[cls] *= anti_penalties.get(cls, 1.0)

    # ── IMPROVEMENT 6: Cycle Score Boost ──────────────────────────────────────
    cycle_boosts = compute_cycle_scores(h)
    for cls in list(sc.keys()):
        sc[cls] *= (1.0 + cycle_boosts.get(cls, 0.0))

    # ── IMPROVEMENT 9: Dead Class / Recency-Miss Penalty ─────────────────────
    with _lock:
        chb_snap = {cls: list(buf) for cls, buf in _class_hit_buf.items()}
    dead_penalties = compute_dead_class_penalties(h, chb_snap)
    for cls in list(sc.keys()):
        sc[cls] *= dead_penalties.get(cls, 1.0)

    # ── CLASS HIT RATE ADJUSTMENT (existing) ─────────────────────────────────
    for cls in list(sc.keys()):
        buf = chb_snap.get(cls, [])
        if len(buf) >= CLASS_HIT_MIN_SAMPLES:
            window   = buf[-CLASS_HIT_BUF_MAX:]
            hit_rate = sum(window) / len(window)
            adjustment = (hit_rate - 0.5) * CLASS_HIT_BOOST_SCALE * 2
            sc[cls] = max(0.0, sc[cls] * (1.0 + adjustment))

    # Re-normalize
    ts = sum(sc.values()) or 1
    sc = {k: v/ts for k, v in sc.items()}

    rk  = sorted(sc.items(), key=lambda x: -x[1])
    ent = -sum(v*math.log2(v) for v in sc.values() if v > 0)
    t3s = rk[2][1] if len(rk) >= 3 else 0.0
    t3c = rk[2][0] if len(rk) >= 3 else None
    return [rk[0][0], rk[1][0]], sc, ent, rk[0][1], rk[1][1], t3s, t3c, markov_preds


def should_play(t1, ent, brake_active=False, regime="stable",
                entropy_velocity=0.0, miss_suppression=1.0):
    if brake_active:
        return False
    with _lock:
        top1_t = _top1_threshold
    # IMPROVEMENT 1: Regime-adjusted entropy threshold
    regime_ent_adj = REGIME_ENTROPY_ADJUST.get(regime, 0.0)
    eff_ent_thresh = _entropy_threshold + regime_ent_adj
    # IMPROVEMENT 4: Entropy velocity — rising chaos → tighten threshold
    if entropy_velocity > 0.2:
        eff_ent_thresh -= 0.10
    elif entropy_velocity > 0.1:
        eff_ent_thresh -= 0.05
    # IMPROVEMENT 7: Calibrated confidence + miss suppression
    calibrated_t1 = get_calibrated_conf(t1)
    effective_t1  = calibrated_t1 * miss_suppression
    return effective_t1 > top1_t and ent < eff_ent_thresh

# ── BONUS PICK LOGIC ──────────────────────────────────────────────────────────
def get_bonus_picks(scores, top2):
    with _lock:
        thresh = _dynamic_bonus_thresh
    sc_map = {int(k): float(v) for k, v in scores.items()}
    qualifying = set()
    for cls in HIGH_MULT_CLASSES:
        prob = sc_map.get(cls, 0.0)
        mult = HIGH_MULT_EV.get(cls, 1)
        ev_pass   = (prob * mult) >= HIGH_MULT_EV_TARGET
        conf_pass = prob > thresh
        if ev_pass or conf_pass:
            qualifying.add(cls)
    if not qualifying:
        return None
    ev_ranked = sorted(
        qualifying,
        key=lambda cls: sc_map.get(cls, 0.0) * HIGH_MULT_EV.get(cls, 1),
        reverse=True
    )
    bonus = [cls for cls in ev_ranked if cls not in top2]
    if not bonus:
        return None
    return bonus[:2]

# ── SIM REBUILD ───────────────────────────────────────────────────────────────
def _run_sim(rewards):
    total = len(rewards)
    with _lock:
        train_rounds = _dynamic_train_rounds
    if total < train_rounds + 5:
        return {}, 0, 0.0, [], [], {}
    stats = build_global_stats(rewards)
    hits=misses=sk=pl=0; ls=mx=ws=mw=0; brake=0; se=st=sb=0
    sim_play_results = []
    sim_loss_confs   = []
    sim_markov_hits  = {1:[], 2:[], 3:[], 4:[]}
    sim_boost_log    = []
    sim_bonus_log    = []
    with _lock:
        cur_trigger = _dynamic_brake_trigger
        cur_pause   = _dynamic_brake_pause
        cur_top1_t  = _top1_threshold
    for i in range(train_rounds, total - 1):
        h  = rewards[:i+1]
        tn = rewards[i+1]
        top2, sc, ent, t1s, _, _, _, markov_preds = score_round(h, *stats)
        regime = detect_regime(h)
        ev, ea = compute_entropy_trend(h)
        miss_sup = compute_miss_suppression(ent, t1s, regime)
        ba = brake > 0
        if brake > 0: brake -= 1
        play = should_play(t1s, ent, ba, regime, ev, miss_sup)
        if not play:
            sk += 1
            if ba: sb += 1
            elif ent >= _entropy_threshold: se += 1
            else: st += 1
            continue
        pl += 1
        sim_bonus = get_bonus_picks(sc, top2)
        all_sim_preds = list(top2)
        if sim_bonus:
            all_sim_preds += [p for p in sim_bonus if p not in all_sim_preds]
        hit = tn in all_sim_preds
        sim_play_results.append(hit)
        for order, pred_cls in markov_preds.items():
            sim_markov_hits[order].append(pred_cls == tn)
        with _lock:
            pat_info = dict(_last_pattern_info)
        any_triggered = pat_info.get("_any_triggered", False)
        sim_boost_log.append((any_triggered, hit))
        bonus_triggered = sim_bonus is not None
        sim_bonus_log.append((bonus_triggered, hit))

        # IMPROVEMENT 5: Record miss context
        if not hit:
            record_miss_context({
                "entropy": ent, "top1": t1s, "regime": regime,
                "ev": ev, "miss_sup": miss_sup,
            })
        # IMPROVEMENT 7: Record calibration
        record_calibration(t1s, hit)

        if hit:
            hits+=1; ls=0; ws+=1; mw=max(mw,ws)
        else:
            misses+=1; ls+=1; mx=max(mx,ls); ws=0
            sim_loss_confs.append(t1s)
        if ls >= cur_trigger:
            brake = cur_pause
    sim_total = total - train_rounds - 1
    acc       = hits / pl * 100 if pl else 0
    play_pct  = pl / sim_total if sim_total else 0.0
    return {
        "total":total,"sim_total":sim_total,
        "played":pl,"skipped":sk,"hits":hits,"misses":misses,
        "accuracy":round(acc,2),"play_pct":round(play_pct*100,1),
        "max_loss":mx,"max_win":mw,
        "skip_brake":sb,"skip_entropy":se,"skip_top1":st,
        "brake_trigger": cur_trigger,
        "brake_pause":   cur_pause,
        "train_rounds":  train_rounds,
        "top1_threshold": round(cur_top1_t, 4),
    }, brake, play_pct, sim_play_results, sim_loss_confs, {
        "markov_hits": sim_markov_hits,
        "boost_log":   sim_boost_log,
        "bonus_log":   sim_bonus_log,
    }

# ── BUILD CACHED PRED ─────────────────────────────────────────────────────────
def _build_cached_pred(rewards, raw_rounds, brake):
    global _current_regime, _entropy_velocity, _entropy_acceleration
    with _lock:
        train_rounds  = _dynamic_train_rounds
        cur_top1_t    = _top1_threshold
        cur_trigger   = _dynamic_brake_trigger
        cur_pause     = _dynamic_brake_pause
        boost_max     = _dynamic_boost_max
        lookback      = _dynamic_lookback
        decay         = _dynamic_decay
        bonus_thresh  = _dynamic_bonus_thresh
        dw            = dict(_dynamic_markov_w)
    if len(rewards) < train_rounds + 5:
        return None
    stats = build_global_stats(rewards)
    top2, scores, ent, t1s, t2s, t3s, t3c, _ = score_round(rewards, *stats)

    # Compute all enhancement signals
    regime       = detect_regime(rewards)
    ev, ea       = compute_entropy_trend(rewards)
    miss_sup     = compute_miss_suppression(ent, t1s, regime)
    momentum     = compute_class_momentum(rewards)
    cycle_boosts = compute_cycle_scores(rewards)

    # Update global state
    _current_regime        = regime
    _entropy_velocity      = ev
    _entropy_acceleration  = ea

    eth  = _entropy_threshold
    play = should_play(t1s, ent, brake == 0 and False or brake > 0, regime, ev, miss_sup)
    # Re-check properly
    play = t1s > cur_top1_t and not (brake > 0)
    # Use enhanced should_play
    play = should_play(t1s, ent, brake > 0, regime, ev, miss_sup)

    skip_reason = None
    if brake > 0:              skip_reason = f"Loss brake ({brake} rounds left)"
    elif not play and ev > 0.2: skip_reason = f"Rising entropy velocity ({ev:.3f})"
    elif not play and ent >= eth + REGIME_ENTROPY_ADJUST.get(regime, 0.0):
        skip_reason = f"High entropy ({ent:.4f})"
    elif not play:             skip_reason = f"Low confidence ({t1s:.4f} ≤ {cur_top1_t:.4f})"

    last_round = raw_rounds[-1]["round"] if raw_rounds else None
    next_round = (last_round + 1) if last_round else None

    SMALL_MULT  = {1, 5, 6, 8}
    pred3       = None
    pred3_conf  = None
    top2_are_small = all(c in SMALL_MULT for c in top2)
    if (play and top2_are_small and t3c is not None and (t2s - t3s) <= 0.01):
        pred3      = t3c
        pred3_conf = round(t3s * 100, 2)

    bonus_picks   = get_bonus_picks(scores, top2)
    bonus_details = None
    if bonus_picks:
        sc_map = {int(k): v for k, v in scores.items()}
        bonus_details = []
        for b in bonus_picks:
            prob = sc_map.get(b, 0.0)
            mult = HIGH_MULT_EV.get(b, 1)
            ev_b = round(prob * mult, 3)
            ev_triggered = ev_b >= HIGH_MULT_EV_TARGET
            bonus_details.append({
                "idx": b, "name": CLASS_NAMES.get(b,"?"),
                "color": CLASS_COLORS.get(b,"#888"),
                "conf": round(prob*100, 2),
                "ev": round(ev_b, 3),
                "ev_triggered": ev_triggered,
            })

    with _lock:
        pattern_snap    = dict(_last_pattern_info)
        dyn_thresh_snap = dict(_dynamic_threshold)
    pattern_summary = {}
    for g, info in pattern_snap.items():
        if g in ("raw_scores", "_any_triggered"):
            continue
        if isinstance(info, dict):
            pattern_summary[g] = {
                "score":         info.get("score", 0.0),
                "threshold":     info.get("threshold", 0.0),
                "triggered":     info.get("triggered", False),
                "boost_applied": info.get("boost_applied", 0.0),
            }

    with _lock:
        chb_snap = {}
        for cls, buf in _class_hit_buf.items():
            w = buf[-CLASS_HIT_BUF_MAX:]
            chb_snap[cls] = {
                "samples":  len(w),
                "hit_rate": round(sum(w)/len(w), 4) if w else None,
            }

    # Calibrated confidence
    calib_t1 = get_calibrated_conf(t1s)

    return {
        "next_round":         next_round,   "latest_round":  last_round,
        "pred1":              top2[0],      "pred2":         top2[1],
        "pred3":              pred3,
        "pred1_name":         CLASS_NAMES.get(top2[0],"?"),
        "pred2_name":         CLASS_NAMES.get(top2[1],"?"),
        "pred3_name":         CLASS_NAMES.get(pred3,"?") if pred3 else None,
        "pred1_color":        CLASS_COLORS.get(top2[0],"#888"),
        "pred2_color":        CLASS_COLORS.get(top2[1],"#888"),
        "pred3_color":        CLASS_COLORS.get(pred3,"#888") if pred3 else None,
        "pred1_conf":         round(t1s*100,2),
        "pred2_conf":         round(t2s*100,2),
        "pred3_conf":         pred3_conf,
        "pred1_conf_calib":   round(calib_t1*100, 2),   # NEW: calibrated
        "entropy":            round(ent,4),
        "action":             "PLAY" if play else "SKIP",
        "skip_reason":        skip_reason,
        "bonus_picks":        bonus_details,
        "skip_show_bonus_only": (not play) and (bonus_details is not None),
        "all_scores":         {k: round(v*100,2) for k,v in scores.items()},
        "last_10":            rewards[-10:],
        "total_rounds":       len(rewards),
        "pattern_info":       pattern_summary,
        "dynamic_thresholds": {g: round(v,4) for g,v in dyn_thresh_snap.items()},
        "brake_trigger":      cur_trigger,
        "brake_pause":        cur_pause,
        "top1_threshold":     round(cur_top1_t, 4),
        "entropy_threshold":  round(eth, 4),
        "train_rounds":       train_rounds,
        "pattern_lookback":   lookback,
        "pattern_decay":      round(decay, 4),
        "pattern_boost_max":  round(boost_max, 4),
        "bonus_conf_thresh":  round(bonus_thresh, 4),
        "markov_weights":     {k: round(v,4) for k,v in dw.items()},
        "class_hit_rates":    chb_snap,
        # NEW enhancement fields
        "regime":             regime,
        "entropy_velocity":   ev,
        "entropy_acceleration": ea,
        "miss_suppression":   miss_sup,
        "momentum":           {str(k): round(v,4) for k,v in momentum.items()},
        "cycle_boosts":       {str(k): round(v,4) for k,v in cycle_boosts.items()},
        "_play":              play,
        "_top2":              list(top2),
        "_top3":              ([top2[0], top2[1], pred3] if pred3 else list(top2)),
        "_bonus_picks":       list(bonus_picks) if bonus_picks else None,
        "_bonus_triggered":   bonus_picks is not None,
        "_next_round":        next_round,
        "_pattern_scores":    {g: pattern_snap.get(g, {}).get("score", 0.0)
                               for g in PATTERN_GROUPS},
        "_any_boosted":       pattern_snap.get("_any_triggered", False),
        "_t1s":               t1s,
    }

# ── FETCHER LOOP ──────────────────────────────────────────────────────────────
def fetcher_loop():
    global _pending_pred, _cached_pred, _entropy_threshold, _brake_left
    global _dynamic_brake_trigger, _dynamic_brake_pause
    global _dynamic_train_rounds
    global _dynamic_boost_max, _dynamic_boost_min
    global _dynamic_bonus_thresh
    global _skip_pending_pred

    while True:
        try:
            if _should_reset():
                _do_reset()
                time.sleep(POLL_INTERVAL)
                continue
            with _lock:
                _fetch_status["last_attempt"] = time.strftime("%H:%M:%S")
                _fetch_status["status"]       = "fetching"
            count, records = fetch_new()
            with _lock:
                if count > 0:
                    _fetch_status["last_success"]  = time.strftime("%H:%M:%S")
                    _fetch_status["total_fetched"] += count
                    _fetch_status["last_error"]    = None
                    _fetch_status["status"]        = "ok"
                else:
                    if _fetch_status.get("status") != "error":
                        _fetch_status["status"] = "ok_no_new"
            if count > 0:
                rewards = [r["reward_index"] for r in records]
                with _lock:
                    _raw_rounds.clear(); _raw_rounds.extend(records)
                    _rewards.clear();    _rewards.extend(rewards)
                    pending      = _pending_pred
                    skip_pending = _skip_pending_pred

                # ── Resolve PLAY pending ──────────────────────────────────────
                if pending is not None:
                    pred_round = pending["round"]
                    actual_rec = next((r for r in records if r["round"] == pred_round), None)
                    if actual_rec is not None:
                        actual_val = actual_rec["reward_index"]
                        all_preds  = list(pending.get("top3", pending["top2"]))
                        if pending.get("bonus_picks"):
                            all_preds += [p for p in pending["bonus_picks"] if p not in all_preds]
                        hit = actual_val in all_preds

                        _record_class_hits(all_preds, actual_val, source="play")

                        # IMPROVEMENT 7: Record calibration from live play
                        record_calibration(pending.get("t1s", 0.25), hit)

                        # IMPROVEMENT 5: Record miss context
                        if not hit:
                            record_miss_context({
                                "entropy":  pending.get("entropy", 3.0),
                                "top1":     pending.get("t1s", 0.25),
                                "regime":   pending.get("regime", "stable"),
                                "ev":       pending.get("ev", 0.0),
                                "miss_sup": pending.get("miss_sup", 1.0),
                            })

                        if hit and pending.get("pattern_scores"):
                            for group_name, pscore in pending["pattern_scores"].items():
                                update_pattern_hit(group_name, pscore)
                        with _lock:
                            _brake_play_results.append(hit)
                            if len(_brake_play_results) > BRAKE_HITRATE_WINDOW * 2:
                                _brake_play_results.pop(0)
                            if not hit:
                                _brake_loss_confs.append(pending.get("t1s", 0.25))
                                if len(_brake_loss_confs) > BRAKE_CONF_BUFFER * 2:
                                    _brake_loss_confs.pop(0)
                            play_results_snap = list(_brake_play_results)
                            loss_confs_snap   = list(_brake_loss_confs)
                        new_trigger, new_pause, _, _ = recalibrate_brake(
                            play_results_snap, loss_confs_snap)
                        with _lock:
                            _dynamic_brake_trigger = new_trigger
                            _dynamic_brake_pause   = new_pause
                        with _lock:
                            _boost_eval_log.append((pending.get("any_boosted", False), hit))
                            if len(_boost_eval_log) > PATTERN_BOOST_EVAL_WINDOW * 3:
                                _boost_eval_log.pop(0)
                            boost_log_snap = list(_boost_eval_log)
                        recalibrate_boost_cap(boost_log_snap)
                        with _lock:
                            _bonus_eval_log.append((pending.get("bonus_triggered", False), hit))
                            if len(_bonus_eval_log) > BONUS_EVAL_WINDOW * 3:
                                _bonus_eval_log.pop(0)
                            bonus_log_snap = list(_bonus_eval_log)
                        recalibrate_bonus_thresh(bonus_log_snap)
                        markov_preds_pending = pending.get("markov_preds", {})
                        with _lock:
                            for order, pred_cls in markov_preds_pending.items():
                                _markov_hit_buf[order].append(pred_cls == actual_val)
                                if len(_markov_hit_buf[order]) > MARKOV_WEIGHT_BUFFER * 3:
                                    _markov_hit_buf[order].pop(0)
                            markov_buf_snap = {o: list(v) for o, v in _markov_hit_buf.items()}
                        new_mw = recalibrate_markov_weights(markov_buf_snap)
                        with _lock:
                            _dynamic_markov_w.update(new_mw)
                        entry = {
                            "round": pred_round, "top2": pending["top2"],
                            "top3":  pending.get("top3"),
                            "bonus_picks": pending.get("bonus_picks"),
                            "actual": actual_val, "hit": hit, "action": "PLAY",
                        }
                        with _lock:
                            _live_log.append(entry)
                            _pending_pred = None
                        pending = None
                        print(f"[Live] #{pred_round}: pred={entry['top2']} "
                              f"actual={actual_val} → {'HIT ✓' if hit else 'MISS ✗'}")
                    elif records and pred_round <= records[-1]["round"]:
                        with _lock:
                            _pending_pred = None
                        pending = None
                        print(f"[Fetcher] Stale pending cleared (#{pred_round})")

                # ── Resolve SKIP pending ──────────────────────────────────────
                if skip_pending is not None:
                    skip_round = skip_pending["round"]
                    skip_rec   = next((r for r in records if r["round"] == skip_round), None)
                    if skip_rec is not None:
                        actual_val  = skip_rec["reward_index"]
                        skip_preds  = list(skip_pending.get("top3", skip_pending["top2"]))
                        if skip_pending.get("bonus_picks"):
                            skip_preds += [p for p in skip_pending["bonus_picks"]
                                           if p not in skip_preds]
                        _record_class_hits(skip_preds, actual_val, source="skip")
                        with _lock:
                            _skip_pending_pred = None
                        skip_pending = None
                    elif records and skip_round <= records[-1]["round"]:
                        with _lock:
                            _skip_pending_pred = None
                        skip_pending = None

                sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = \
                    _run_sim(rewards)
                total_rounds = len(rewards)
                sim_hit_rate = (sim_dict.get("hits", 0) / sim_dict.get("played", 1)
                                if sim_dict.get("played", 0) > 0 else 0.60)
                new_tr = compute_dynamic_train_rounds(total_rounds)
                if new_tr != _dynamic_train_rounds:
                    print(f"[TrainRounds] {_dynamic_train_rounds}→{new_tr}")
                    _dynamic_train_rounds = new_tr
                cur = _entropy_threshold
                if play_pct < TARGET_PLAY_MIN:   new_eth = min(ENT_THRESH_MAX, cur + 0.034)
                elif play_pct > TARGET_PLAY_MAX: new_eth = max(ENT_THRESH_MIN, cur - 0.034)
                else:                            new_eth = cur
                if abs(new_eth - cur) > 0.001:
                    print(f"[Adaptive] Entropy: {cur:.3f}→{new_eth:.3f} (play%={play_pct*100:.1f}%)")
                _entropy_threshold = new_eth
                sim_dict["entropy_threshold"] = round(new_eth, 3)
                recalibrate_top1_threshold(play_pct)
                recalibrate_pattern_params(total_rounds, sim_hit_rate)
                with _lock:
                    live_boost_len = len(_boost_eval_log)
                if live_boost_len < 20 and sim_extras.get("boost_log"):
                    recalibrate_boost_cap(sim_extras["boost_log"])
                with _lock:
                    live_bonus_len = len(_bonus_eval_log)
                if live_bonus_len < 15 and sim_extras.get("bonus_log"):
                    recalibrate_bonus_thresh(sim_extras["bonus_log"])
                with _lock:
                    live_markov_len = min(len(v) for v in _markov_hit_buf.values())
                if live_markov_len < 10 and sim_extras.get("markov_hits"):
                    new_mw = recalibrate_markov_weights(sim_extras["markov_hits"])
                    with _lock:
                        _dynamic_markov_w.update(new_mw)
                        if not any(_markov_hit_buf.values()):
                            for o, v in sim_extras["markov_hits"].items():
                                _markov_hit_buf[o].extend(v[-MARKOV_WEIGHT_BUFFER:])
                with _lock:
                    live_buf_len = len(_brake_play_results)
                if live_buf_len < 10 and sim_play_res:
                    seed_results = sim_play_res[-BRAKE_HITRATE_WINDOW:]
                    seed_confs   = sim_loss_confs[-BRAKE_CONF_BUFFER:]
                    new_trigger, new_pause, _, _ = recalibrate_brake(seed_results, seed_confs)
                    with _lock:
                        _dynamic_brake_trigger = new_trigger
                        _dynamic_brake_pause   = new_pause
                        if not _brake_play_results:
                            _brake_play_results.extend(seed_results)
                        if not _brake_loss_confs:
                            _brake_loss_confs.extend(seed_confs)
                    print(f"[DynBrake] Seeded: trigger={new_trigger} pause={new_pause}")
                with _lock:
                    _sim_stats.clear(); _sim_stats.update(sim_dict)
                    _brake_left = brake
                cached = _build_cached_pred(rewards, records, brake)
                with _lock:
                    _cached_pred = cached
                if cached and cached["_next_round"] is not None:
                    nr = cached["_next_round"]
                    if cached["_play"]:
                        with _lock:
                            cur_p = _pending_pred
                        if cur_p is None or cur_p["round"] != nr:
                            stats_now = build_global_stats(rewards)
                            _, _, _, _, _, _, _, mp = score_round(rewards, *stats_now)
                            np_ = {
                                "round":           nr,
                                "top2":            cached["_top2"],
                                "top3":            cached["_top3"],
                                "bonus_picks":     cached["_bonus_picks"],
                                "bonus_triggered": cached.get("_bonus_triggered", False),
                                "pattern_scores":  cached.get("_pattern_scores", {}),
                                "any_boosted":     cached.get("_any_boosted", False),
                                "t1s":             cached.get("_t1s", 0.25),
                                # NEW: pass enhancement context for miss learning
                                "regime":          cached.get("regime", "stable"),
                                "ev":              cached.get("entropy_velocity", 0.0),
                                "miss_sup":        cached.get("miss_suppression", 1.0),
                                "markov_preds":    mp,
                            }
                            with _lock:
                                _pending_pred = np_
                            print(f"[Fetcher] Pending → #{nr} top2={cached['_top2']} "
                                  f"regime={cached.get('regime','?')} "
                                  f"ev={cached.get('entropy_velocity',0):.3f}")
                    else:
                        with _lock:
                            cur_sp = _skip_pending_pred
                        if cur_sp is None or cur_sp["round"] != nr:
                            sp_ = {
                                "round":       nr,
                                "top2":        cached["_top2"],
                                "top3":        cached["_top3"],
                                "bonus_picks": cached["_bonus_picks"],
                            }
                            with _lock:
                                _skip_pending_pred = sp_
                            print(f"[Fetcher] SkipPending → #{nr} top2={cached['_top2']}")

                with _lock:
                    lr = _raw_rounds[-1]["round"] if _raw_rounds else "?"
                print(f"[Fetcher] +{count} rounds. Latest: #{lr}. Total: {len(records)}")
        except Exception as e:
            print(f"[Fetcher] Unexpected: {e}")
            import traceback; traceback.print_exc()
            with _lock:
                _fetch_status["last_error"] = str(e)
                _fetch_status["status"]     = "error"
        time.sleep(POLL_INTERVAL)

# ══════════════════════════════════════════════════════════════════════════════
# PWA ROUTES
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/pwa/sw.js")
def serve_sw():
    sw_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sw.js")
    with open(sw_path, "r") as f: content = f.read()
    return Response(content, mimetype="application/javascript",
                    headers={"Service-Worker-Allowed": "/",
                             "Cache-Control": "no-cache, no-store, must-revalidate"})

@app.route("/pwa/manifest.json")
def serve_manifest():
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    with open(manifest_path, "r") as f: content = f.read()
    return Response(content, mimetype="application/manifest+json",
                    headers={"Cache-Control": "no-cache"})

# ── API ROUTES ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES, class_colors=CLASS_COLORS)

@app.route("/api/predict")
def api_predict():
    with _lock:
        cached = _cached_pred
    if cached is None:
        return jsonify({"error": "Not enough data yet"}), 503
    pub = {k: v for k, v in cached.items() if not k.startswith("_")}
    return jsonify(pub)

@app.route("/api/stats")
def api_stats():
    with _lock:
        stats     = dict(_sim_stats)
        live      = list(_live_log)
        dyn_t     = dict(_dynamic_threshold)
        hit_bufs  = {g: len(v) for g, v in _hit_pattern_scores.items()}
        dyn_bt    = _dynamic_brake_trigger
        dyn_bp    = _dynamic_brake_pause
        play_res  = list(_brake_play_results)
        loss_c    = list(_brake_loss_confs)
        dyn_tr    = _dynamic_train_rounds
        top1_t    = _top1_threshold
        dyn_mw    = dict(_dynamic_markov_w)
        dyn_bt2   = _dynamic_bonus_thresh
        dyn_bmax  = _dynamic_boost_max
        dyn_lk    = _dynamic_lookback
        dyn_dc    = _dynamic_decay
        chb_summary = {}
        for cls, buf in _class_hit_buf.items():
            w = buf[-CLASS_HIT_BUF_MAX:]
            chb_summary[cls] = {
                "samples":  len(w),
                "hit_rate": round(sum(w)/len(w), 4) if w else None,
                "name":     CLASS_NAMES.get(cls, "?"),
            }
        cur_regime  = _current_regime
        ent_vel     = _entropy_velocity
        ent_acc     = _entropy_acceleration
        miss_ctx_n  = len(_miss_contexts)
        calib_n     = len(_calib_buffer)
    cur = 0; mx_live = 0
    for e in reversed(live):
        if not e["hit"]: cur += 1; mx_live = max(mx_live, cur)
        else: break
    stats["live_log"]            = live
    stats["live_cur_streak"]     = cur
    stats["live_max_loss"]       = mx_live
    stats["dynamic_thresholds"]  = {g: round(v,4) for g,v in dyn_t.items()}
    stats["pattern_hit_counts"]  = hit_bufs
    stats["brake_trigger"]       = dyn_bt
    stats["brake_pause"]         = dyn_bp
    stats["brake_hitrate"]       = (round(sum(play_res[-BRAKE_HITRATE_WINDOW:]) /
                                          len(play_res[-BRAKE_HITRATE_WINDOW:]), 4)
                                    if len(play_res) >= 5 else None)
    stats["brake_avg_loss_conf"] = (round(sum(loss_c[-BRAKE_CONF_BUFFER:]) /
                                          len(loss_c[-BRAKE_CONF_BUFFER:]), 4)
                                    if len(loss_c) >= 3 else None)
    stats["brake_play_buf_len"]  = len(play_res)
    stats["brake_loss_buf_len"]  = len(loss_c)
    stats["train_rounds"]        = dyn_tr
    stats["top1_threshold"]      = round(top1_t, 4)
    stats["markov_weights"]      = {k: round(v,4) for k,v in dyn_mw.items()}
    stats["bonus_conf_thresh"]   = round(dyn_bt2, 4)
    stats["pattern_boost_max"]   = round(dyn_bmax, 4)
    stats["pattern_lookback"]    = dyn_lk
    stats["pattern_decay"]       = round(dyn_dc, 4)
    stats["class_hit_rates"]     = chb_summary
    # NEW enhancement stats
    stats["regime"]              = cur_regime
    stats["entropy_velocity"]    = ent_vel
    stats["entropy_acceleration"]= ent_acc
    stats["miss_context_count"]  = miss_ctx_n
    stats["calib_buffer_count"]  = calib_n
    return jsonify(stats)

@app.route("/api/status")
def api_status():
    with _lock:
        fs     = dict(_fetch_status)
        total  = len(_rewards)
        latest = _raw_rounds[-1]["round"] if _raw_rounds else None
        dyn_bt = _dynamic_brake_trigger
        dyn_bp = _dynamic_brake_pause
        top1_t = _top1_threshold
        dyn_tr = _dynamic_train_rounds
    now_ist = datetime.now(IST)
    fs["total_rounds"]      = total
    fs["latest_round"]      = latest
    fs["server_time_ist"]   = now_ist.strftime("%H:%M:%S IST")
    fs["entropy_threshold"] = round(_entropy_threshold, 3)
    fs["top1_threshold"]    = round(top1_t, 4)
    fs["brake_trigger"]     = dyn_bt
    fs["brake_pause"]       = dyn_bp
    fs["train_rounds"]      = dyn_tr
    fs["regime"]            = _current_regime
    fs["entropy_velocity"]  = _entropy_velocity
    reset_today = now_ist.replace(hour=RESET_HOUR_IST, minute=RESET_MINUTE_IST,
                                  second=0, microsecond=0)
    from datetime import timedelta
    if now_ist >= reset_today:
        reset_today += timedelta(days=1)
    secs = (reset_today - now_ist).seconds
    fs["next_reset_in"] = f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
    return jsonify(fs)

@app.route("/api/history")
def api_history():
    with _lock:
        raw = list(_raw_rounds[-50:])
    return jsonify([{
        "round": r["round"], "reward_index": r["reward_index"],
        "name":  CLASS_NAMES.get(r["reward_index"],"?"),
        "color": CLASS_COLORS.get(r["reward_index"],"#888"),
    } for r in reversed(raw)])

@app.route("/api/pattern")
def api_pattern():
    with _lock:
        pattern_info = dict(_last_pattern_info)
        dyn_t        = dict(_dynamic_threshold)
        hit_bufs     = {g: list(v) for g, v in _hit_pattern_scores.items()}
        lookback     = _dynamic_lookback
        decay        = _dynamic_decay
        boost_max    = _dynamic_boost_max
    result = {}
    for g in PATTERN_GROUPS:
        info = pattern_info.get(g, {})
        buf  = hit_bufs.get(g, [])
        result[g] = {
            "score":          info.get("score", 0.0) if isinstance(info, dict) else 0.0,
            "threshold":      round(dyn_t.get(g, PATTERN_GROUPS[g]["default_threshold"]), 4),
            "triggered":      info.get("triggered", False) if isinstance(info, dict) else False,
            "boost_applied":  info.get("boost_applied", 0.0) if isinstance(info, dict) else 0.0,
            "hit_buffer_len": len(buf),
            "hit_avg":        round(sum(buf)/len(buf), 4) if buf else None,
            "classes":        list(PATTERN_GROUPS[g]["classes"]),
        }
    result["_params"] = {
        "lookback":  lookback,
        "decay":     round(decay, 4),
        "boost_max": round(boost_max, 4),
    }
    return jsonify(result)

@app.route("/api/brake")
def api_brake():
    with _lock:
        dyn_bt   = _dynamic_brake_trigger
        dyn_bp   = _dynamic_brake_pause
        bl       = _brake_left
        play_res = list(_brake_play_results)
        loss_c   = list(_brake_loss_confs)
    window  = play_res[-BRAKE_HITRATE_WINDOW:]
    hitrate = round(sum(window)/len(window), 4) if window else None
    avg_lc  = round(sum(loss_c)/len(loss_c), 4) if loss_c else None
    return jsonify({
        "brake_trigger":          dyn_bt,
        "brake_pause":            dyn_bp,
        "brake_left":             bl,
        "trigger_range":          [BRAKE_TRIGGER_MIN, BRAKE_TRIGGER_MAX],
        "pause_range":            [BRAKE_PAUSE_MIN,   BRAKE_PAUSE_MAX],
        "hitrate_window":         len(window),
        "hitrate":                hitrate,
        "hitrate_low_threshold":  BRAKE_HITRATE_LOW,
        "hitrate_high_threshold": BRAKE_HITRATE_HIGH,
        "loss_conf_buffer_len":   len(loss_c),
        "avg_loss_conf":          avg_lc,
        "recent_loss_confs":      loss_c[-10:],
        "conf_high_threshold":    BRAKE_CONF_HIGH,
        "conf_low_threshold":     BRAKE_CONF_LOW,
    })

@app.route("/api/adaptive")
def api_adaptive():
    with _lock:
        return jsonify({
            "entropy_threshold":   {"value": round(_entropy_threshold, 4),
                                    "min": ENT_THRESH_MIN, "max": ENT_THRESH_MAX},
            "top1_threshold":      {"value": round(_top1_threshold, 4),
                                    "min": TOP1_THRESH_MIN, "max": TOP1_THRESH_MAX},
            "brake_trigger":       {"value": _dynamic_brake_trigger,
                                    "min": BRAKE_TRIGGER_MIN, "max": BRAKE_TRIGGER_MAX},
            "brake_pause":         {"value": _dynamic_brake_pause,
                                    "min": BRAKE_PAUSE_MIN, "max": BRAKE_PAUSE_MAX},
            "train_rounds":        {"value": _dynamic_train_rounds,
                                    "min": TRAIN_ROUNDS_MIN, "max": TRAIN_ROUNDS_MAX},
            "pattern_lookback":    {"value": _dynamic_lookback,
                                    "min": PATTERN_LOOKBACK_MIN, "max": PATTERN_LOOKBACK_MAX},
            "pattern_decay":       {"value": round(_dynamic_decay, 4),
                                    "min": PATTERN_DECAY_MIN, "max": PATTERN_DECAY_MAX},
            "pattern_boost_max":   {"value": round(_dynamic_boost_max, 4),
                                    "min": PATTERN_BOOST_MIN_DEFAULT, "max": 0.50},
            "bonus_conf_thresh":   {"value": round(_dynamic_bonus_thresh, 4),
                                    "min": BONUS_CONF_THRESH_MIN, "max": BONUS_CONF_THRESH_MAX},
            "markov_weights":      {k: round(v,4) for k,v in _dynamic_markov_w.items()},
            "markov_weight_range": {"min": MARKOV_WEIGHT_MIN, "max": MARKOV_WEIGHT_MAX},
            "play_target":         {"min": TARGET_PLAY_MIN, "max": TARGET_PLAY_MAX},
            "ev_thresholds":       {str(cls): {"multiplier": mult,
                                               "min_prob": round(HIGH_MULT_EV_TARGET/mult, 4)}
                                    for cls, mult in HIGH_MULT_EV.items()},
            "class_hit_buf_max":   CLASS_HIT_BUF_MAX,
            "class_hit_min_samples": CLASS_HIT_MIN_SAMPLES,
            "class_hit_boost_scale": CLASS_HIT_BOOST_SCALE,
            # NEW enhancement config
            "regime_detection":    {
                "window": REGIME_WINDOW,
                "current": _current_regime,
                "entropy_adjustments": REGIME_ENTROPY_ADJUST,
            },
            "entropy_trend":       {
                "window": ENTROPY_TREND_WINDOW,
                "velocity": _entropy_velocity,
                "acceleration": _entropy_acceleration,
            },
            "momentum":            {"short": MOMENTUM_SHORT, "long": MOMENTUM_LONG},
            "anti_pattern":        {"window": ANTI_PATTERN_WINDOW,
                                    "break_thresh": ANTI_BREAK_THRESH},
            "cycle_periods":       CYCLE_PERIODS,
            "calibration_samples": len(_calib_buffer),
            "miss_contexts":       len(_miss_contexts),
            "dead_class":          {"window": DEAD_CLASS_WINDOW,
                                    "miss_thresh": DEAD_CLASS_MISS_THRESH,
                                    "penalty": DEAD_CLASS_PENALTY},
        })

# NEW: regime debug endpoint
@app.route("/api/regime")
def api_regime():
    with _lock:
        rewards = list(_rewards)
    if not rewards:
        return jsonify({"error": "No data"}), 503
    regime      = detect_regime(rewards)
    ev, ea      = compute_entropy_trend(rewards)
    momentum    = compute_class_momentum(rewards)
    cycle       = compute_cycle_scores(rewards)
    anti        = compute_anti_pattern_penalties(rewards)
    with _lock:
        chb_snap = {cls: list(buf[-DEAD_CLASS_WINDOW:]) for cls, buf in _class_hit_buf.items()}
    dead_pen = compute_dead_class_penalties(rewards, {cls: list(buf) for cls, buf in _class_hit_buf.items()})
    return jsonify({
        "regime":           regime,
        "entropy_velocity": ev,
        "entropy_accel":    ea,
        "momentum":         {str(k): v for k, v in momentum.items()},
        "cycle_boosts":     {str(k): v for k, v in cycle.items()},
        "anti_penalties":   {str(k): v for k, v in anti.items()},
        "dead_class_penalties": {str(k): v for k, v in dead_pen.items()},
        "miss_contexts":    len(_miss_contexts),
        "calib_samples":    len(_calib_buffer),
    })

# ── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    global _last_reset_date, _pending_pred, _cached_pred, _entropy_threshold, _brake_left
    global _dynamic_brake_trigger, _dynamic_brake_pause, _dynamic_train_rounds
    global _skip_pending_pred, _current_regime, _entropy_velocity, _entropy_acceleration

    _skip_pending_pred    = None
    _current_regime       = "stable"
    _entropy_velocity     = 0.0
    _entropy_acceleration = 0.0

    now_ist  = datetime.now(IST)
    past_530 = (now_ist.hour > RESET_HOUR_IST or
                (now_ist.hour == RESET_HOUR_IST and now_ist.minute >= RESET_MINUTE_IST))
    if past_530:
        _last_reset_date = now_ist.date()

    records = _load_file()
    if records:
        rewards = [r["reward_index"] for r in records]
        with _lock:
            _raw_rounds.extend(records)
            _rewards.extend(rewards)
        total_rounds = len(records)
        _dynamic_train_rounds = compute_dynamic_train_rounds(total_rounds)

        # Warm up enhancement systems
        _current_regime       = detect_regime(rewards)
        _entropy_velocity, _entropy_acceleration = compute_entropy_trend(rewards)

        print(f"[Startup] Loaded {total_rounds} rounds. "
              f"train_rounds={_dynamic_train_rounds} "
              f"regime={_current_regime} ev={_entropy_velocity:.3f}")
        sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = \
            _run_sim(rewards)
        sim_hit_rate = (sim_dict.get("hits", 0) / sim_dict.get("played", 1)
                        if sim_dict.get("played", 0) > 0 else 0.60)
        recalibrate_pattern_params(total_rounds, sim_hit_rate)
        recalibrate_top1_threshold(play_pct)
        if sim_extras.get("boost_log"):
            recalibrate_boost_cap(sim_extras["boost_log"])
        if sim_extras.get("bonus_log"):
            recalibrate_bonus_thresh(sim_extras["bonus_log"])
        if sim_extras.get("markov_hits"):
            new_mw = recalibrate_markov_weights(sim_extras["markov_hits"])
            with _lock:
                _dynamic_markov_w.update(new_mw)
                for o, v in sim_extras["markov_hits"].items():
                    _markov_hit_buf[o].extend(v[-MARKOV_WEIGHT_BUFFER:])
        if sim_play_res:
            seed_results = sim_play_res[-BRAKE_HITRATE_WINDOW:]
            seed_confs   = sim_loss_confs[-BRAKE_CONF_BUFFER:]
            new_trigger, new_pause, hr, ac = recalibrate_brake(seed_results, seed_confs)
            _dynamic_brake_trigger = new_trigger
            _dynamic_brake_pause   = new_pause
            with _lock:
                _brake_play_results.extend(seed_results)
                _brake_loss_confs.extend(seed_confs)
            print(f"[Startup] DynBrake: trigger={new_trigger} pause={new_pause}")
        cur = _entropy_threshold
        if play_pct < TARGET_PLAY_MIN:   new_eth = min(ENT_THRESH_MAX, cur + 0.034)
        elif play_pct > TARGET_PLAY_MAX: new_eth = max(ENT_THRESH_MIN, cur - 0.034)
        else:                            new_eth = cur
        _entropy_threshold = new_eth
        sim_dict["entropy_threshold"] = round(new_eth, 3)
        with _lock:
            _sim_stats.update(sim_dict)
            _brake_left = brake
        cached = _build_cached_pred(rewards, records, brake)
        with _lock:
            _cached_pred = cached
        if cached and cached["_play"] and cached["_next_round"] is not None:
            stats_now = build_global_stats(rewards)
            _, _, _, _, _, _, _, mp = score_round(rewards, *stats_now)
            _pending_pred = {
                "round":           cached["_next_round"],
                "top2":            cached["_top2"],
                "top3":            cached["_top3"],
                "bonus_picks":     cached["_bonus_picks"],
                "bonus_triggered": cached.get("_bonus_triggered", False),
                "pattern_scores":  cached.get("_pattern_scores", {}),
                "any_boosted":     cached.get("_any_boosted", False),
                "t1s":             cached.get("_t1s", 0.25),
                "regime":          cached.get("regime", "stable"),
                "ev":              cached.get("entropy_velocity", 0.0),
                "miss_sup":        cached.get("miss_suppression", 1.0),
                "markov_preds":    mp,
            }
            print(f"[Startup] Pending → #{_pending_pred['round']} top2={_pending_pred['top2']}")
        elif cached and not cached["_play"] and cached["_next_round"] is not None:
            _skip_pending_pred = {
                "round":       cached["_next_round"],
                "top2":        cached["_top2"],
                "top3":        cached["_top3"],
                "bonus_picks": cached["_bonus_picks"],
            }
            print(f"[Startup] SkipPending → #{_skip_pending_pred['round']} "
                  f"top2={_skip_pending_pred['top2']}")
        action = cached.get("action","N/A") if cached else "N/A"
        print(f"[Startup] Done — acc={sim_dict.get('accuracy')}% "
              f"brake={brake} action={action} "
              f"top1_t={round(_top1_threshold,4)} "
              f"regime={_current_regime} ev={_entropy_velocity:.3f}")
    else:
        print("[Startup] No data file. Waiting for API fetch...")

    threading.Thread(target=fetcher_loop, daemon=True).start()
    print(f"[Startup] Fetcher started. IST: {now_ist.strftime('%H:%M:%S')}. Reset at 05:30 IST.")

startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
