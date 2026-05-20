import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.220      # default (overridden dynamically)
SKIP_ENTROPY_THRESHOLD = 2.70       # default (overridden dynamically)
BRAKE_TRIGGER          = 3          # default (overridden dynamically)
BRAKE_PAUSE            = 3          # default (overridden dynamically)
POLL_INTERVAL          = 5

HIGH_MULT_CLASSES      = {2, 3, 4, 7}

# Entropy adaptation targets
TARGET_PLAY_MIN  = 0.38
TARGET_PLAY_MAX  = 0.45
ENT_THRESH_MIN   = 2.65
ENT_THRESH_MAX   = 2.90

# Top1 threshold adaptation (mirrors entropy adaptation)
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

# Defaults — overridden dynamically at runtime
PATTERN_LOOKBACK_DEFAULT = 10
PATTERN_LOOKBACK_MIN     = 6
PATTERN_LOOKBACK_MAX     = 16

PATTERN_DECAY_DEFAULT    = 0.5
PATTERN_DECAY_MIN        = 0.25
PATTERN_DECAY_MAX        = 0.75

PATTERN_HIT_BUFFER       = 100

# Dynamic boost cap — learned from boosted vs non-boosted hit rates
PATTERN_BOOST_MIN_DEFAULT = 0.15
PATTERN_BOOST_MAX_DEFAULT = 0.35
PATTERN_BOOST_SCALE       = 0.04
PATTERN_BOOST_EVAL_WINDOW = 50     # rounds to measure boosted vs non-boosted
PATTERN_BOOST_STEP        = 0.02   # how fast boost cap moves

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
# Base weights for each Markov order + other signal components
# These are learned per-order hit rates, updated every cycle
MARKOV_WEIGHT_BUFFER = 60   # remember last 60 played rounds per order
MARKOV_WEIGHT_MIN    = 0.02
MARKOV_WEIGHT_MAX    = 0.40
MARKOV_WEIGHT_STEP   = 0.015

# ── DYNAMIC BONUS THRESHOLD CONFIG ───────────────────────────────────────────
BONUS_CONF_THRESH_DEFAULT = 0.10
BONUS_CONF_THRESH_MIN     = 0.06
BONUS_CONF_THRESH_MAX     = 0.18
BONUS_CONF_THRESH_STEP    = 0.005
BONUS_EVAL_WINDOW         = 40    # last N played rounds to evaluate bonus hit rate

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
_lock              = threading.Lock()
_rewards           = []
_raw_rounds        = []
_sim_stats         = {}
_brake_left        = 0
_live_log          = []
_pending_pred      = None
_entropy_threshold = SKIP_ENTROPY_THRESHOLD
_top1_threshold    = SKIP_TOP1_THRESHOLD
_last_reset_date   = None

_cached_pred = None

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

# Boost effectiveness tracking: list of (was_boosted: bool, hit: bool)
_boost_eval_log      = []

# ── DYNAMIC BRAKE STATE ───────────────────────────────────────────────────────
_dynamic_brake_trigger = BRAKE_TRIGGER_DEFAULT
_dynamic_brake_pause   = BRAKE_PAUSE_DEFAULT
_brake_play_results    = []
_brake_loss_confs      = []

# ── DYNAMIC TRAIN_ROUNDS STATE ────────────────────────────────────────────────
_dynamic_train_rounds  = TRAIN_ROUNDS_DEFAULT

# ── DYNAMIC MARKOV WEIGHTS STATE ─────────────────────────────────────────────
# Per-order correctness buffer: order → [True/False, ...]
# order 1=m1, 2=m2, 3=m3, 4=m4
_markov_hit_buf = {1: [], 2: [], 3: [], 4: []}
# Live weights (updated each cycle)
_dynamic_markov_w = {
    "wb":  0.05,   # base frequency
    "wm1": 0.10,   # order-1 markov
    "wm2": 0.15,   # order-2 markov
    "wm3": 0.25,   # order-3 markov
    "wm4": 0.20,   # order-4 markov
    "wr":  0.10,   # recency frequency
    "wv":  0.10,   # short-window frequency
    "wo":  0.05,   # gap/overdue bonus
}

# ── DYNAMIC BONUS THRESHOLD STATE ────────────────────────────────────────────
_dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT
# (was_bonus_play: bool, hit: bool) for recent played rounds
_bonus_eval_log        = []


# ── DAILY RESET ───────────────────────────────────────────────────────────────
def _should_reset():
    global _last_reset_date
    now_ist  = datetime.now(IST)
    today    = now_ist.date()
    past_530 = (now_ist.hour > RESET_HOUR_IST or
                (now_ist.hour == RESET_HOUR_IST and now_ist.minute >= RESET_MINUTE_IST))
    return past_530 and _last_reset_date != today

def _do_reset():
    global _last_reset_date, _pending_pred, _cached_pred
    global _dynamic_brake_trigger, _dynamic_brake_pause
    global _dynamic_train_rounds, _top1_threshold
    global _dynamic_lookback, _dynamic_decay, _dynamic_boost_max, _dynamic_boost_min
    global _dynamic_bonus_thresh
    now_ist = datetime.now(IST)
    print(f"[Reset] 5:30 AM IST — wiping data ({now_ist.date()})")
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    with _lock:
        _rewards.clear(); _raw_rounds.clear()
        _sim_stats.clear(); _live_log.clear()
        _pending_pred = None
        _cached_pred  = None
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
    _dynamic_brake_trigger = BRAKE_TRIGGER_DEFAULT
    _dynamic_brake_pause   = BRAKE_PAUSE_DEFAULT
    _dynamic_train_rounds  = TRAIN_ROUNDS_DEFAULT
    _top1_threshold        = SKIP_TOP1_THRESHOLD
    _dynamic_lookback      = PATTERN_LOOKBACK_DEFAULT
    _dynamic_decay         = PATTERN_DECAY_DEFAULT
    _dynamic_boost_max     = PATTERN_BOOST_MAX_DEFAULT
    _dynamic_boost_min     = PATTERN_BOOST_MIN_DEFAULT
    _dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT
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
    """
    Scale warmup window with data volume.
    Small dataset → smaller warmup (less wasted data).
    Large dataset → larger warmup (more stable baseline stats).
    """
    # Every 100 rounds of data, add ~3 warmup rounds, clamped to [30, 80]
    scaled = TRAIN_ROUNDS_MIN + int((total_rounds / 100) * 3)
    return max(TRAIN_ROUNDS_MIN, min(TRAIN_ROUNDS_MAX, scaled))

# ── DYNAMIC MARKOV WEIGHTS ────────────────────────────────────────────────────
def recalibrate_markov_weights(hit_bufs):
    """
    For each Markov order 1-4, compute recent hit rate.
    Orders with higher hit rates get boosted weight; lower get reduced.
    Base/recency/gap weights stay fixed as anchors.

    hit_bufs: {order: [True/False, ...]}
    Returns updated weight dict.
    """
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

        # Nudge weight toward hit_rate performance
        # If order is hitting > 65% → boost; < 45% → reduce
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
    """
    Adjust pattern lookback and decay based on data volume and model health.

    total_rounds  : total data available
    sim_hit_rate  : recent sim hit rate (hits/played)

    - Lookback scales with data: more data → can look further back
    - Decay: if model hit rate is low, slow decay (0.3 = more history weight);
             if hit rate is high, faster decay (0.7 = recent pattern dominant)
    """
    global _dynamic_lookback, _dynamic_decay

    # Lookback: scale with data volume
    # 100 rounds → lookback 6; 500 rounds → lookback 12; 800+ → 16
    new_lookback = max(PATTERN_LOOKBACK_MIN,
                       min(PATTERN_LOOKBACK_MAX,
                           int(6 + (total_rounds / 800) * 10)))

    # Decay: inverse relationship with hit rate
    # Low hit rate (model cold) → slower decay (0.30) = use more history
    # High hit rate (model hot) → faster decay (0.70) = trust recent pattern
    if sim_hit_rate < 0.50:
        new_decay = max(PATTERN_DECAY_MIN, _dynamic_decay - 0.03)
    elif sim_hit_rate > 0.68:
        new_decay = min(PATTERN_DECAY_MAX, _dynamic_decay + 0.03)
    else:
        new_decay = _dynamic_decay

    if new_lookback != _dynamic_lookback:
        print(f"[PatternParam] Lookback: {_dynamic_lookback}→{new_lookback} "
              f"(total_rounds={total_rounds})")
        _dynamic_lookback = new_lookback

    if abs(new_decay - _dynamic_decay) > 0.001:
        print(f"[PatternParam] Decay: {_dynamic_decay:.3f}→{new_decay:.3f} "
              f"(sim_hit_rate={sim_hit_rate:.2%})")
        _dynamic_decay = new_decay

# ── DYNAMIC BOOST EFFECTIVENESS ───────────────────────────────────────────────
def recalibrate_boost_cap(boost_eval_log):
    """
    Compare hit rate when pattern boost was triggered vs when it wasn't.
    If boosted rounds hit less than non-boosted → shrink boost cap.
    If boosted rounds hit more → boost cap can stay or grow.

    boost_eval_log: list of (was_boosted: bool, hit: bool)
    """
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
        # Boost is hurting — shrink cap
        new_max = max(PATTERN_BOOST_MIN_DEFAULT, cur_max - PATTERN_BOOST_STEP)
        new_min = max(0.05, cur_min - PATTERN_BOOST_STEP * 0.5)
    elif boosted_rate > unboosted_rate + 0.05:
        # Boost is helping — allow it to grow
        new_max = min(0.50, cur_max + PATTERN_BOOST_STEP)
        new_min = min(0.25, cur_min + PATTERN_BOOST_STEP * 0.5)
    else:
        new_max = cur_max
        new_min = cur_min

    if abs(new_max - cur_max) > 0.001:
        print(f"[BoostCap] max: {cur_max:.3f}→{new_max:.3f} "
              f"(boosted={boosted_rate:.2%} unboosted={unboosted_rate:.2%}, "
              f"n_b={len(boosted_hits)} n_u={len(unboosted_hits)})")
        _dynamic_boost_max = new_max
        _dynamic_boost_min = new_min

# ── DYNAMIC BONUS THRESHOLD ───────────────────────────────────────────────────
def recalibrate_bonus_thresh(bonus_eval_log):
    """
    Track whether rounds where a bonus pick was the correct answer
    actually come from high-confidence bonus classes.
    If bonus picks are consistently losing → raise threshold (be pickier).
    If bonus picks are consistently hitting → lower threshold (be more generous).

    bonus_eval_log: list of (was_bonus_correct: bool, bonus_conf: float)
    Actually simpler: list of (bonus_triggered: bool, hit: bool)
    """
    global _dynamic_bonus_thresh

    if len(bonus_eval_log) < 15:
        return

    window = bonus_eval_log[-BONUS_EVAL_WINDOW:]
    bonus_rounds   = [(triggered, hit) for triggered, hit in window if triggered]
    no_bonus_rounds = [(triggered, hit) for triggered, hit in window if not triggered]

    if len(bonus_rounds) < 5:
        return

    bonus_hit_rate    = sum(h for _, h in bonus_rounds) / len(bonus_rounds)
    no_bonus_hit_rate = (sum(h for _, h in no_bonus_rounds) / len(no_bonus_rounds)
                         if no_bonus_rounds else 0.60)

    cur = _dynamic_bonus_thresh

    if bonus_hit_rate < no_bonus_hit_rate - 0.08:
        # Bonus rounds losing more than normal → raise threshold (harder to trigger)
        new_thresh = min(BONUS_CONF_THRESH_MAX, cur + BONUS_CONF_THRESH_STEP)
    elif bonus_hit_rate > no_bonus_hit_rate + 0.05:
        # Bonus rounds winning more → lower threshold (trigger more often)
        new_thresh = max(BONUS_CONF_THRESH_MIN, cur - BONUS_CONF_THRESH_STEP)
    else:
        new_thresh = cur

    if abs(new_thresh - cur) > 0.0001:
        print(f"[BonusThresh] {cur:.4f}→{new_thresh:.4f} "
              f"(bonus_hr={bonus_hit_rate:.2%} base_hr={no_bonus_hit_rate:.2%}, "
              f"n={len(bonus_rounds)})")
        _dynamic_bonus_thresh = new_thresh

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
    print(f"[Pattern] Group '{group_name}' hit recorded. "
          f"score={pattern_score:.4f} "
          f"new_threshold={_dynamic_threshold[group_name]:.4f} "
          f"(buffer={len(buf)})")

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
            print(f"[DynBrake] Trigger: {cur_trigger}→{new_trigger} "
                  f"(hitrate={hitrate:.2%}, window={len(window)})")

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
            print(f"[DynBrake] Pause: {cur_pause}→{new_pause} "
                  f"(avg_loss_conf={avg_conf:.4f}, buffer={len(window_confs)})")

    return new_trigger, new_pause, hitrate, avg_conf

# ── TOP1 THRESHOLD ADAPTATION ────────────────────────────────────────────────
def recalibrate_top1_threshold(play_pct):
    """
    Mirror entropy threshold adaptation: adjust top1 threshold
    to keep play% within [TARGET_PLAY_MIN, TARGET_PLAY_MAX].
    Both thresholds act as independent valves on the same play% target.
    """
    global _top1_threshold
    cur = _top1_threshold
    if play_pct < TARGET_PLAY_MIN:
        # Playing too rarely — lower the bar to play more
        new = max(TOP1_THRESH_MIN, cur - TOP1_THRESH_STEP)
    elif play_pct > TARGET_PLAY_MAX:
        # Playing too often — raise the bar to play less
        new = min(TOP1_THRESH_MAX, cur + TOP1_THRESH_STEP)
    else:
        new = cur
    if abs(new - cur) > 0.0001:
        print(f"[Adaptive] Top1Thresh: {cur:.4f}→{new:.4f} (play%={play_pct*100:.1f}%)")
        _top1_threshold = new

# ── SCORE ROUND ───────────────────────────────────────────────────────────────
def score_round(h, prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,ag,ar):
    n=len(h)
    if n<4:
        ranked=sorted(prob.items(),key=lambda x:-x[1])
        return [ranked[0][0],ranked[1][0]],{k:1/8 for k in range(1,9)},3.0,0.125,0.125,0.125,ranked[2][0] if len(ranked)>=3 else None, {}

    l1,l2,l3,l4=h[-1],h[-2],h[-3],h[-4]
    k1=(l1,);k2=(l2,l1);k3=(l3,l2,l1);k4=(l4,l3,l2,l1)
    def rel(t,key): return min(1.0,sum(t[key].values())/30) if key in t else 0
    r2=rel(t2,k2);r3=rel(t3,k3);r4=rel(t4,k4)
    rec=h[-100:] if n>=100 else h; rs=h[-20:] if n>=20 else h
    rc=Counter(rec);rsc=Counter(rs)
    lp={}
    for i2,r in enumerate(h): lp[r]=i2
    rv=h[-1];rl2=1
    for i2 in range(n-2,-1,-1):
        if h[i2]==rv: rl2+=1
        else: break
    gh=defaultdict(list);lsh={}
    for i2,r in enumerate(h):
        if r in lsh: gh[r].append(i2-lsh[r])
        lsh[r]=i2
    agh={r:sum(gh[r])/len(gh[r]) if gh.get(r) else ag.get(r,8) for r in range(1,9)}
    rh=defaultdict(list);i2=0
    while i2<n:
        j=i2
        while j<len(h) and h[j]==h[i2]: j+=1
        rh[h[i2]].append(j-i2);i2=j
    arh={r:sum(rh[r])/len(rh[r]) if rh.get(r) else ar.get(r,1.5) for r in range(1,9)}

    # ── DYNAMIC MARKOV WEIGHTS ────────────────────────────────────────────────
    with _lock:
        dw = dict(_dynamic_markov_w)

    sc={}
    for idx in range(1,9):
        base=prob.get(idx,0)
        m1=tp1.get(k1,{}).get(idx,base)
        m2=tp2.get(k2,{}).get(idx,m1) if r2>0 else m1
        m3=tp3.get(k3,{}).get(idx,m2) if r3>0 else m2
        m4=tp4.get(k4,{}).get(idx,m3) if r4>0 else m3
        rp=rc.get(idx,0)/len(rec);rsp=rsc.get(idx,0)/len(rs)
        pos=lp.get(idx,0);agv=agh.get(idx,8)
        od=(n-1-pos)/agv if agv else 0
        ob=min(0.5,max(0.0,(od-1.0)*0.1))
        rpen=0.05 if idx==rv and rl2>=arh.get(idx,1.5) else 0

        # Use dynamic weights, scale order weights by their reliability signal
        wb  = dw["wb"]
        wm1 = dw["wm1"]
        wm2 = dw["wm2"] * r2
        wm3 = dw["wm3"] * r3
        wm4 = dw["wm4"] * r4
        wr  = dw["wr"]
        wv  = dw["wv"]
        wo  = dw["wo"]

        tw  = wb+wm1+wm2+wm3+wm4+wr+wv+wo or 1
        raw = (wb*base+wm1*m1+wm2*m2+wm3*m3+wm4*m4+wr*rp+wv*rsp+wo*ob)
        sc[idx]=max(0.0,raw/tw-rpen)

    # Also capture per-class markov order predictions for hit-tracking
    markov_preds = {}
    for order, tp, key in [(1,tp1,k1),(2,tp2,k2),(3,tp3,k3),(4,tp4,k4)]:
        if key in tp:
            top_cls = max(tp[key], key=tp[key].get)
            markov_preds[order] = top_cls

    ts=sum(sc.values()) or 1
    sc={k:v/ts for k,v in sc.items()}

    # ── PATTERN BOOST ────────────────────────────────────────────────────────
    with _lock:
        dyn_thresh = dict(_dynamic_threshold)

    pattern_scores = compute_pattern_scores(h)
    sc, pattern_info = apply_pattern_boost(sc, pattern_scores, dyn_thresh)

    with _lock:
        _last_pattern_info.clear()
        _last_pattern_info.update(pattern_info)
        _last_pattern_info["raw_scores"] = {g: round(v, 4) for g, v in pattern_scores.items()}

    rk=sorted(sc.items(),key=lambda x:-x[1])
    ent=-sum(v*math.log2(v) for v in sc.values() if v>0)
    t3s = rk[2][1] if len(rk) >= 3 else 0.0
    t3c = rk[2][0] if len(rk) >= 3 else None
    return [rk[0][0],rk[1][0]],sc,ent,rk[0][1],rk[1][1],t3s,t3c,markov_preds

def should_play(t1, ent, brake_active=False):
    if brake_active: return False
    with _lock:
        top1_t = _top1_threshold
    return t1 > top1_t and ent < _entropy_threshold

# ── BONUS PICK LOGIC ──────────────────────────────────────────────────────────
def get_bonus_picks(scores, top2):
    with _lock:
        thresh = _dynamic_bonus_thresh

    ranked     = sorted(scores.items(), key=lambda x: -x[1])
    ranked_ids = [int(k) for k, _ in ranked]
    in_top4    = bool(set(ranked_ids[:4]).intersection(HIGH_MULT_CLASSES))
    above_thresh = any(scores.get(k, scores.get(str(k), 0)) > thresh for k in HIGH_MULT_CLASSES)
    if not in_top4 and not above_thresh:
        return None
    hm = [(int(k), v) for k, v in ranked if int(k) in HIGH_MULT_CLASSES]
    if not hm:
        return None
    bonus = [cls for cls, _ in hm[:2]]
    if not [b for b in bonus if b not in top2]:
        return None
    return bonus

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
    # Per-order: did that order's top prediction match actual?
    sim_markov_hits  = {1:[], 2:[], 3:[], 4:[]}
    sim_boost_log    = []   # (was_boosted, hit)
    sim_bonus_log    = []   # (bonus_triggered, hit)

    with _lock:
        cur_trigger   = _dynamic_brake_trigger
        cur_pause     = _dynamic_brake_pause
        cur_top1_t    = _top1_threshold

    for i in range(train_rounds, total - 1):
        h=rewards[:i+1]; tn=rewards[i+1]
        top2,sc,ent,t1s,_,_,_,markov_preds = score_round(h,*stats)
        ba=brake>0
        if brake>0: brake-=1
        play = t1s > cur_top1_t and ent < _entropy_threshold and not ba
        if not play:
            sk+=1
            if ba: sb+=1
            elif ent>=_entropy_threshold: se+=1
            else: st+=1
            continue

        pl+=1; hit=tn in top2
        sim_play_results.append(hit)

        # Track per-order Markov hit
        for order, pred_cls in markov_preds.items():
            sim_markov_hits[order].append(pred_cls == tn)

        # Track boost effectiveness
        with _lock:
            pat_info = dict(_last_pattern_info)
        any_triggered = pat_info.get("_any_triggered", False)
        sim_boost_log.append((any_triggered, hit))

        # Track bonus effectiveness
        bonus = get_bonus_picks(sc, top2)
        bonus_triggered = bonus is not None
        sim_bonus_log.append((bonus_triggered, hit))

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
    eth = _entropy_threshold

    play = t1s > cur_top1_t and ent < eth and brake == 0

    skip_reason = None
    if brake > 0:              skip_reason = f"Loss brake ({brake} rounds left)"
    elif ent >= eth:           skip_reason = f"High entropy ({ent:.4f} ≥ {eth:.3f})"
    elif t1s <= cur_top1_t:    skip_reason = f"Low confidence ({t1s:.4f} ≤ {cur_top1_t:.4f})"

    last_round = raw_rounds[-1]["round"] if raw_rounds else None
    next_round = (last_round + 1) if last_round else None

    SMALL_MULT = {1, 5, 6, 8}
    pred3       = None
    pred3_conf  = None
    top2_are_small = all(c in SMALL_MULT for c in top2)
    if (play and top2_are_small and t3c is not None and (t2s - t3s) <= 0.01):
        pred3      = t3c
        pred3_conf = round(t3s * 100, 2)

    bonus_picks   = get_bonus_picks(scores, top2) if play else None
    bonus_details = None
    if bonus_picks:
        sc_map = {int(k): v for k, v in scores.items()}
        bonus_details = [
            {"idx": b, "name": CLASS_NAMES.get(b,"?"),
             "color": CLASS_COLORS.get(b,"#888"),
             "conf": round(sc_map.get(b,0)*100, 2)}
            for b in bonus_picks
        ]

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
        "entropy":            round(ent,4),
        "action":             "PLAY" if play else "SKIP",
        "skip_reason":        skip_reason,
        "bonus_picks":        bonus_details,
        "all_scores":         {k: round(v*100,2) for k,v in scores.items()},
        "last_10":            rewards[-10:],
        "total_rounds":       len(rewards),
        "pattern_info":       pattern_summary,
        "dynamic_thresholds": {g: round(v, 4) for g, v in dyn_thresh_snap.items()},
        "brake_trigger":      cur_trigger,
        "brake_pause":        cur_pause,
        "top1_threshold":     round(cur_top1_t, 4),
        "entropy_threshold":  round(eth, 4),
        "train_rounds":       train_rounds,
        "pattern_lookback":   lookback,
        "pattern_decay":      round(decay, 4),
        "pattern_boost_max":  round(boost_max, 4),
        "bonus_conf_thresh":  round(bonus_thresh, 4),
        "markov_weights":     {k: round(v, 4) for k, v in dw.items()},
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
                    pending = _pending_pred

                if pending is not None:
                    pred_round = pending["round"]
                    actual_rec = next((r for r in records if r["round"] == pred_round), None)
                    if actual_rec is not None:
                        actual_val = actual_rec["reward_index"]
                        all_preds  = list(pending.get("top3", pending["top2"]))
                        if pending.get("bonus_picks"):
                            all_preds += [p for p in pending["bonus_picks"] if p not in all_preds]
                        hit = actual_val in all_preds

                        # ── UPDATE PATTERN THRESHOLDS ON HIT ─────────────────
                        if hit and pending.get("pattern_scores"):
                            for group_name, pscore in pending["pattern_scores"].items():
                                update_pattern_hit(group_name, pscore)

                        # ── UPDATE LIVE BRAKE BUFFERS ─────────────────────────
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

                        # ── UPDATE BOOST EVAL LOG ─────────────────────────────
                        with _lock:
                            _boost_eval_log.append((pending.get("any_boosted", False), hit))
                            if len(_boost_eval_log) > PATTERN_BOOST_EVAL_WINDOW * 3:
                                _boost_eval_log.pop(0)
                            boost_log_snap = list(_boost_eval_log)

                        recalibrate_boost_cap(boost_log_snap)

                        # ── UPDATE BONUS EVAL LOG ─────────────────────────────
                        with _lock:
                            _bonus_eval_log.append((pending.get("bonus_triggered", False), hit))
                            if len(_bonus_eval_log) > BONUS_EVAL_WINDOW * 3:
                                _bonus_eval_log.pop(0)
                            bonus_log_snap = list(_bonus_eval_log)

                        recalibrate_bonus_thresh(bonus_log_snap)

                        # ── UPDATE MARKOV HIT BUFFERS ─────────────────────────
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

                sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = \
                    _run_sim(rewards)

                # ── RECALIBRATE ALL DYNAMIC PARAMS FROM SIM ───────────────────
                total_rounds = len(rewards)
                sim_hit_rate = (sim_dict.get("hits", 0) / sim_dict.get("played", 1)
                                if sim_dict.get("played", 0) > 0 else 0.60)

                # 1. Dynamic train rounds
                new_tr = compute_dynamic_train_rounds(total_rounds)
                if new_tr != _dynamic_train_rounds:
                    print(f"[TrainRounds] {_dynamic_train_rounds}→{new_tr} "
                          f"(total_rounds={total_rounds})")
                    _dynamic_train_rounds = new_tr

                # 2. Entropy threshold
                cur = _entropy_threshold
                if play_pct < TARGET_PLAY_MIN:   new_eth = min(ENT_THRESH_MAX, cur + 0.034)
                elif play_pct > TARGET_PLAY_MAX: new_eth = max(ENT_THRESH_MIN, cur - 0.034)
                else:                            new_eth = cur
                if abs(new_eth - cur) > 0.001:
                    print(f"[Adaptive] Entropy: {cur:.3f}→{new_eth:.3f} "
                          f"(play%={play_pct*100:.1f}%)")
                _entropy_threshold = new_eth
                sim_dict["entropy_threshold"] = round(new_eth, 3)

                # 3. Top1 threshold
                recalibrate_top1_threshold(play_pct)

                # 4. Pattern lookback + decay
                recalibrate_pattern_params(total_rounds, sim_hit_rate)

                # 5. Boost cap (from sim boost log if live log is thin)
                with _lock:
                    live_boost_len = len(_boost_eval_log)
                if live_boost_len < 20 and sim_extras.get("boost_log"):
                    recalibrate_boost_cap(sim_extras["boost_log"])

                # 6. Bonus threshold (from sim if live log thin)
                with _lock:
                    live_bonus_len = len(_bonus_eval_log)
                if live_bonus_len < 15 and sim_extras.get("bonus_log"):
                    recalibrate_bonus_thresh(sim_extras["bonus_log"])

                # 7. Markov weights (from sim if live buffers thin)
                with _lock:
                    live_markov_len = min(len(v) for v in _markov_hit_buf.values())
                if live_markov_len < 10 and sim_extras.get("markov_hits"):
                    new_mw = recalibrate_markov_weights(sim_extras["markov_hits"])
                    with _lock:
                        _dynamic_markov_w.update(new_mw)
                        if not any(_markov_hit_buf.values()):
                            for o, v in sim_extras["markov_hits"].items():
                                _markov_hit_buf[o].extend(v[-MARKOV_WEIGHT_BUFFER:])

                # 8. Brake (from sim if live buffers thin)
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

                if cached and cached["_play"] and cached["_next_round"] is not None:
                    nr = cached["_next_round"]
                    with _lock:
                        cur_p = _pending_pred
                    if cur_p is None or cur_p["round"] != nr:
                        # Get current markov preds for live tracking
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
                            "markov_preds":    mp,
                        }
                        with _lock:
                            _pending_pred = np_
                        print(f"[Fetcher] Pending → #{nr} top2={cached['_top2']} "
                              f"trigger={cached['brake_trigger']} "
                              f"pause={cached['brake_pause']} "
                              f"top1_t={cached['top1_threshold']} "
                              f"lookback={cached['pattern_lookback']} "
                              f"decay={cached['pattern_decay']}")

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
#  PWA ROUTES
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
    cur = 0; mx_live = 0
    for e in reversed(live):
        if not e["hit"]: cur += 1; mx_live = max(mx_live, cur)
        else: break
    stats["live_log"]            = live
    stats["live_cur_streak"]     = cur
    stats["live_max_loss"]       = mx_live
    stats["dynamic_thresholds"]  = {g: round(v, 4) for g, v in dyn_t.items()}
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
    stats["markov_weights"]      = {k: round(v, 4) for k, v in dyn_mw.items()}
    stats["bonus_conf_thresh"]   = round(dyn_bt2, 4)
    stats["pattern_boost_max"]   = round(dyn_bmax, 4)
    stats["pattern_lookback"]    = dyn_lk
    stats["pattern_decay"]       = round(dyn_dc, 4)
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
    """Single endpoint showing all live dynamic parameter values and their ranges."""
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
            "markov_weights":      {k: round(v, 4) for k, v in _dynamic_markov_w.items()},
            "markov_weight_range": {"min": MARKOV_WEIGHT_MIN, "max": MARKOV_WEIGHT_MAX},
            "play_target":         {"min": TARGET_PLAY_MIN, "max": TARGET_PLAY_MAX},
        })

# ── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    global _last_reset_date, _pending_pred, _cached_pred, _entropy_threshold, _brake_left
    global _dynamic_brake_trigger, _dynamic_brake_pause, _dynamic_train_rounds

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

        # Compute dynamic train_rounds before sim
        _dynamic_train_rounds = compute_dynamic_train_rounds(total_rounds)
        print(f"[Startup] Loaded {total_rounds} rounds. "
              f"train_rounds={_dynamic_train_rounds}. Building sim...")

        sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = \
            _run_sim(rewards)

        sim_hit_rate = (sim_dict.get("hits", 0) / sim_dict.get("played", 1)
                        if sim_dict.get("played", 0) > 0 else 0.60)

        # Seed all dynamic params from sim
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
                "markov_preds":    mp,
            }
            print(f"[Startup] Pending → #{_pending_pred['round']} "
                  f"top2={_pending_pred['top2']}")

        action = cached.get("action","N/A") if cached else "N/A"
        print(f"[Startup] Done — acc={sim_dict.get('accuracy')}% "
              f"brake={brake} action={action} "
              f"top1_t={round(_top1_threshold,4)} "
              f"lookback={_dynamic_lookback} decay={round(_dynamic_decay,3)}")
    else:
        print("[Startup] No data file. Waiting for API fetch...")

    threading.Thread(target=fetcher_loop, daemon=True).start()
    print(f"[Startup] Fetcher started. IST: {now_ist.strftime('%H:%M:%S')}. Reset at 05:30 IST.")

startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
