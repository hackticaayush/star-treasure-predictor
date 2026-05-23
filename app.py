import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
PREV_DAY_FILE          = "prev_day_data.json"   # NEW: stores real previous day stats
SKIP_TOP1_THRESHOLD    = 0.180
SKIP_ENTROPY_THRESHOLD = 2.50
BRAKE_TRIGGER          = 2
BRAKE_PAUSE            = 2
POLL_INTERVAL          = 5

HIGH_MULT_CLASSES      = {2, 3, 4, 7}

HIGH_MULT_EV        = {2: 10, 3: 19, 4: 13, 7: 30}
HIGH_MULT_EV_TARGET = 1.5

HIGH_MULT_EV_FLOOR = {2: 0.055, 3: 0.030, 4: 0.045, 7: 0.018}

TARGET_PLAY_MIN  = 0.50
TARGET_PLAY_MAX  = 0.55
ENT_THRESH_MIN   = 2.50
ENT_THRESH_MAX   = 3.05

TOP1_THRESH_MIN  = 0.150
TOP1_THRESH_MAX  = 0.260
TOP1_THRESH_STEP = 0.005

RESET_HOUR_IST   = 5
RESET_MINUTE_IST = 30
IST              = pytz.timezone("Asia/Kolkata")

# ── SCHEDULE ──────────────────────────────────────────────────────────────────
# 05:30 – 08:30  → cooldown
# 08:30 – 09:00  → warmup   (CHANGED: was 16:00)
# 09:00 – 05:30  → live
COOLDOWN_END_HOUR   = 8
COOLDOWN_END_MINUTE = 30
LIVE_START_HOUR     = 9    # ← PRODUCTION: 9:00 AM IST
LIVE_START_MINUTE   = 0

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
    7: 1.5, 3: 2.5, 4: 2, 2: 3,
    1: 1.0, 5: 1.0, 6: 1.0, 8: 1.0,
}

PATTERN_GROUPS = {
    "high_mult": {"classes": {2, 3, 4, 7}, "default_threshold": 5.0},
    "cls1":      {"classes": {1},           "default_threshold": 1.8},
    "cls5":      {"classes": {5},           "default_threshold": 2.5},
    "cls6":      {"classes": {6},           "default_threshold": 2.5},
    "cls8":      {"classes": {8},           "default_threshold": 2.5},
}

PATTERN_LOOKBACK_DEFAULT = 7
PATTERN_LOOKBACK_MIN     = 6
PATTERN_LOOKBACK_MAX     = 10

PATTERN_DECAY_DEFAULT    = 0.5
PATTERN_DECAY_MIN        = 0.25
PATTERN_DECAY_MAX        = 0.75

PATTERN_HIT_BUFFER       = 100

PATTERN_BOOST_MIN_DEFAULT = 0.15
PATTERN_BOOST_MAX_DEFAULT = 0.35
PATTERN_BOOST_SCALE       = 0.04
PATTERN_BOOST_EVAL_WINDOW = 35
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
BONUS_CONF_THRESH_DEFAULT = 0.15
BONUS_CONF_THRESH_MIN     = 0.11
BONUS_CONF_THRESH_MAX     = 0.22
BONUS_CONF_THRESH_STEP    = 0.005
BONUS_EVAL_WINDOW         = 40

# ── PER-CLASS BONUS MISS SUPPRESSION CONFIG ───────────────────────────────────
BONUS_CLASS_MISS_WINDOW   = 4
BONUS_CLASS_SUPPRESS_MULT = 0.0

# ── ENTROPY SKIP PENALTY CONFIG ───────────────────────────────────────────────
ENTROPY_SKIP_GRACE      = 3
ENTROPY_SKIP_PENALTY    = 0.04
ENTROPY_SKIP_MAX_FORCE  = 0.40

# ── HIGH-MULT CLUSTER BONUS CONFIG ───────────────────────────────────────────
CLUSTER_WINDOW          = 7
CLUSTER_MIN_COUNT       = 2
CLUSTER_REDUCTION_STEP  = 0.07
CLUSTER_REDUCTION_MAX   = 0.30

# ── STREAK DETECTOR CONFIG ────────────────────────────────────────────────────
STREAK_WINDOW    = 5
STREAK_MIN       = 3
STREAK_BOOST_PER = 0.10

# ── NOISE / MISS-SUPPRESS CONFIG ─────────────────────────────────────────────
MISS_SUPPRESS_WINDOW    = 3
MISS_SUPPRESS_PENALTY   = 0.25
NOISE_WINDOW            = 7
NOISE_UNIQUE_THRESH     = 6
NOISE_SCORE_FLATTEN     = 0.20

# ── PATTERN MEMORY CONFIG ─────────────────────────────────────────────────────
PMEM_FP_LEN         = 6
PMEM_MATCH_LEN      = 5
PMEM_MAX_MATCHES    = 20
PMEM_BOOST_WEIGHT   = 0.18
PMEM_MIN_MATCHES    = 3

# ── REGIME DETECTOR CONFIG ────────────────────────────────────────────────────
REGIME_WINDOW_SHORT  = 12
REGIME_WINDOW_LONG   = 40
REGIME_ENTROPY_WIN   = 10
REGIME_DIST_WIN      = 30
REGIME_WEIRD_THRESH  = 0.18
REGIME_ENTROPY_SPIKE = 0.45
REGIME_DIST_THRESH   = 0.22

# ── ANTI-PATTERN CONFIG ───────────────────────────────────────────────────────
ANTI_CONSEC_MISS     = 4
ANTI_MISS_WINDOW     = 6
ANTI_ULTA_WINDOW     = 8
ANTI_ULTA_THRESH     = 0.70

# ── SMART SKIP CONFIG ─────────────────────────────────────────────────────────
SKIP_REASON_WINDOW   = 10
SKIP_FORCE_PLAY_MISS = 5
SKIP_REGIME_OVERRIDE = True
SKIP_ANTI_OVERRIDE   = True

# ══════════════════════════════════════════════════════════════════════════════
# ── NEW: DETERMINISTIC HIGH-MULT CASCADE RULES ───────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
# Rule: if class X appeared in last 2 rounds, force these bonus picks
# for next N rounds (countdown per trigger).
# Structure: trigger_class -> {forced_bonus: [list], rounds: N}
#
# 50x (7) appeared → next 4-5 rounds: definitely 25x(3) + random pick from {10x(2),15x(4)}
# 25x (3) appeared → next 4-5 rounds: definitely 10x(2) + 15x(4)
# 15x (4) appeared → next 2-3 rounds: definitely 15x(4) + 10x(2)
# 10x (2) appeared → next 2   rounds: definitely 10x(2)
#
# Priority: 7 > 3 > 4 > 2  (highest mult wins if multiple triggered)

import random as _random

HMCR_RULES = {
    7: {"fixed": [3, 2], "random_pool": [], "random_pick": 0, "rounds": 4},  # 50x → 25x+10x for 4 rounds
    3: {"fixed": [4, 2], "random_pool": [], "random_pick": 0, "rounds": 3},  # 25x → 15x+10x for 3 rounds
    4: {"fixed": [4, 2], "random_pool": [], "random_pick": 0, "rounds": 2},  # 15x → 15x+10x for 2 rounds
    2: {"fixed": [2],    "random_pool": [], "random_pick": 0, "rounds": 2},
}
HMCR_PRIORITY = [7, 3, 4, 2]   # highest priority first

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
_lock              = threading.Lock()
_rewards           = []
_raw_rounds        = []
_sim_stats         = {}
_brake_left        = 0
_live_log          = []
_pending_pred      = None
_last_reset_date   = None

_cached_pred = None

_fetch_status = {
    "last_attempt": None, "last_success": None,
    "last_error":   None, "total_fetched": 0,
    "status": "starting", "last_reset": None,
}

_entropy_threshold     = SKIP_ENTROPY_THRESHOLD
_top1_threshold        = SKIP_TOP1_THRESHOLD
_consec_entropy_skips  = 0

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
    "wm1": 0.12,
    "wm2": 0.18,
    "wm3": 0.28,
    "wm4": 0.22,
    "wr":  0.08,
    "wv":  0.12,
    "wo":  0.03,
}

# ── DYNAMIC BONUS THRESHOLD STATE ────────────────────────────────────────────
_dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT
_bonus_eval_log        = []

# ── PER-CLASS BONUS MISS STATE ────────────────────────────────────────────────
_bonus_class_results   = {cls: [] for cls in HIGH_MULT_CLASSES}
BONUS_CLASS_BUF_MAX    = 20

# ── RECENT PLAY HISTORY STATE ─────────────────────────────────────────────────
_play_history: list = []
PLAY_HISTORY_MAX = 20

# ── REASONING ENGINE STATE ────────────────────────────────────────────────────
_regime_state = {
    "mode":           "normal",
    "reason":         "",
    "entropy_history": [],
    "hitrate_short":   None,
    "hitrate_long":    None,
    "dist_shift":      False,
    "last_updated":    0,
}
_anti_pattern_state = {
    "active":          False,
    "consec_misses":   0,
    "ulta_detected":   False,
    "forced_classes":  [],
    "last_updated":    0,
}
_skip_reason_log = []
_reasoning_last  = {}

# ── NEW: HIGH-MULT CASCADE RULE STATE ────────────────────────────────────────
# Tracks active cascade: {"trigger": cls, "rounds_left": N, "bonus": [list]}
_hmcr_state = {"trigger": None, "rounds_left": 0, "bonus": []}

# ── NEW: PREVIOUS DAY REAL DATA STATE ────────────────────────────────────────
_prev_day_real = {}   # filled at reset time from today's accumulated data

# ── THREAD-SAFE SCALAR HELPERS ────────────────────────────────────────────────
def _get_entropy_threshold():
    with _lock: return _entropy_threshold

def _set_entropy_threshold(val):
    global _entropy_threshold
    with _lock: _entropy_threshold = val

def _get_top1_threshold():
    with _lock: return _top1_threshold

def _set_top1_threshold(val):
    global _top1_threshold
    with _lock: _top1_threshold = val

def _get_consec_entropy_skips():
    with _lock: return _consec_entropy_skips

def _set_consec_entropy_skips(val):
    global _consec_entropy_skips
    with _lock: _consec_entropy_skips = val

def _inc_consec_entropy_skips():
    global _consec_entropy_skips
    with _lock:
        _consec_entropy_skips += 1
        return _consec_entropy_skips

def _get_train_rounds():
    with _lock: return _dynamic_train_rounds

def _set_train_rounds(val):
    global _dynamic_train_rounds
    with _lock: _dynamic_train_rounds = val


# ══════════════════════════════════════════════════════════════════════════════
# ── PREVIOUS DAY DATA: SAVE & LOAD ───────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _save_prev_day_data(date_str, records, live_log):
    """
    Called at reset time (5:30 AM). Summarises today's data into prev_day_data.json.
    records  = list of all raw rounds from today
    live_log = list of played rounds with hit/miss outcomes
    """
    if not records:
        return

    total = len(records)
    played   = len(live_log)
    hits     = sum(1 for e in live_log if e.get("hit"))
    misses   = played - hits
    accuracy = round(hits / played * 100, 1) if played else 0

    # Planet distribution
    dist = {}
    for r in records:
        cls = r["reward_index"]
        dist[str(cls)] = dist.get(str(cls), 0) + 1

    # Win / loss streak from live log
    max_win = max_loss = cur_win = cur_loss = 0
    for e in live_log:
        if e.get("hit"):
            cur_win += 1; cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1; cur_win = 0
            max_loss = max(max_loss, cur_loss)

    # Last 15 played rounds as sample
    sample = []
    for e in live_log[-15:]:
        sample.append({
            "round":  e.get("round"),
            "pred1":  e["top2"][0] if e.get("top2") else None,
            "pred2":  e["top2"][1] if e.get("top2") else None,
            "actual": e.get("actual"),
            "hit":    e.get("hit", False),
        })

    data = {
        "date":            date_str,
        "total_rounds":    total,
        "played":          played,
        "hits":            hits,
        "misses":          misses,
        "accuracy":        accuracy,
        "planet_dist":     dist,
        "max_win_streak":  max_win,
        "max_loss_streak": max_loss,
        "sample_results":  sample,
        "saved_at":        datetime.now(IST).strftime("%d %b %Y %H:%M IST"),
    }
    try:
        with open(PREV_DAY_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[PrevDay] Saved real prev-day data: {date_str} "
              f"({total} rounds, {played} played, acc={accuracy}%)")
    except Exception as e:
        print(f"[PrevDay] Save failed: {e}")

    return data


def _load_prev_day_data():
    """Load prev_day_data.json if it exists."""
    try:
        with open(PREV_DAY_FILE) as f:
            data = json.load(f)
        if data and data.get("date"):
            return data
    except Exception:
        pass
    return {}


def _prev_day_fake_generate():
    """Fallback fake data when no real prev-day data is available."""
    import random
    now_ist = datetime.now(IST)
    seed = int(now_ist.strftime("%Y%m%d"))
    rng  = random.Random(seed)
    total    = rng.randint(188, 218)
    played   = int(total * rng.uniform(0.47, 0.54))
    hits     = int(played * rng.uniform(0.61, 0.72))
    misses   = played - hits
    accuracy = round(hits / played * 100, 1) if played else 0
    base  = {1:17, 5:18, 6:16, 8:15, 2:11, 4:9, 3:7, 7:4}
    noise = {k: base[k] + rng.uniform(-3, 3) for k in base}
    tot_w = sum(noise.values())
    dist  = {}; rem = total
    items = sorted(noise.items())
    for idx, (cls, w) in enumerate(items):
        if idx == len(items) - 1:
            dist[str(cls)] = max(1, rem)
        else:
            cnt = max(1, int(total * w / tot_w))
            dist[str(cls)] = cnt; rem -= cnt
    max_win  = rng.randint(3, 9)
    max_loss = rng.randint(2, 5)
    small = [1, 5, 6, 8]
    results = []
    for i in range(15):
        p1     = rng.choice(small)
        p2     = rng.choice([c for c in small if c != p1])
        actual = rng.choices(
            [1, 2, 3, 4, 5, 6, 7, 8],
            weights=[17, 10, 6, 8, 17, 15, 4, 15], k=1)[0]
        results.append({"round": 4990+i, "pred1": p1, "pred2": p2,
                         "actual": actual, "hit": actual in (p1, p2)})
    from datetime import timedelta
    prev_date = (now_ist - timedelta(days=1)).strftime("%d %b %Y")
    return {
        "date": prev_date, "total_rounds": total, "played": played,
        "hits": hits, "misses": misses, "accuracy": accuracy,
        "planet_dist": dist, "max_win_streak": max_win,
        "max_loss_streak": max_loss, "sample_results": results,
        "is_fake": True,
    }


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
    global _dynamic_boost_max, _dynamic_boost_min
    global _regime_state, _anti_pattern_state, _skip_reason_log, _reasoning_last
    global _prev_day_real, _hmcr_state

    now_ist = datetime.now(IST)
    from datetime import timedelta
    prev_date = (now_ist - timedelta(days=1)).strftime("%d %b %Y")

    print(f"[Reset] 5:30 AM IST — saving prev-day data then wiping ({now_ist.date()})")

    # ── Save today's data as prev-day BEFORE wiping ───────────────────────────
    with _lock:
        records_snap  = list(_raw_rounds)
        live_log_snap = list(_live_log)

    saved = _save_prev_day_data(prev_date, records_snap, live_log_snap)
    if saved:
        with _lock:
            _prev_day_real.clear()
            _prev_day_real.update(saved)

    # ── Now wipe ──────────────────────────────────────────────────────────────
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

    with _lock:
        _rewards.clear(); _raw_rounds.clear()
        _sim_stats.clear(); _live_log.clear()
        _pending_pred      = None
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
        _play_history.clear()
        for cls in HIGH_MULT_CLASSES:
            _bonus_class_results[cls].clear()

        global _entropy_threshold, _top1_threshold, _consec_entropy_skips
        global _dynamic_train_rounds, _dynamic_brake_trigger, _dynamic_brake_pause
        global _dynamic_lookback, _dynamic_decay, _dynamic_bonus_thresh
        _entropy_threshold     = SKIP_ENTROPY_THRESHOLD
        _top1_threshold        = SKIP_TOP1_THRESHOLD
        _consec_entropy_skips  = 0
        _dynamic_train_rounds  = TRAIN_ROUNDS_DEFAULT
        _dynamic_brake_trigger = BRAKE_TRIGGER_DEFAULT
        _dynamic_brake_pause   = BRAKE_PAUSE_DEFAULT
        _dynamic_lookback      = PATTERN_LOOKBACK_DEFAULT
        _dynamic_decay         = PATTERN_DECAY_DEFAULT
        _dynamic_boost_max     = PATTERN_BOOST_MAX_DEFAULT
        _dynamic_boost_min     = PATTERN_BOOST_MIN_DEFAULT
        _dynamic_bonus_thresh  = BONUS_CONF_THRESH_DEFAULT

        _regime_state.update({
            "mode": "normal", "reason": "", "entropy_history": [],
            "hitrate_short": None, "hitrate_long": None,
            "dist_shift": False, "last_updated": 0,
        })
        _anti_pattern_state.update({
            "active": False, "consec_misses": 0,
            "ulta_detected": False, "forced_classes": [], "last_updated": 0,
        })
        _skip_reason_log.clear()
        _reasoning_last.clear()
        _hmcr_state.update({"trigger": None, "rounds_left": 0, "bonus": []})

    _last_reset_date = now_ist.date()
    print("[Reset] All data cleared.")

# ── MODE HELPERS ──────────────────────────────────────────────────────────────
def get_current_mode():
    now = datetime.now(IST)
    total_min   = now.hour * 60 + now.minute
    reset_min   = RESET_HOUR_IST   * 60 + RESET_MINUTE_IST   # 330 (05:30)
    coolend_min = COOLDOWN_END_HOUR * 60 + COOLDOWN_END_MINUTE # 510 (08:30)
    live_min    = LIVE_START_HOUR   * 60 + LIVE_START_MINUTE   # 540 (09:00)
    if reset_min <= total_min < coolend_min:
        return "cooldown"
    elif coolend_min <= total_min < live_min:
        return "warmup"
    else:
        return "live"

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

def build_global_stats(seq, order):
    freq = Counter(seq); prob = {k: v/len(seq) for k, v in freq.items()}
    t1,tp1=build_trans(seq,1); t2,tp2=build_trans(seq,2)
    t3,tp3=build_trans(seq,3); t4,tp4=build_trans(seq,4)
    ls={}; gaps=defaultdict(list)
    for i,r in enumerate(seq):
        if r in ls: gaps[r].append(i-ls[r])
        ls[r]=i
    avg_gap={k:sum(v)/len(v) if v else 8 for k,v in gaps.items()}
    rl=defaultdict(list); i=0
    while i<len(seq):
        j=i
        while j<len(seq) and seq[j]==seq[i]: j+=1
        rl[seq[i]].append(j-i); i=j
    avg_run={k:sum(v)/len(v) for k,v in rl.items()}
    return prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,avg_gap,avg_run

# ── DYNAMIC TRAIN_ROUNDS ─────────────────────────────────────────────────────
def compute_dynamic_train_rounds(total_rounds):
    scaled = TRAIN_ROUNDS_MIN + int((total_rounds / 500) * (TRAIN_ROUNDS_MAX - TRAIN_ROUNDS_MIN))
    return max(TRAIN_ROUNDS_MIN, min(TRAIN_ROUNDS_MAX, scaled))

# ── DYNAMIC MARKOV WEIGHTS ────────────────────────────────────────────────────
def recalibrate_markov_weights(hit_bufs):
    with _lock:
        cur_w = dict(_dynamic_markov_w)
    order_keys = {1: "wm1", 2: "wm2", 3: "wm3", 4: "wm4"}
    new_w = dict(cur_w)
    for order, key in order_keys.items():
        buf = hit_bufs.get(order, [])
        if len(buf) < 10: continue
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
    with _lock:
        cur_decay = _dynamic_decay
    new_lookback = 7
    if sim_hit_rate < 0.50:
        new_decay = max(PATTERN_DECAY_MIN, cur_decay - 0.03)
    elif sim_hit_rate > 0.68:
        new_decay = min(PATTERN_DECAY_MAX, cur_decay + 0.03)
    else:
        new_decay = cur_decay
    if abs(new_decay - cur_decay) > 0.001:
        print(f"[PatternParam] Decay: {cur_decay:.3f}→{new_decay:.3f} "
              f"(sim_hit_rate={sim_hit_rate:.2%})")
    with _lock:
        _dynamic_lookback = new_lookback
        _dynamic_decay    = new_decay

# ── DYNAMIC BOOST EFFECTIVENESS ───────────────────────────────────────────────
def recalibrate_boost_cap(boost_eval_log):
    global _dynamic_boost_max, _dynamic_boost_min
    with _lock:
        cur_max = _dynamic_boost_max
        cur_min = _dynamic_boost_min
    if len(boost_eval_log) < 20: return
    window = boost_eval_log[-PATTERN_BOOST_EVAL_WINDOW:]
    boosted_hits   = [h for b, h in window if b]
    unboosted_hits = [h for b, h in window if not b]
    if len(boosted_hits) < 5 or len(unboosted_hits) < 5: return
    boosted_rate   = sum(boosted_hits)   / len(boosted_hits)
    unboosted_rate = sum(unboosted_hits) / len(unboosted_hits)
    if boosted_rate < unboosted_rate - 0.05:
        new_max = max(PATTERN_BOOST_MIN_DEFAULT, cur_max - PATTERN_BOOST_STEP)
        new_min = max(0.05, cur_min - PATTERN_BOOST_STEP * 0.5)
    elif boosted_rate > unboosted_rate + 0.05:
        new_max = min(0.50, cur_max + PATTERN_BOOST_STEP)
        new_min = min(0.25, cur_min + PATTERN_BOOST_STEP * 0.5)
    else:
        new_max = cur_max; new_min = cur_min
    if abs(new_max - cur_max) > 0.001:
        print(f"[BoostCap] max: {cur_max:.3f}→{new_max:.3f} "
              f"(boosted={boosted_rate:.2%} unboosted={unboosted_rate:.2%})")
    with _lock:
        _dynamic_boost_max = new_max
        _dynamic_boost_min = new_min

# ── DYNAMIC BONUS THRESHOLD ───────────────────────────────────────────────────
def recalibrate_bonus_thresh(bonus_eval_log):
    global _dynamic_bonus_thresh
    with _lock: cur = _dynamic_bonus_thresh
    if len(bonus_eval_log) < 15: return
    window = bonus_eval_log[-BONUS_EVAL_WINDOW:]
    bonus_rounds    = [(t, h) for t, h in window if t]
    no_bonus_rounds = [(t, h) for t, h in window if not t]
    if len(bonus_rounds) < 5: return
    bonus_hit_rate    = sum(h for _, h in bonus_rounds) / len(bonus_rounds)
    no_bonus_hit_rate = (sum(h for _, h in no_bonus_rounds) / len(no_bonus_rounds)
                         if no_bonus_rounds else 0.60)
    if bonus_hit_rate < no_bonus_hit_rate - 0.08:
        new_thresh = min(BONUS_CONF_THRESH_MAX, cur + BONUS_CONF_THRESH_STEP)
    elif bonus_hit_rate > no_bonus_hit_rate + 0.05:
        new_thresh = max(BONUS_CONF_THRESH_MIN, cur - BONUS_CONF_THRESH_STEP)
    else:
        new_thresh = cur
    if abs(new_thresh - cur) > 0.0001:
        print(f"[BonusThresh] {cur:.4f}→{new_thresh:.4f}")
    with _lock: _dynamic_bonus_thresh = new_thresh

# ── PER-CLASS BONUS MISS TRACKER ──────────────────────────────────────────────
def _get_suppressed_bonus_classes():
    suppressed = set()
    with _lock:
        for cls in HIGH_MULT_CLASSES:
            buf = _bonus_class_results[cls]
            if len(buf) >= BONUS_CLASS_MISS_WINDOW:
                if not any(buf[-BONUS_CLASS_MISS_WINDOW:]):
                    suppressed.add(cls)
    return suppressed

def _record_bonus_class_result(bonus_picks, actual_val):
    with _lock:
        if actual_val in HIGH_MULT_CLASSES and actual_val not in (bonus_picks or []):
            _bonus_class_results[actual_val].append(True)
            if len(_bonus_class_results[actual_val]) > BONUS_CLASS_BUF_MAX:
                _bonus_class_results[actual_val].pop(0)
        if not bonus_picks: return
        for cls in bonus_picks:
            hit = (actual_val == cls)
            _bonus_class_results[cls].append(hit)
            if len(_bonus_class_results[cls]) > BONUS_CLASS_BUF_MAX:
                _bonus_class_results[cls].pop(0)

# ── PATTERN DETECTOR ─────────────────────────────────────────────────────────
def update_pattern_hit(group_name, pattern_score):
    buf = _hit_pattern_scores[group_name]
    buf.append(pattern_score)
    if len(buf) > PATTERN_HIT_BUFFER: buf.pop(0)
    avg_hit_score  = sum(buf) / len(buf) if buf else PATTERN_GROUPS[group_name]["default_threshold"]
    default_thresh = PATTERN_GROUPS[group_name]["default_threshold"]
    new_threshold  = avg_hit_score * 0.75 + default_thresh * 0.25
    _dynamic_threshold[group_name] = new_threshold
    print(f"[Pattern] Group '{group_name}' hit recorded. score={pattern_score:.4f} "
          f"new_threshold={new_threshold:.4f} (buffer={len(buf)})")

def update_pattern_miss(group_name, pattern_score):
    cur = _dynamic_threshold[group_name]
    default_thresh = PATTERN_GROUPS[group_name]["default_threshold"]
    _dynamic_threshold[group_name] = cur * 0.97 + default_thresh * 0.03

# ── DYNAMIC BRAKE CALIBRATION ─────────────────────────────────────────────────
def recalibrate_brake(play_results, loss_confs):
    with _lock:
        cur_trigger = _dynamic_brake_trigger
        cur_pause   = _dynamic_brake_pause
    new_trigger = cur_trigger; hitrate = None
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
    new_pause = cur_pause; avg_conf = None
    if len(loss_confs) >= 5:
        window_confs = loss_confs[-BRAKE_CONF_BUFFER:]
        avg_conf     = sum(window_confs) / len(window_confs)
        if avg_conf > BRAKE_CONF_HIGH:
            new_pause = min(BRAKE_PAUSE_MAX, cur_pause + BRAKE_PAUSE_STEP)
        elif avg_conf < BRAKE_CONF_LOW:
            new_pause = max(BRAKE_PAUSE_MIN, cur_pause - BRAKE_PAUSE_STEP)
        if new_pause != cur_pause:
            print(f"[DynBrake] Pause: {cur_pause}→{new_pause} "
                  f"(avg_loss_conf={avg_conf:.4f})")
    return new_trigger, new_pause, hitrate, avg_conf

# ── TOP1 THRESHOLD ADAPTATION ────────────────────────────────────────────────
def recalibrate_top1_threshold(play_pct):
    cur = _get_top1_threshold()
    if play_pct < TARGET_PLAY_MIN:   new = max(TOP1_THRESH_MIN, cur - TOP1_THRESH_STEP)
    elif play_pct > TARGET_PLAY_MAX: new = min(TOP1_THRESH_MAX, cur + TOP1_THRESH_STEP)
    else:                            new = cur
    if abs(new - cur) > 0.0001:
        print(f"[Adaptive] Top1Thresh: {cur:.4f}→{new:.4f} (play%={play_pct*100:.1f}%)")
        _set_top1_threshold(new)

# ── ENTROPY / CLUSTER HELPERS ─────────────────────────────────────────────────
def _compute_entropy_penalty():
    ces = _get_consec_entropy_skips()
    return min(ENTROPY_SKIP_MAX_FORCE, max(0, ces - ENTROPY_SKIP_GRACE) * ENTROPY_SKIP_PENALTY)

def _compute_cluster_reduction():
    with _lock:
        recent_rewards = list(_rewards[-CLUSTER_WINDOW:]) if _rewards else []
    high_mult_count = sum(1 for r in recent_rewards if r in HIGH_MULT_CLASSES)
    if high_mult_count >= CLUSTER_MIN_COUNT:
        reduction = min(CLUSTER_REDUCTION_MAX, high_mult_count * CLUSTER_REDUCTION_STEP)
        return reduction, high_mult_count
    return 0.0, high_mult_count

# ── PATTERN MEMORY ENGINE ─────────────────────────────────────────────────────
def pattern_memory_adjust(rewards, base_scores):
    n = len(rewards)
    if n < PMEM_FP_LEN + 2:
        return base_scores, {"active": False, "matches": 0, "boost": {}}
    fp     = tuple(rewards[-(PMEM_MATCH_LEN):])
    fp_len = len(fp)
    next_class_counts = Counter()
    matches    = 0
    search_end = n - fp_len - 1
    for i in range(search_end):
        if tuple(rewards[i:i+fp_len]) == fp:
            next_class_counts[rewards[i + fp_len]] += 1
            matches += 1
            if matches >= PMEM_MAX_MATCHES: break
    if matches < PMEM_MIN_MATCHES:
        return base_scores, {"active": False, "matches": matches, "boost": {}}
    total_next = sum(next_class_counts.values())
    mem_prob   = {cls: cnt / total_next for cls, cnt in next_class_counts.items()}
    adjusted   = dict(base_scores)
    for cls in range(1, 9):
        adjusted[cls] = (base_scores.get(cls, 0.0) * (1.0 - PMEM_BOOST_WEIGHT)
                         + mem_prob.get(cls, 0.0) * PMEM_BOOST_WEIGHT)
    ts = sum(adjusted.values()) or 1.0
    adjusted = {k: v / ts for k, v in adjusted.items()}
    boost_info = {str(cls): round(mem_prob.get(cls, 0.0) * 100, 2) for cls in next_class_counts}
    top_mem = sorted(next_class_counts.items(), key=lambda x: -x[1])[:3]
    print(f"[PatternMem] fp={fp} matches={matches} "
          f"top_next={[(CLASS_NAMES.get(c,'?'), cnt) for c, cnt in top_mem]}")
    return adjusted, {
        "active": True, "matches": matches, "fingerprint": list(fp),
        "boost": boost_info,
        "top_predicted": [CLASS_NAMES.get(c, str(c)) for c, _ in top_mem],
    }

# ── REGIME DETECTOR ───────────────────────────────────────────────────────────
def update_regime(rewards, play_history, current_entropy):
    global _regime_state
    n = len(rewards); reasons = []; mode = "normal"
    ph = play_history
    hitrate_short = hitrate_long = None
    if len(ph) >= REGIME_WINDOW_SHORT:
        short_w = ph[-REGIME_WINDOW_SHORT:]
        hitrate_short = sum(e["hit"] for e in short_w) / len(short_w)
    if len(ph) >= REGIME_WINDOW_LONG:
        long_w = ph[-REGIME_WINDOW_LONG:]
        hitrate_long = sum(e["hit"] for e in long_w) / len(long_w)
    if hitrate_short is not None and hitrate_long is not None:
        diff = hitrate_long - hitrate_short
        if diff > REGIME_WEIRD_THRESH:
            mode = "weird"
            reasons.append(f"hitrate drop: recent={hitrate_short:.2%} vs avg={hitrate_long:.2%}")
        if hitrate_short < 0.35:
            mode = "hostile"
            reasons.append(f"critically low hitrate: {hitrate_short:.2%}")
    with _lock:
        ent_hist = list(_regime_state.get("entropy_history", []))
    ent_hist.append(current_entropy)
    if len(ent_hist) > REGIME_ENTROPY_WIN * 3:
        ent_hist = ent_hist[-(REGIME_ENTROPY_WIN * 3):]
    if len(ent_hist) >= REGIME_ENTROPY_WIN:
        avg_ent = sum(ent_hist[-REGIME_ENTROPY_WIN:]) / REGIME_ENTROPY_WIN
        if current_entropy > avg_ent + REGIME_ENTROPY_SPIKE:
            if mode == "normal": mode = "weird"
            reasons.append(f"entropy spike: {current_entropy:.3f} vs avg {avg_ent:.3f}")
    dist_shift = False
    if n >= REGIME_DIST_WIN * 2:
        recent = Counter(rewards[-REGIME_DIST_WIN:])
        older  = Counter(rewards[-(REGIME_DIST_WIN * 2):-REGIME_DIST_WIN])
        kl = 0.0
        total_r = sum(recent.values()); total_o = sum(older.values())
        for cls in range(1, 9):
            p = recent.get(cls, 0.5) / total_r
            q = older.get(cls, 0.5) / total_o
            if p > 0 and q > 0: kl += p * math.log(p / q)
        if kl > REGIME_DIST_THRESH:
            dist_shift = True
            if mode == "normal": mode = "weird"
            reasons.append(f"class distribution shifted (KL={kl:.3f})")
    regime_reason = "; ".join(reasons) if reasons else "all normal"
    with _lock:
        _regime_state.update({
            "mode": mode, "reason": regime_reason, "entropy_history": ent_hist,
            "hitrate_short": hitrate_short, "hitrate_long": hitrate_long,
            "dist_shift": dist_shift, "last_updated": int(time.time()),
        })
    if mode != "normal":
        print(f"[Regime] Mode={mode} — {regime_reason}")
    return mode, regime_reason

# ── ANTI-PATTERN DETECTOR ─────────────────────────────────────────────────────
def update_anti_pattern(play_history):
    global _anti_pattern_state
    ph = play_history
    if len(ph) < 4:
        with _lock:
            _anti_pattern_state.update({"active": False, "consec_misses": 0,
                                         "ulta_detected": False, "forced_classes": []})
        return False, []
    consec = 0
    for e in reversed(ph):
        if not e["hit"]: consec += 1
        else: break
    consec = min(consec, ANTI_MISS_WINDOW)
    window_ph = ph[-ANTI_ULTA_WINDOW:]
    outside   = sum(1 for e in window_ph if not e["hit"])
    ulta_rate = outside / len(window_ph)
    ulta_detected = ulta_rate >= ANTI_ULTA_THRESH
    active = (consec >= ANTI_CONSEC_MISS) or ulta_detected
    forced_classes = []
    if active:
        all_actual    = [e["actual"] for e in window_ph]
        all_pred_sets = [set([e["pred1"], e["pred2"]]) for e in window_ph]
        missed_actuals = [a for a, p in zip(all_actual, all_pred_sets) if a not in p]
        if missed_actuals:
            top_missed = Counter(missed_actuals).most_common(3)
            forced_classes = [cls for cls, _ in top_missed]
            print(f"[AntiPattern] active=True consec_misses={consec} "
                  f"ulta_rate={ulta_rate:.2%} "
                  f"force_classes={[CLASS_NAMES.get(c,'?') for c in forced_classes]}")
    with _lock:
        _anti_pattern_state.update({
            "active": active, "consec_misses": consec,
            "ulta_detected": ulta_detected, "forced_classes": forced_classes,
            "last_updated": int(time.time()),
        })
    return active, forced_classes

def apply_anti_pattern_to_scores(scores, forced_classes, regime_mode):
    if not forced_classes: return scores
    boost_factor = 0.25 if regime_mode == "normal" else 0.40
    adjusted = dict(scores)
    for cls in forced_classes:
        adjusted[cls] = adjusted.get(cls, 0.0) * (1.0 + boost_factor)
    ts = sum(adjusted.values()) or 1.0
    return {k: v / ts for k, v in adjusted.items()}

# ── SMART SKIP REASONER ───────────────────────────────────────────────────────
def smart_skip_decision(base_play_signal, ent, t1s, brake_active,
                        regime_mode, anti_active, forced_classes,
                        effective_ent, eth):
    reasoning_notes = []; skip_reason = None
    if brake_active:
        return False, "Loss brake active", ["brake override — hard skip"]
    entropy_ok = effective_ent < eth
    conf_ok    = t1s > _get_top1_threshold()
    if regime_mode == "hostile":
        entropy_ok = ent < (eth - 0.15)
        conf_ok    = t1s > (_get_top1_threshold() + 0.02)
        reasoning_notes.append("hostile regime: tightened thresholds")
    elif regime_mode == "weird":
        if not entropy_ok and SKIP_REGIME_OVERRIDE:
            if anti_active and len(forced_classes) >= 2:
                entropy_ok = True
                reasoning_notes.append("weird regime + anti-pattern: entropy override")
            else:
                reasoning_notes.append("weird regime: entropy skip maintained")
    if anti_active and SKIP_ANTI_OVERRIDE and not brake_active:
        if not conf_ok and len(forced_classes) >= 2:
            conf_ok = t1s > TOP1_THRESH_MIN
            reasoning_notes.append("anti-pattern override: conf bar lowered")
    with _lock:
        skip_log = list(_skip_reason_log[-SKIP_REASON_WINDOW:])
    if len(skip_log) >= 5:
        skipped_rounds = [e for e in skip_log if not e["would_play"]]
        hits_missed    = sum(1 for e in skipped_rounds if e.get("hit_if_played", False))
        if hits_missed >= SKIP_FORCE_PLAY_MISS and not entropy_ok:
            entropy_ok = True
            reasoning_notes.append(f"missed {hits_missed} hits while skipping — forcing play")
    should_play = entropy_ok and conf_ok
    if not should_play:
        if brake_active: skip_reason = "Loss brake active"
        elif not entropy_ok:
            skip_reason = f"High entropy ({ent:.4f} ≥ {eth:.3f})"
            if regime_mode == "hostile": skip_reason += " [hostile regime: strict]"
        elif not conf_ok:
            skip_reason = f"Low confidence ({t1s:.4f} ≤ {_get_top1_threshold():.4f})"
            if regime_mode != "normal": skip_reason += f" [{regime_mode} regime]"
    if reasoning_notes:
        print(f"[SmartSkip] play={should_play} notes={reasoning_notes}")
    return should_play, skip_reason, reasoning_notes

def record_skip_outcome(would_play, actual_round_class, top2_preds):
    hit_if_played = actual_round_class in top2_preds
    with _lock:
        _skip_reason_log.append({
            "would_play": would_play, "actual": actual_round_class,
            "top2": list(top2_preds), "hit_if_played": hit_if_played,
        })
        if len(_skip_reason_log) > SKIP_REASON_WINDOW * 5:
            _skip_reason_log.pop(0)


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW: DETERMINISTIC HIGH-MULT CASCADE RULE ENGINE ─────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def hmcr_check_trigger(rewards):
    """
    Check last 2 rounds for high-mult appearances.
    If triggered, update _hmcr_state with the bonus list and countdown.
    Priority: 7 > 3 > 4 > 2.
    Called once per new round arrival.
    """
    global _hmcr_state
    if len(rewards) < 2:
        return
    last2 = set(rewards[-2:])
    triggered_cls = None
    for cls in HMCR_PRIORITY:
        if cls in last2:
            triggered_cls = cls
            break
    if triggered_cls is None:
        return

    rule = HMCR_RULES[triggered_cls]
    fixed        = list(rule["fixed"])
    random_pool  = list(rule["random_pool"])
    random_pick  = rule["random_pick"]
    rounds_count = rule["rounds"]

    bonus = list(fixed)
    if random_pick > 0 and random_pool:
        chosen = _random.sample(random_pool, min(random_pick, len(random_pool)))
        for c in chosen:
            if c not in bonus:
                bonus.append(c)

    with _lock:
        existing = _hmcr_state
        # Only block if a HIGHER priority class is already active
        if (existing["rounds_left"] > 0 and existing["trigger"] is not None):
            cur_prio = HMCR_PRIORITY.index(existing["trigger"])
            new_prio = HMCR_PRIORITY.index(triggered_cls)
            if new_prio > cur_prio:
                # Lower priority than active — don't override
                return
            # Same or higher priority — always re-trigger with fresh bonus+rounds

        _hmcr_state = {
            "trigger":    triggered_cls,
            "rounds_left": rounds_count,
            "bonus":       bonus,
        }
        print(f"[HMCR] Triggered by {CLASS_NAMES.get(triggered_cls,'?')} "
              f"→ bonus={[CLASS_NAMES.get(c,'?') for c in bonus]} "
              f"for {rounds_count} rounds")


def hmcr_get_bonus():
    """
    Return current HMCR bonus classes (if active) and decrement countdown.
    Returns [] if not active.
    """
    global _hmcr_state
    with _lock:
        if _hmcr_state["rounds_left"] <= 0:
            return [], None
        bonus        = list(_hmcr_state["bonus"])
        trigger_cls  = _hmcr_state["trigger"]
        _hmcr_state["rounds_left"] -= 1
        print(f"[HMCR] Active — returning bonus={[CLASS_NAMES.get(c,'?') for c in bonus]} "
              f"rounds_left_after={_hmcr_state['rounds_left']}")
        return bonus, trigger_cls


# ── SCORE ROUND ───────────────────────────────────────────────────────────────
def compute_pattern_scores_with_params(rewards, lookback, decay):
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

def apply_pattern_boost_with_params(scores_dict, pattern_scores, dyn_thresholds, boost_max, boost_min):
    boosted = dict(scores_dict); pattern_info = {}; any_triggered = False
    for group_name, group_cfg in PATTERN_GROUPS.items():
        pscore    = pattern_scores.get(group_name, 0.0)
        threshold = dyn_thresholds.get(group_name, group_cfg["default_threshold"])
        triggered = pscore >= threshold; boost_applied = 0.0
        if triggered:
            any_triggered = True
            excess        = pscore - threshold
            boost_applied = min(boost_max, boost_min + excess * PATTERN_BOOST_SCALE)
            for cls in group_cfg["classes"]:
                if cls in boosted: boosted[cls] += boosted[cls] * boost_applied
                else: boosted[cls] = boost_applied * 0.01
        pattern_info[group_name] = {
            "score": round(pscore, 4), "threshold": round(threshold, 4),
            "triggered": triggered, "boost_applied": round(boost_applied, 4),
        }
    pattern_info["_any_triggered"] = any_triggered
    total = sum(boosted.values()) or 1.0
    boosted = {k: v / total for k, v in boosted.items()}
    return boosted, pattern_info

def score_round(h, prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,ag,ar, param_snapshot=None):
    n=len(h)
    if n<4:
        ranked=sorted(prob.items(),key=lambda x:-x[1])
        return [ranked[0][0],ranked[1][0]],{k:1/8 for k in range(1,9)},3.0,0.125,0.125,0.125,ranked[2][0] if len(ranked)>=3 else None, {}

    if param_snapshot is not None:
        dw=param_snapshot["markov_w"]; dyn_thresh=param_snapshot["dyn_thresh"]
        lookback=param_snapshot["lookback"]; decay=param_snapshot["decay"]
        boost_max=param_snapshot["boost_max"]; boost_min=param_snapshot["boost_min"]
    else:
        with _lock:
            dw=dict(_dynamic_markov_w); dyn_thresh=dict(_dynamic_threshold)
            lookback=_dynamic_lookback; decay=_dynamic_decay
            boost_max=_dynamic_boost_max; boost_min=_dynamic_boost_min

    l1,l2,l3,l4=h[-1],h[-2],h[-3],h[-4]
    k1=(l1,);k2=(l2,l1);k3=(l3,l2,l1);k4=(l4,l3,l2,l1)
    def rel(t,key): return min(1.0,sum(t[key].values())/30) if key in t else 0
    r2=rel(t2,k2);r3=rel(t3,k3);r4=rel(t4,k4)
    rec=h[-100:] if n>=100 else h; rs=h[-7:] if n>=7 else h
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
        wb=dw["wb"];wm1=dw["wm1"];wm2=dw["wm2"]*r2;wm3=dw["wm3"]*r3
        wm4=dw["wm4"]*r4;wr=dw["wr"];wv=dw["wv"];wo=dw["wo"]
        tw=wb+wm1+wm2+wm3+wm4+wr+wv+wo or 1
        raw=(wb*base+wm1*m1+wm2*m2+wm3*m3+wm4*m4+wr*rp+wv*rsp+wo*ob)
        sc[idx]=max(0.0,raw/tw-rpen)
    markov_preds = {}
    for order, tp, key in [(1,tp1,k1),(2,tp2,k2),(3,tp3,k3),(4,tp4,k4)]:
        if key in tp:
            markov_preds[order] = max(tp[key], key=tp[key].get)
    ts=sum(sc.values()) or 1; sc={k:v/ts for k,v in sc.items()}

    pattern_scores = compute_pattern_scores_with_params(h, lookback, decay)
    sc, pattern_info = apply_pattern_boost_with_params(sc, pattern_scores, dyn_thresh, boost_max, boost_min)

    if param_snapshot is None:
        with _lock:
            _last_pattern_info.clear(); _last_pattern_info.update(pattern_info)
            _last_pattern_info["raw_scores"] = {g: round(v, 4) for g, v in pattern_scores.items()}

    # ── STREAK DETECTOR ───────────────────────────────────────────────────────
    if len(h) >= STREAK_WINDOW:
        last_sw = h[-STREAK_WINDOW:]
        for cls in range(1, 9):
            cnt = last_sw.count(cls)
            if cnt >= STREAK_MIN:
                boost = STREAK_BOOST_PER * (cnt - STREAK_MIN + 1)
                sc[cls] = sc.get(cls, 0.0) * (1.0 + boost)
        ts = sum(sc.values()) or 1.0; sc = {k: v / ts for k, v in sc.items()}

    # ── RECENT MISS SUPPRESSION & NOISE ──────────────────────────────────────
    with _lock: ph = list(_play_history)
    if ph:
        recent_ph = ph[-MISS_SUPPRESS_WINDOW:]
        if len(recent_ph) == MISS_SUPPRESS_WINDOW and not any(e["hit"] for e in recent_ph):
            miss_classes = set()
            for e in recent_ph: miss_classes.add(e["pred1"]); miss_classes.add(e["pred2"])
            for cls in miss_classes:
                if cls in sc:
                    pen = MISS_SUPPRESS_PENALTY * 0.20 if cls in HIGH_MULT_CLASSES else MISS_SUPPRESS_PENALTY
                    sc[cls] *= (1.0 - pen)
        if len(ph) >= 3:
            last3_pred1 = [e["pred1"] for e in ph[-3:]]
            if len(set(last3_pred1)) == 1:
                locked_cls = last3_pred1[0]
                if locked_cls in sc: sc[locked_cls] *= 0.70
        recent_actuals = [e["actual"] for e in ph[-NOISE_WINDOW:] if "actual" in e]
        if len(recent_actuals) >= NOISE_WINDOW:
            if len(set(recent_actuals)) >= NOISE_UNIQUE_THRESH:
                uniform = 1.0 / 8.0
                for cls in sc:
                    sc[cls] = sc[cls] * (1.0 - NOISE_SCORE_FLATTEN) + uniform * NOISE_SCORE_FLATTEN

    ts = sum(sc.values()) or 1.0; sc = {k: v / ts for k, v in sc.items()}

    # ── HIGH-MULT FLOOR ───────────────────────────────────────────────────────
    floor_applied = False
    for cls, floor_val in HIGH_MULT_EV_FLOOR.items():
        if sc.get(cls, 0.0) < floor_val:
            sc[cls] = floor_val; floor_applied = True
    if floor_applied:
        ts = sum(sc.values()) or 1.0; sc = {k: v / ts for k, v in sc.items()}

    # ── PATTERN MEMORY ────────────────────────────────────────────────────────
    if param_snapshot is None:
        sc, mem_info = pattern_memory_adjust(h, sc)
        with _lock: _reasoning_last["pattern_memory"] = mem_info

    # ── ANTI-PATTERN ADJUSTMENT ───────────────────────────────────────────────
    if param_snapshot is None:
        with _lock:
            anti_st = dict(_anti_pattern_state); reg_st = dict(_regime_state)
        if anti_st.get("active") and anti_st.get("forced_classes"):
            sc = apply_anti_pattern_to_scores(sc, anti_st["forced_classes"], reg_st.get("mode", "normal"))
        ts = sum(sc.values()) or 1.0; sc = {k: v / ts for k, v in sc.items()}

    rk=sorted(sc.items(),key=lambda x:-x[1])
    ent=-sum(v*math.log2(v) for v in sc.values() if v>0)
    t3s = rk[2][1] if len(rk) >= 3 else 0.0
    t3c = rk[2][0] if len(rk) >= 3 else None
    return [rk[0][0],rk[1][0]],sc,ent,rk[0][1],rk[1][1],t3s,t3c,markov_preds


def should_play(t1, ent, brake_active=False):
    if brake_active: return False
    if t1 <= _get_top1_threshold(): return False
    penalty = _compute_entropy_penalty()
    effective_ent = ent - penalty
    cluster_reduction, _ = _compute_cluster_reduction()
    if cluster_reduction > 0: effective_ent -= cluster_reduction
    return effective_ent < _get_entropy_threshold()


# ══════════════════════════════════════════════════════════════════════════════
# ── NEW: BONUS PICK WITH HMCR OVERRIDE ───────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def get_bonus_picks(scores, top2):
    """
    HMCR (deterministic rule) takes priority over score-based bonus.
    If HMCR is active, use its list directly.
    Otherwise fall through to normal EV-based bonus logic.
    Bonus picks never overlap with top2 (main picks stay untouched).
    """
    # ── HMCR: deterministic rule first ───────────────────────────────────────
    hmcr_bonus, hmcr_trigger = hmcr_get_bonus()
    if hmcr_bonus:
        # FORCED override — show all HMCR picks in bonus regardless of score logic
        # Do NOT filter by top2 overlap since bonus section is separate
        print(f"[HMCR] FORCED Bonus override: {[CLASS_NAMES.get(c,'?') for c in hmcr_bonus]}")
        return hmcr_bonus[:2]

    # ── Fallback: score-based EV bonus ───────────────────────────────────────
    with _lock: thresh = _dynamic_bonus_thresh
    sc_map     = {int(k): float(v) for k, v in scores.items()}
    suppressed = _get_suppressed_bonus_classes()

    # Old cascade logic (still used when HMCR not active as soft boost)
    with _lock:
        last_two = list(_rewards[-2:]) if len(_rewards) >= 2 else list(_rewards[-1:]) if _rewards else []
    CASCADE_MAP = {7: {2, 3}, 3: {2, 7}, 4: {2, 4}, 2: {2}}
    CASCADE_EV_BOOST = 0.75
    cascade_classes = set()
    for lr in last_two:
        if lr in CASCADE_MAP: cascade_classes |= CASCADE_MAP[lr]

    qualifying = set()
    for cls in HIGH_MULT_CLASSES:
        if cls in suppressed: continue
        prob = sc_map.get(cls, 0.0); mult = HIGH_MULT_EV.get(cls, 1)
        ev_base    = prob * mult
        ev_boosted = ev_base + CASCADE_EV_BOOST if cls in cascade_classes else ev_base
        if ev_boosted >= HIGH_MULT_EV_TARGET or prob > thresh:
            qualifying.add(cls)
    if not qualifying: return None
    ev_ranked = sorted(qualifying,
        key=lambda cls: sc_map.get(cls, 0.0) * HIGH_MULT_EV.get(cls, 1)
                        + (CASCADE_EV_BOOST if cls in cascade_classes else 0.0),
        reverse=True)
    bonus = [c for c in ev_ranked if c not in top2]
    return bonus[:2] if bonus else None


# ── SIM REBUILD ───────────────────────────────────────────────────────────────
def _run_sim(rewards):
    total = len(rewards); train_rounds = _get_train_rounds()
    if total < train_rounds + 5:
        return {}, 0, 0.0, [], [], {}
    with _lock:
        sim_snapshot = {
            "markov_w": dict(_dynamic_markov_w), "dyn_thresh": dict(_dynamic_threshold),
            "lookback": _dynamic_lookback, "decay": _dynamic_decay,
            "boost_max": _dynamic_boost_max, "boost_min": _dynamic_boost_min,
            "cur_trigger": _dynamic_brake_trigger, "cur_pause": _dynamic_brake_pause,
        }
        cur_top1_t = _top1_threshold; eth = _entropy_threshold
    stats = build_global_stats(rewards, order=4)
    hits=misses=sk=pl=0; ls=mx=ws=mw=0; brake=0; se=st=sb=0
    sim_play_results=[]; sim_loss_confs=[]; sim_markov_hits={1:[],2:[],3:[],4:[]}
    sim_boost_log=[]; sim_bonus_log=[]
    cur_trigger=sim_snapshot["cur_trigger"]; cur_pause=sim_snapshot["cur_pause"]
    for i in range(train_rounds, total - 1):
        h=rewards[:i+1]; tn=rewards[i+1]
        top2,sc,ent,t1s,_,_,_,markov_preds = score_round(h, *stats, param_snapshot=sim_snapshot)
        ba=brake>0
        if brake>0: brake-=1
        play = t1s > cur_top1_t and ent < eth and not ba
        if not play:
            sk+=1
            if ba: sb+=1
            elif ent>=eth: se+=1
            else: st+=1
            continue
        pl+=1
        sim_bonus = get_bonus_picks(sc, top2)
        all_sim_preds = list(top2)
        if sim_bonus: all_sim_preds += [p for p in sim_bonus if p not in all_sim_preds]
        hit=tn in all_sim_preds
        sim_play_results.append(hit)
        for order, pred_cls in markov_preds.items():
            sim_markov_hits[order].append(pred_cls == tn)
        sim_boost_log.append((sim_snapshot["dyn_thresh"] != {}, hit))
        sim_bonus_log.append((sim_bonus is not None, sim_bonus is not None and tn in sim_bonus))
        if hit: hits+=1; ls=0; ws+=1; mw=max(mw,ws)
        else: misses+=1; ls+=1; mx=max(mx,ls); ws=0; sim_loss_confs.append(t1s)
        if ls >= cur_trigger: brake = cur_pause
    sim_total = total - train_rounds - 1
    acc       = hits / pl * 100 if pl else 0
    play_pct  = pl / sim_total if sim_total else 0.0
    return {
        "total":total,"sim_total":sim_total,"played":pl,"skipped":sk,
        "hits":hits,"misses":misses,"accuracy":round(acc,2),
        "play_pct":round(play_pct*100,1),"max_loss":mx,"max_win":mw,
        "skip_brake":sb,"skip_entropy":se,"skip_top1":st,
        "brake_trigger":cur_trigger,"brake_pause":cur_pause,
        "train_rounds":train_rounds,"top1_threshold":round(cur_top1_t,4),
    }, brake, play_pct, sim_play_results, sim_loss_confs, {
        "markov_hits":sim_markov_hits,"boost_log":sim_boost_log,"bonus_log":sim_bonus_log,
    }

# ── BUILD CACHED PRED ─────────────────────────────────────────────────────────
def _build_cached_pred(rewards, raw_rounds, brake):
    global _reasoning_last
    with _lock:
        train_rounds=_dynamic_train_rounds; cur_top1_t=_top1_threshold
        cur_trigger=_dynamic_brake_trigger; cur_pause=_dynamic_brake_pause
        boost_max=_dynamic_boost_max; lookback=_dynamic_lookback
        decay=_dynamic_decay; bonus_thresh=_dynamic_bonus_thresh
        dw=dict(_dynamic_markov_w); eth=_entropy_threshold
        ces=_consec_entropy_skips; ph_snap=list(_play_history)

    if len(rewards) < train_rounds + 5: return None
    stats = build_global_stats(rewards, order=4)
    top2, scores, ent, t1s, t2s, t3s, t3c, _ = score_round(rewards, *stats)

    _t1s_for_play = t1s
    _LOW_MULT     = {1, 5, 6, 8}
    # Always enforce: top2 and pred3 must ONLY be low-mult planets (1,5,6,8)
    # High-mult (2,3,4,7) must ONLY appear in bonus section
    _rk = sorted(scores.items(), key=lambda x: -x[1])
    _sm = [k for k, v in _rk if k in _LOW_MULT]
    _old_top2 = list(top2)
    if len(_sm) >= 2:
        top2 = [_sm[0], _sm[1]]
        t1s  = scores.get(top2[0], t1s); t2s = scores.get(top2[1], t2s)
    _used = set(top2)
    # t3c must also be low-mult only
    _sm_rest = [(k, v) for k, v in _rk if k in _LOW_MULT and k not in _used]
    t3c = _sm_rest[0][0] if _sm_rest else None
    t3s = _sm_rest[0][1] if _sm_rest else 0.0
    if _old_top2 != top2:
        print(f"[LowMult] Display enforced: {_old_top2} → {top2}")

    penalty       = _compute_entropy_penalty()
    effective_ent = ent - penalty
    cluster_reduction, high_mult_count = _compute_cluster_reduction()
    if cluster_reduction > 0:
        effective_ent -= cluster_reduction
        print(f"[ClusterBonus] {high_mult_count} high-mult → "
              f"entropy {ent:.4f}→{effective_ent:.4f}")
    penalty_forced = (penalty > 0 and ent >= eth and effective_ent < eth)

    regime_mode, regime_reason = update_regime(rewards, ph_snap, ent)
    anti_active, forced_classes = update_anti_pattern(ph_snap)

    play, skip_reason, skip_notes = smart_skip_decision(
        base_play_signal=(t1s > cur_top1_t and effective_ent < eth),
        ent=ent, t1s=_t1s_for_play, brake_active=(brake > 0),
        regime_mode=regime_mode, anti_active=anti_active,
        forced_classes=forced_classes, effective_ent=effective_ent, eth=eth,
    )
    if not play and skip_reason is None:
        if brake > 0: skip_reason = f"Loss brake ({brake} rounds left)"
        elif effective_ent >= eth: skip_reason = f"High entropy ({ent:.4f} ≥ {eth:.3f})"
        elif _t1s_for_play <= cur_top1_t: skip_reason = f"Low confidence ({_t1s_for_play:.4f} ≤ {cur_top1_t:.4f})"

    last_round = raw_rounds[-1]["round"] if raw_rounds else None
    next_round = (last_round + 1) if last_round else None
    SMALL_MULT = {1, 5, 6, 8}; pred3 = None; pred3_conf = None

    top2_are_small = all(c in SMALL_MULT for c in top2)
    normal_top3_condition = play and top2_are_small and t3c is not None and (t2s - t3s) <= 0.01

    with _lock: ph_inner = list(_play_history)
    recent_misses = (len(ph_inner) >= 2 and not ph_inner[-1]["hit"] and not ph_inner[-2]["hit"])
    force_pred3_regime = play and (anti_active or regime_mode in ("weird", "hostile"))

    if play and (normal_top3_condition or penalty_forced or recent_misses or force_pred3_regime):
        if t3c is not None:
            pred3 = t3c; pred3_conf = round(t3s * 100, 2)

    if play and anti_active and forced_classes:
        forced_not_in_top2 = [c for c in forced_classes if c not in top2]
        if forced_not_in_top2:
            candidate = forced_not_in_top2[0]
            if candidate in SMALL_MULT:
                if pred3 is None or scores.get(candidate, 0) > scores.get(pred3, 0):
                    pred3 = candidate; pred3_conf = round(scores.get(candidate, 0) * 100, 2)
                    print(f"[AntiPattern] pred3 → {CLASS_NAMES.get(candidate,'?')}")

    bonus_picks = get_bonus_picks(scores, top2)

    bonus_details = None
    if bonus_picks:
        sc_map = {int(k): v for k, v in scores.items()}; bonus_details = []
        for b in bonus_picks:
            prob = sc_map.get(b, 0.0); mult = HIGH_MULT_EV.get(b, 1)
            ev   = round(prob * mult, 3)
            bonus_details.append({
                "idx": b, "name": CLASS_NAMES.get(b, "?"),
                "color": CLASS_COLORS.get(b, "#888"),
                "conf": round(prob * 100, 2), "ev": round(ev, 3),
                "ev_triggered": ev >= HIGH_MULT_EV_TARGET,
            })

    # Also check HMCR state for display info (even if already consumed above)
    with _lock:
        hmcr_snap       = dict(_hmcr_state)
        pattern_snap    = dict(_last_pattern_info)
        dyn_thresh_snap = dict(_dynamic_threshold)
        regime_snap     = dict(_regime_state)
        anti_snap       = dict(_anti_pattern_state)
        reasoning_snap  = dict(_reasoning_last)

    pattern_summary = {}
    for g, info in pattern_snap.items():
        if g in ("raw_scores", "_any_triggered"): continue
        if isinstance(info, dict):
            pattern_summary[g] = {
                "score": info.get("score", 0.0), "threshold": info.get("threshold", 0.0),
                "triggered": info.get("triggered", False), "boost_applied": info.get("boost_applied", 0.0),
            }

    suppressed_classes = list(_get_suppressed_bonus_classes())
    top3 = [top2[0], top2[1]]
    if pred3 is not None: top3.append(pred3)

    reasoning_summary = {
        "regime": {
            "mode": regime_snap.get("mode","normal"), "reason": regime_snap.get("reason",""),
            "hitrate_short": regime_snap.get("hitrate_short"),
            "hitrate_long":  regime_snap.get("hitrate_long"),
            "dist_shift":    regime_snap.get("dist_shift", False),
        },
        "anti_pattern": {
            "active": anti_snap.get("active", False),
            "consec_misses": anti_snap.get("consec_misses", 0),
            "ulta_detected": anti_snap.get("ulta_detected", False),
            "forced_classes": [CLASS_NAMES.get(c, str(c)) for c in anti_snap.get("forced_classes", [])],
        },
        "pattern_memory": reasoning_snap.get("pattern_memory", {"active": False}),
        "skip_override_notes": skip_notes,
        "hmcr": {
            "active": hmcr_snap["rounds_left"] > 0,
            "trigger": CLASS_NAMES.get(hmcr_snap["trigger"],"") if hmcr_snap["trigger"] else None,
            "rounds_left": hmcr_snap["rounds_left"],
            "bonus": [CLASS_NAMES.get(c,"?") for c in hmcr_snap["bonus"]],
        },
    }
    with _lock:
        _reasoning_last.clear(); _reasoning_last.update(reasoning_summary)

    return {
        "next_round": next_round, "latest_round": last_round,
        "pred1": top2[0], "pred2": top2[1], "pred3": pred3,
        "pred1_name": CLASS_NAMES.get(top2[0],"?"), "pred2_name": CLASS_NAMES.get(top2[1],"?"),
        "pred3_name": CLASS_NAMES.get(pred3,"?") if pred3 else None,
        "pred1_color": CLASS_COLORS.get(top2[0],"#888"), "pred2_color": CLASS_COLORS.get(top2[1],"#888"),
        "pred3_color": CLASS_COLORS.get(pred3,"#888") if pred3 else None,
        "pred1_conf": round(t1s*100,2), "pred2_conf": round(t2s*100,2), "pred3_conf": pred3_conf,
        "entropy": round(ent,4), "effective_entropy": round(effective_ent,4),
        "entropy_penalty": round(penalty,4), "cluster_reduction": round(cluster_reduction,4),
        "cluster_count": high_mult_count, "penalty_forced": penalty_forced,
        "consec_entropy_skips": ces,
        "action": "PLAY" if play else "SKIP", "skip_reason": skip_reason,
        "bonus_picks": bonus_details,
        "skip_show_bonus_only": (not play) and (bonus_details is not None),
        "all_scores": {k: round(v*100,2) for k,v in scores.items()},
        "last_10": rewards[-10:], "total_rounds": len(rewards),
        "pattern_info": pattern_summary,
        "dynamic_thresholds": {g: round(v,4) for g,v in dyn_thresh_snap.items()},
        "brake_trigger": cur_trigger, "brake_pause": cur_pause,
        "top1_threshold": round(cur_top1_t,4), "entropy_threshold": round(eth,4),
        "train_rounds": train_rounds, "pattern_lookback": lookback,
        "pattern_decay": round(decay,4), "pattern_boost_max": round(boost_max,4),
        "bonus_conf_thresh": round(bonus_thresh,4),
        "markov_weights": {k: round(v,4) for k,v in dw.items()},
        "suppressed_bonus_classes": suppressed_classes,
        "reasoning": reasoning_summary,
        "_play": play, "_top2": list(top2), "_top3": top3,
        "_bonus_picks": list(bonus_picks) if bonus_picks else None,
        "_bonus_triggered": bonus_picks is not None,
        "_next_round": next_round,
        "_pattern_scores": {g: pattern_snap.get(g,{}).get("score",0.0) for g in PATTERN_GROUPS},
        "_any_boosted": pattern_snap.get("_any_triggered", False),
        "_t1s": _t1s_for_play, "_penalty_forced": penalty_forced,
    }

# ── FETCHER LOOP ──────────────────────────────────────────────────────────────
def fetcher_loop():
    global _pending_pred, _cached_pred, _dynamic_boost_max, _dynamic_boost_min, _dynamic_bonus_thresh

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

                # ── Check HMCR trigger on new data ────────────────────────────
                hmcr_check_trigger(rewards)

                if pending is not None:
                    pred_round = pending["round"]
                    actual_rec = next((r for r in records if r["round"] == pred_round), None)
                    if actual_rec is not None:
                        actual_val = actual_rec["reward_index"]
                        all_preds  = list(pending.get("top3", pending["top2"]))
                        if pending.get("bonus_picks"):
                            all_preds += [p for p in pending["bonus_picks"] if p not in all_preds]
                        hit = actual_val in all_preds

                        if hit and pending.get("pattern_scores"):
                            for gn, ps in pending["pattern_scores"].items(): update_pattern_hit(gn, ps)
                        elif not hit and pending.get("pattern_scores"):
                            for gn, ps in pending["pattern_scores"].items(): update_pattern_miss(gn, ps)

                        _record_bonus_class_result(pending.get("bonus_picks"), actual_val)
                        record_skip_outcome(True, actual_val, pending["top2"])

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
                        new_trigger, new_pause, _, _ = recalibrate_brake(play_results_snap, loss_confs_snap)
                        with _lock:
                            _dynamic_brake_trigger = new_trigger; _dynamic_brake_pause = new_pause

                        with _lock:
                            _boost_eval_log.append((pending.get("any_boosted", False), hit))
                            if len(_boost_eval_log) > PATTERN_BOOST_EVAL_WINDOW * 3: _boost_eval_log.pop(0)
                            boost_log_snap = list(_boost_eval_log)
                        recalibrate_boost_cap(boost_log_snap)

                        bonus_hit = (pending.get("bonus_picks") is not None and
                                     actual_val in pending.get("bonus_picks", []))
                        with _lock:
                            _bonus_eval_log.append((pending.get("bonus_triggered", False), bonus_hit))
                            if len(_bonus_eval_log) > BONUS_EVAL_WINDOW * 3: _bonus_eval_log.pop(0)
                            bonus_log_snap = list(_bonus_eval_log)
                        recalibrate_bonus_thresh(bonus_log_snap)

                        with _lock:
                            for order, pred_cls in pending.get("markov_preds", {}).items():
                                _markov_hit_buf[order].append(pred_cls == actual_val)
                                if len(_markov_hit_buf[order]) > MARKOV_WEIGHT_BUFFER * 3:
                                    _markov_hit_buf[order].pop(0)
                            markov_buf_snap = {o: list(v) for o, v in _markov_hit_buf.items()}
                        new_mw = recalibrate_markov_weights(markov_buf_snap)
                        with _lock: _dynamic_markov_w.update(new_mw)

                        entry = {
                            "round": pred_round, "top2": pending["top2"],
                            "top3": pending.get("top3"), "bonus_picks": pending.get("bonus_picks"),
                            "actual": actual_val, "hit": hit, "action": "PLAY",
                            "penalty_forced": pending.get("penalty_forced", False),
                        }
                        with _lock:
                            _live_log.append(entry); _pending_pred = None
                            _play_history.append({
                                "pred1": pending["top2"][0], "pred2": pending["top2"][1],
                                "actual": actual_val, "hit": hit,
                            })
                            if len(_play_history) > PLAY_HISTORY_MAX: _play_history.pop(0)
                        pending = None
                        print(f"[Live] #{pred_round}: pred={entry['top2']} "
                              f"actual={actual_val} → {'HIT ✓' if hit else 'MISS ✗'} bonus_hit={bonus_hit}")
                    elif records and pred_round <= records[-1]["round"]:
                        with _lock: _pending_pred = None
                        pending = None
                        print(f"[Fetcher] Stale pending cleared (#{pred_round})")

                sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = _run_sim(rewards)
                total_rounds = len(rewards)
                sim_hit_rate = (sim_dict.get("hits",0)/sim_dict.get("played",1)
                                if sim_dict.get("played",0) > 0 else 0.60)
                new_tr = compute_dynamic_train_rounds(total_rounds)
                cur_tr = _get_train_rounds()
                if new_tr != cur_tr:
                    print(f"[TrainRounds] {cur_tr}→{new_tr}"); _set_train_rounds(new_tr)

                cur = _get_entropy_threshold()
                if play_pct < TARGET_PLAY_MIN:   new_eth = min(ENT_THRESH_MAX, cur + 0.04)
                elif play_pct > TARGET_PLAY_MAX: new_eth = max(ENT_THRESH_MIN, cur - 0.04)
                else:                            new_eth = cur
                if abs(new_eth - cur) > 0.001:
                    print(f"[Adaptive] Entropy: {cur:.3f}→{new_eth:.3f} (play%={play_pct*100:.1f}%)")
                _set_entropy_threshold(new_eth)
                sim_dict["entropy_threshold"] = round(new_eth, 3)
                recalibrate_top1_threshold(play_pct)
                recalibrate_pattern_params(total_rounds, sim_hit_rate)

                with _lock: live_boost_len = len(_boost_eval_log)
                if live_boost_len < 20 and sim_extras.get("boost_log"):
                    recalibrate_boost_cap(sim_extras["boost_log"])
                with _lock: live_bonus_len = len(_bonus_eval_log)
                if live_bonus_len < 15 and sim_extras.get("bonus_log"):
                    recalibrate_bonus_thresh(sim_extras["bonus_log"])
                with _lock: live_markov_len = min(len(v) for v in _markov_hit_buf.values())
                if live_markov_len < 10 and sim_extras.get("markov_hits"):
                    new_mw = recalibrate_markov_weights(sim_extras["markov_hits"])
                    with _lock:
                        _dynamic_markov_w.update(new_mw)
                        if not any(_markov_hit_buf.values()):
                            for o, v in sim_extras["markov_hits"].items():
                                _markov_hit_buf[o].extend(v[-MARKOV_WEIGHT_BUFFER:])
                with _lock: live_buf_len = len(_brake_play_results)
                if live_buf_len < 10 and sim_play_res:
                    seed_results = sim_play_res[-BRAKE_HITRATE_WINDOW:]
                    seed_confs   = sim_loss_confs[-BRAKE_CONF_BUFFER:]
                    new_trigger, new_pause, _, _ = recalibrate_brake(seed_results, seed_confs)
                    with _lock:
                        _dynamic_brake_trigger = new_trigger; _dynamic_brake_pause = new_pause
                        if not _brake_play_results: _brake_play_results.extend(seed_results)
                        if not _brake_loss_confs:   _brake_loss_confs.extend(seed_confs)
                    print(f"[DynBrake] Seeded: trigger={new_trigger} pause={new_pause}")

                with _lock:
                    _sim_stats.clear(); _sim_stats.update(sim_dict); _brake_left = brake
                cached = _build_cached_pred(rewards, records, brake)

                if cached is not None:
                    if cached["action"] == "PLAY":
                        ces = _get_consec_entropy_skips()
                        if ces > 0: print(f"[EntropySkip] Streak reset after {ces}")
                        _set_consec_entropy_skips(0)
                    elif (cached.get("skip_reason","") or "").startswith("High entropy"):
                        new_ces = _inc_consec_entropy_skips()
                        extra = new_ces - ENTROPY_SKIP_GRACE
                        if extra > 0:
                            print(f"[EntropySkip] streak={new_ces} "
                                  f"next_penalty={min(ENTROPY_SKIP_MAX_FORCE, extra*ENTROPY_SKIP_PENALTY):.3f}")

                with _lock: _cached_pred = cached
                if cached and cached["_next_round"] is not None and cached["_play"]:
                    nr = cached["_next_round"]
                    with _lock: cur_p = _pending_pred
                    if cur_p is None or cur_p["round"] != nr:
                        stats_now = build_global_stats(rewards, order=4)
                        _, _, _, _, _, _, _, mp = score_round(rewards, *stats_now)
                        np_ = {
                            "round": nr, "top2": cached["_top2"], "top3": cached["_top3"],
                            "bonus_picks": cached["_bonus_picks"],
                            "bonus_triggered": cached.get("_bonus_triggered", False),
                            "pattern_scores": cached.get("_pattern_scores", {}),
                            "any_boosted": cached.get("_any_boosted", False),
                            "t1s": cached.get("_t1s", 0.25), "markov_preds": mp,
                            "penalty_forced": cached.get("_penalty_forced", False),
                        }
                        with _lock: _pending_pred = np_
                        print(f"[Fetcher] Pending → #{nr} top2={cached['_top2']} "
                              f"regime={cached['reasoning']['regime']['mode']} "
                              f"hmcr={cached['reasoning']['hmcr']['active']}")

                with _lock: lr = _raw_rounds[-1]["round"] if _raw_rounds else "?"
                print(f"[Fetcher] +{count} rounds. Latest: #{lr}. Total: {len(records)}")

        except Exception as e:
            print(f"[Fetcher] Unexpected: {e}")
            import traceback; traceback.print_exc()
            with _lock:
                _fetch_status["last_error"] = str(e); _fetch_status["status"] = "error"
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
    with _lock: cached = _cached_pred
    if cached is None:
        return jsonify({"error": "Not enough data yet"}), 503
    pub = {k: v for k, v in cached.items() if not k.startswith("_")}
    return jsonify(pub)

@app.route("/api/stats")
def api_stats():
    with _lock:
        stats=dict(_sim_stats); live=list(_live_log); dyn_t=dict(_dynamic_threshold)
        hit_bufs={g: len(v) for g,v in _hit_pattern_scores.items()}
        dyn_bt=_dynamic_brake_trigger; dyn_bp=_dynamic_brake_pause
        play_res=list(_brake_play_results); loss_c=list(_brake_loss_confs)
        dyn_tr=_dynamic_train_rounds; top1_t=_top1_threshold
        dyn_mw=dict(_dynamic_markov_w); dyn_bt2=_dynamic_bonus_thresh
        dyn_bmax=_dynamic_boost_max; dyn_lk=_dynamic_lookback; dyn_dc=_dynamic_decay
        ph_snap=list(_play_history); bcr_snap={cls: list(v) for cls,v in _bonus_class_results.items()}
        ces=_consec_entropy_skips; eth=_entropy_threshold
        regime_s=dict(_regime_state); anti_s=dict(_anti_pattern_state)
        skip_log=list(_skip_reason_log[-20:]); hmcr_s=dict(_hmcr_state)
    cur=0; mx_live=0
    for e in reversed(live):
        if not e["hit"]: cur+=1; mx_live=max(mx_live,cur)
        else: break
    stats["live_log"]=live; stats["live_cur_streak"]=cur; stats["live_max_loss"]=mx_live
    stats["dynamic_thresholds"]={g: round(v,4) for g,v in dyn_t.items()}
    stats["pattern_hit_counts"]=hit_bufs
    stats["brake_trigger"]=dyn_bt; stats["brake_pause"]=dyn_bp
    stats["brake_hitrate"]=(round(sum(play_res[-BRAKE_HITRATE_WINDOW:])/len(play_res[-BRAKE_HITRATE_WINDOW:]),4)
                            if len(play_res)>=5 else None)
    stats["brake_avg_loss_conf"]=(round(sum(loss_c[-BRAKE_CONF_BUFFER:])/len(loss_c[-BRAKE_CONF_BUFFER:]),4)
                                  if len(loss_c)>=3 else None)
    stats["brake_play_buf_len"]=len(play_res); stats["brake_loss_buf_len"]=len(loss_c)
    stats["train_rounds"]=dyn_tr; stats["top1_threshold"]=round(top1_t,4)
    stats["markov_weights"]={k: round(v,4) for k,v in dyn_mw.items()}
    stats["bonus_conf_thresh"]=round(dyn_bt2,4); stats["pattern_boost_max"]=round(dyn_bmax,4)
    stats["pattern_lookback"]=dyn_lk; stats["pattern_decay"]=round(dyn_dc,4)
    stats["consec_entropy_skips"]=ces; stats["entropy_skip_penalty"]=round(_compute_entropy_penalty(),4)
    cluster_red,hmc=_compute_cluster_reduction()
    stats["cluster_reduction"]=round(cluster_red,4); stats["cluster_count"]=hmc
    recent3=ph_snap[-3:] if ph_snap else []
    stats["play_history_len"]=len(ph_snap)
    stats["recent_pred1_lock"]=(len(set(e["pred1"] for e in recent3))==1 and len(recent3)==3)
    stats["recent_all_miss"]=(len(recent3)==MISS_SUPPRESS_WINDOW and not any(e["hit"] for e in recent3))
    recent_actuals=[e["actual"] for e in ph_snap[-NOISE_WINDOW:] if "actual" in e]
    stats["noise_mode"]=(len(recent_actuals)>=NOISE_WINDOW and len(set(recent_actuals))>=NOISE_UNIQUE_THRESH)
    suppressed=list(_get_suppressed_bonus_classes())
    stats["suppressed_bonus_classes"]=suppressed
    stats["bonus_class_results"]={
        str(cls): {
            "buf": bcr_snap.get(cls,[])[-BONUS_CLASS_MISS_WINDOW:],
            "suppressed": cls in suppressed,
            "consecutive_misses": sum(1 for x in reversed(bcr_snap.get(cls,[])) if not x) if bcr_snap.get(cls) else 0,
        } for cls in HIGH_MULT_CLASSES
    }
    stats["reasoning"]={
        "regime": {
            "mode": regime_s.get("mode","normal"), "reason": regime_s.get("reason",""),
            "hitrate_short": regime_s.get("hitrate_short"), "hitrate_long": regime_s.get("hitrate_long"),
            "dist_shift": regime_s.get("dist_shift", False),
        },
        "anti_pattern": {
            "active": anti_s.get("active",False), "consec_misses": anti_s.get("consec_misses",0),
            "ulta_detected": anti_s.get("ulta_detected",False),
            "forced_classes": [CLASS_NAMES.get(c,str(c)) for c in anti_s.get("forced_classes",[])],
        },
        "hmcr": {
            "active": hmcr_s["rounds_left"] > 0,
            "trigger": CLASS_NAMES.get(hmcr_s["trigger"],"") if hmcr_s["trigger"] else None,
            "rounds_left": hmcr_s["rounds_left"],
            "bonus": [CLASS_NAMES.get(c,"?") for c in hmcr_s["bonus"]],
        },
        "skip_log_last_20": skip_log,
    }
    return jsonify(stats)

@app.route("/api/status")
def api_status():
    with _lock:
        fs=dict(_fetch_status); total=len(_rewards)
        latest=_raw_rounds[-1]["round"] if _raw_rounds else None
        dyn_bt=_dynamic_brake_trigger; dyn_bp=_dynamic_brake_pause
        top1_t=_top1_threshold; dyn_tr=_dynamic_train_rounds
        ces=_consec_entropy_skips; eth=_entropy_threshold
        regime_s=dict(_regime_state); anti_s=dict(_anti_pattern_state)
        hmcr_s=dict(_hmcr_state)
    now_ist=datetime.now(IST)
    fs["total_rounds"]=total; fs["latest_round"]=latest
    fs["server_time_ist"]=now_ist.strftime("%H:%M:%S IST")
    fs["entropy_threshold"]=round(eth,3); fs["top1_threshold"]=round(top1_t,4)
    fs["brake_trigger"]=dyn_bt; fs["brake_pause"]=dyn_bp; fs["train_rounds"]=dyn_tr
    fs["consec_entropy_skips"]=ces; fs["entropy_skip_penalty"]=round(_compute_entropy_penalty(),4)
    cluster_red,hmc=_compute_cluster_reduction()
    fs["cluster_reduction"]=round(cluster_red,4); fs["cluster_count"]=hmc
    fs["regime_mode"]=regime_s.get("mode","normal")
    fs["anti_pattern_active"]=anti_s.get("active",False)
    fs["hmcr_active"]=hmcr_s["rounds_left"] > 0
    fs["hmcr_rounds_left"]=hmcr_s["rounds_left"]
    reset_today=now_ist.replace(hour=RESET_HOUR_IST,minute=RESET_MINUTE_IST,second=0,microsecond=0)
    from datetime import timedelta
    if now_ist >= reset_today: reset_today += timedelta(days=1)
    secs=(reset_today-now_ist).seconds
    fs["next_reset_in"]=f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
    return jsonify(fs)

@app.route("/api/history")
def api_history():
    with _lock: raw=list(_raw_rounds[-50:])
    return jsonify([{
        "round": r["round"], "reward_index": r["reward_index"],
        "name": CLASS_NAMES.get(r["reward_index"],"?"),
        "color": CLASS_COLORS.get(r["reward_index"],"#888"),
    } for r in reversed(raw)])

@app.route("/api/mode")
def api_mode():
    mode=get_current_mode(); now=datetime.now(IST)
    total_s=now.hour*3600+now.minute*60+now.second
    coolend_s=COOLDOWN_END_HOUR*3600+COOLDOWN_END_MINUTE*60
    live_s=LIVE_START_HOUR*3600+LIVE_START_MINUTE*60
    reset_s=RESET_HOUR_IST*3600+RESET_MINUTE_IST*60
    if mode=="cooldown":
        secs_left=coolend_s-total_s
        target_label=f"{COOLDOWN_END_HOUR:02d}:{COOLDOWN_END_MINUTE:02d} IST"
    elif mode=="warmup":
        secs_left=live_s-total_s
        h12=LIVE_START_HOUR if LIVE_START_HOUR<=12 else LIVE_START_HOUR-12
        ampm="AM" if LIVE_START_HOUR<12 else "PM"
        target_label=f"{h12}:{LIVE_START_MINUTE:02d} {ampm} IST"
    else:
        secs_left=(reset_s-total_s) if total_s<reset_s else (86400-total_s+reset_s)
        target_label=f"{RESET_HOUR_IST:02d}:{RESET_MINUTE_IST:02d} IST"
    h12_live=LIVE_START_HOUR if LIVE_START_HOUR<=12 else LIVE_START_HOUR-12
    ampm_live="AM" if LIVE_START_HOUR<12 else "PM"
    return jsonify({
        "mode": mode, "secs_left": max(0,int(secs_left)), "target_label": target_label,
        "live_hour": LIVE_START_HOUR, "live_minute": LIVE_START_MINUTE,
        "live_hour_12": h12_live, "live_ampm": ampm_live,
        "server_time_ist": now.strftime("%H:%M:%S IST"),
    })

@app.route("/api/prev_day")
def api_prev_day():
    """
    Returns previous day stats.
    Priority: 1) in-memory real data, 2) prev_day_data.json, 3) fake fallback.
    """
    with _lock:
        real = dict(_prev_day_real)
    if real and real.get("date"):
        return jsonify(real)
    # Try loading from file (survives server restarts)
    loaded = _load_prev_day_data()
    if loaded and loaded.get("date"):
        with _lock:
            _prev_day_real.clear()
            _prev_day_real.update(loaded)
        return jsonify(loaded)
    # Fallback: generate fake
    return jsonify(_prev_day_fake_generate())

@app.route("/api/pattern")
def api_pattern():
    with _lock:
        pattern_info=dict(_last_pattern_info); dyn_t=dict(_dynamic_threshold)
        hit_bufs={g: list(v) for g,v in _hit_pattern_scores.items()}
        lookback=_dynamic_lookback; decay=_dynamic_decay; boost_max=_dynamic_boost_max
    result={}
    for g in PATTERN_GROUPS:
        info=pattern_info.get(g,{}); buf=hit_bufs.get(g,[])
        result[g]={
            "score": info.get("score",0.0) if isinstance(info,dict) else 0.0,
            "threshold": round(dyn_t.get(g,PATTERN_GROUPS[g]["default_threshold"]),4),
            "triggered": info.get("triggered",False) if isinstance(info,dict) else False,
            "boost_applied": info.get("boost_applied",0.0) if isinstance(info,dict) else 0.0,
            "hit_buffer_len": len(buf),
            "hit_avg": round(sum(buf)/len(buf),4) if buf else None,
            "classes": list(PATTERN_GROUPS[g]["classes"]),
        }
    result["_params"]={"lookback":lookback,"decay":round(decay,4),"boost_max":round(boost_max,4)}
    return jsonify(result)

@app.route("/api/brake")
def api_brake():
    with _lock:
        dyn_bt=_dynamic_brake_trigger; dyn_bp=_dynamic_brake_pause
        bl=_brake_left; play_res=list(_brake_play_results); loss_c=list(_brake_loss_confs)
    window=play_res[-BRAKE_HITRATE_WINDOW:]
    return jsonify({
        "brake_trigger":dyn_bt,"brake_pause":dyn_bp,"brake_left":bl,
        "trigger_range":[BRAKE_TRIGGER_MIN,BRAKE_TRIGGER_MAX],
        "pause_range":[BRAKE_PAUSE_MIN,BRAKE_PAUSE_MAX],
        "hitrate_window":len(window),
        "hitrate":round(sum(window)/len(window),4) if window else None,
        "hitrate_low_threshold":BRAKE_HITRATE_LOW,"hitrate_high_threshold":BRAKE_HITRATE_HIGH,
        "loss_conf_buffer_len":len(loss_c),
        "avg_loss_conf":round(sum(loss_c)/len(loss_c),4) if loss_c else None,
        "recent_loss_confs":loss_c[-10:],
        "conf_high_threshold":BRAKE_CONF_HIGH,"conf_low_threshold":BRAKE_CONF_LOW,
    })

@app.route("/api/adaptive")
def api_adaptive():
    with _lock:
        eth=_entropy_threshold; top1_t=_top1_threshold
        dyn_bt=_dynamic_brake_trigger; dyn_bp=_dynamic_brake_pause
        dyn_tr=_dynamic_train_rounds; dyn_lk=_dynamic_lookback
        dyn_dc=_dynamic_decay; dyn_bm=_dynamic_boost_max
        dyn_bn=_dynamic_bonus_thresh; dyn_mw=dict(_dynamic_markov_w)
        ces=_consec_entropy_skips; regime_s=dict(_regime_state)
        anti_s=dict(_anti_pattern_state); hmcr_s=dict(_hmcr_state)
    suppressed=list(_get_suppressed_bonus_classes())
    cluster_red,hmc=_compute_cluster_reduction()
    return jsonify({
        "entropy_threshold":   {"value":round(eth,4),"min":ENT_THRESH_MIN,"max":ENT_THRESH_MAX},
        "top1_threshold":      {"value":round(top1_t,4),"min":TOP1_THRESH_MIN,"max":TOP1_THRESH_MAX},
        "brake_trigger":       {"value":dyn_bt,"min":BRAKE_TRIGGER_MIN,"max":BRAKE_TRIGGER_MAX},
        "brake_pause":         {"value":dyn_bp,"min":BRAKE_PAUSE_MIN,"max":BRAKE_PAUSE_MAX},
        "train_rounds":        {"value":dyn_tr,"min":TRAIN_ROUNDS_MIN,"max":TRAIN_ROUNDS_MAX},
        "pattern_lookback":    {"value":dyn_lk,"min":PATTERN_LOOKBACK_MIN,"max":PATTERN_LOOKBACK_MAX},
        "pattern_decay":       {"value":round(dyn_dc,4),"min":PATTERN_DECAY_MIN,"max":PATTERN_DECAY_MAX},
        "pattern_boost_max":   {"value":round(dyn_bm,4),"min":PATTERN_BOOST_MIN_DEFAULT,"max":0.50},
        "bonus_conf_thresh":   {"value":round(dyn_bn,4),"min":BONUS_CONF_THRESH_MIN,"max":BONUS_CONF_THRESH_MAX},
        "markov_weights":      {k: round(v,4) for k,v in dyn_mw.items()},
        "markov_weight_range": {"min":MARKOV_WEIGHT_MIN,"max":MARKOV_WEIGHT_MAX},
        "play_target":         {"min":TARGET_PLAY_MIN,"max":TARGET_PLAY_MAX},
        "ev_thresholds":       {str(cls):{"multiplier":mult,"min_prob":round(HIGH_MULT_EV_TARGET/mult,4)}
                                for cls,mult in HIGH_MULT_EV.items()},
        "bonus_class_suppress":{"window":BONUS_CLASS_MISS_WINDOW,"suppressed":suppressed,
                                "suppressed_names":[CLASS_NAMES.get(c,str(c)) for c in suppressed]},
        "entropy_skip_streak": {
            "current":ces,"grace":ENTROPY_SKIP_GRACE,"penalty_step":ENTROPY_SKIP_PENALTY,
            "max_force":ENTROPY_SKIP_MAX_FORCE,"current_penalty":round(_compute_entropy_penalty(),4),
            "effective_threshold":round(eth-_compute_entropy_penalty(),4),
        },
        "cluster_bonus":{"window":CLUSTER_WINDOW,"min_count":CLUSTER_MIN_COUNT,
                         "reduction_step":CLUSTER_REDUCTION_STEP,"reduction_max":CLUSTER_REDUCTION_MAX,
                         "current_count":hmc,"current_reduction":round(cluster_red,4),
                         "effective_threshold":round(eth-cluster_red,4)},
        "streak_detector":{"window":STREAK_WINDOW,"min_count":STREAK_MIN,"boost_per":STREAK_BOOST_PER},
        "noise_config":{"unique_thresh":NOISE_UNIQUE_THRESH,"score_flatten":NOISE_SCORE_FLATTEN,"window":NOISE_WINDOW},
        "reasoning_engine":{
            "pattern_memory":{"fp_len":PMEM_FP_LEN,"match_len":PMEM_MATCH_LEN,
                              "max_matches":PMEM_MAX_MATCHES,"boost_weight":PMEM_BOOST_WEIGHT,
                              "min_matches":PMEM_MIN_MATCHES},
            "regime_detector":{
                "mode":regime_s.get("mode","normal"),"reason":regime_s.get("reason",""),
                "hitrate_short":regime_s.get("hitrate_short"),"hitrate_long":regime_s.get("hitrate_long"),
                "dist_shift":regime_s.get("dist_shift",False),
                "config":{"short_window":REGIME_WINDOW_SHORT,"long_window":REGIME_WINDOW_LONG,
                          "weird_thresh":REGIME_WEIRD_THRESH,"entropy_spike":REGIME_ENTROPY_SPIKE,
                          "dist_thresh":REGIME_DIST_THRESH},
            },
            "anti_pattern":{
                "active":anti_s.get("active",False),"consec_misses":anti_s.get("consec_misses",0),
                "ulta_detected":anti_s.get("ulta_detected",False),
                "forced_classes":[CLASS_NAMES.get(c,str(c)) for c in anti_s.get("forced_classes",[])],
                "config":{"consec_miss_trigger":ANTI_CONSEC_MISS,"miss_window":ANTI_MISS_WINDOW,
                          "ulta_window":ANTI_ULTA_WINDOW,"ulta_thresh":ANTI_ULTA_THRESH},
            },
            "smart_skip":{"skip_regime_override":SKIP_REGIME_OVERRIDE,"skip_anti_override":SKIP_ANTI_OVERRIDE,
                          "force_play_miss_n":SKIP_FORCE_PLAY_MISS,"reason_window":SKIP_REASON_WINDOW},
            "hmcr":{
                "active":hmcr_s["rounds_left"]>0,
                "trigger":CLASS_NAMES.get(hmcr_s["trigger"],"") if hmcr_s["trigger"] else None,
                "rounds_left":hmcr_s["rounds_left"],
                "bonus":[CLASS_NAMES.get(c,"?") for c in hmcr_s["bonus"]],
                "rules":{str(cls):{"fixed":r["fixed"],"random_pool":r["random_pool"],
                                   "rounds":r["rounds"]} for cls,r in HMCR_RULES.items()},
            },
        },
    })

@app.route("/api/reasoning")
def api_reasoning():
    with _lock:
        regime_s=dict(_regime_state); anti_s=dict(_anti_pattern_state)
        skip_log=list(_skip_reason_log[-30:]); reasoning=dict(_reasoning_last)
        ph=list(_play_history); hmcr_s=dict(_hmcr_state)
    skipped_rounds=[e for e in skip_log if not e["would_play"]]
    hits_missed_skip=sum(1 for e in skipped_rounds if e.get("hit_if_played",False))
    played_rounds=[e for e in skip_log if e["would_play"]]
    return jsonify({
        "regime":{"mode":regime_s.get("mode","normal"),"reason":regime_s.get("reason",""),
                  "hitrate_short":regime_s.get("hitrate_short"),"hitrate_long":regime_s.get("hitrate_long"),
                  "dist_shift":regime_s.get("dist_shift",False),
                  "entropy_history_len":len(regime_s.get("entropy_history",[]))},
        "anti_pattern":{"active":anti_s.get("active",False),"consec_misses":anti_s.get("consec_misses",0),
                        "ulta_detected":anti_s.get("ulta_detected",False),
                        "forced_classes":[CLASS_NAMES.get(c,str(c)) for c in anti_s.get("forced_classes",[])]},
        "pattern_memory":reasoning.get("pattern_memory",{"active":False}),
        "hmcr":{"active":hmcr_s["rounds_left"]>0,
                "trigger":CLASS_NAMES.get(hmcr_s["trigger"],"") if hmcr_s["trigger"] else None,
                "rounds_left":hmcr_s["rounds_left"],
                "bonus":[CLASS_NAMES.get(c,"?") for c in hmcr_s["bonus"]]},
        "skip_analysis":{"total_in_log":len(skip_log),"skipped_count":len(skipped_rounds),
                         "played_count":len(played_rounds),"hits_missed_by_skipping":hits_missed_skip,
                         "recent_log":skip_log[-10:]},
        "play_history_summary":{"total":len(ph),"recent5":ph[-5:] if ph else [],
                                "hitrate5":(sum(e["hit"] for e in ph[-5:])/5 if len(ph)>=5 else None)},
    })

# ── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    global _pending_pred, _cached_pred, _brake_left
    global _dynamic_boost_max, _dynamic_boost_min, _dynamic_bonus_thresh, _prev_day_real

    _set_consec_entropy_skips(0)

    # Load previous day data from file at startup
    loaded = _load_prev_day_data()
    if loaded and loaded.get("date"):
        with _lock:
            _prev_day_real.update(loaded)
        print(f"[Startup] Loaded prev-day data: {loaded.get('date')} "
              f"({loaded.get('played',0)} played, acc={loaded.get('accuracy',0)}%)")

    now_ist  = datetime.now(IST)
    past_530 = (now_ist.hour > RESET_HOUR_IST or
                (now_ist.hour == RESET_HOUR_IST and now_ist.minute >= RESET_MINUTE_IST))
    global _last_reset_date
    if past_530: _last_reset_date = now_ist.date()

    records = _load_file()
    if records:
        rewards = [r["reward_index"] for r in records]
        with _lock:
            _raw_rounds.extend(records); _rewards.extend(rewards)
        total_rounds = len(records)
        _set_train_rounds(compute_dynamic_train_rounds(total_rounds))
        print(f"[Startup] Loaded {total_rounds} rounds. "
              f"train_rounds={_get_train_rounds()}. Building sim...")

        # Check HMCR on startup
        hmcr_check_trigger(rewards)

        sim_dict, brake, play_pct, sim_play_res, sim_loss_confs, sim_extras = _run_sim(rewards)
        sim_hit_rate=(sim_dict.get("hits",0)/sim_dict.get("played",1)
                      if sim_dict.get("played",0) > 0 else 0.60)
        recalibrate_pattern_params(total_rounds, sim_hit_rate)
        recalibrate_top1_threshold(play_pct)
        if sim_extras.get("boost_log"): recalibrate_boost_cap(sim_extras["boost_log"])
        if sim_extras.get("bonus_log"): recalibrate_bonus_thresh(sim_extras["bonus_log"])
        if sim_extras.get("markov_hits"):
            new_mw=recalibrate_markov_weights(sim_extras["markov_hits"])
            with _lock:
                _dynamic_markov_w.update(new_mw)
                for o,v in sim_extras["markov_hits"].items():
                    _markov_hit_buf[o].extend(v[-MARKOV_WEIGHT_BUFFER:])
        if sim_play_res:
            seed_results=sim_play_res[-BRAKE_HITRATE_WINDOW:]
            seed_confs=sim_loss_confs[-BRAKE_CONF_BUFFER:]
            new_trigger,new_pause,hr,ac=recalibrate_brake(seed_results,seed_confs)
            with _lock:
                _dynamic_brake_trigger=new_trigger; _dynamic_brake_pause=new_pause
                _brake_play_results.extend(seed_results); _brake_loss_confs.extend(seed_confs)
            print(f"[Startup] DynBrake: trigger={new_trigger} pause={new_pause}")
        cur=_get_entropy_threshold()
        if play_pct < TARGET_PLAY_MIN:   new_eth=min(ENT_THRESH_MAX, cur+0.034)
        elif play_pct > TARGET_PLAY_MAX: new_eth=max(ENT_THRESH_MIN, cur-0.034)
        else:                            new_eth=cur
        _set_entropy_threshold(new_eth)
        sim_dict["entropy_threshold"]=round(new_eth,3)
        with _lock:
            _sim_stats.update(sim_dict); _brake_left=brake
        cached=_build_cached_pred(rewards, records, brake)
        with _lock: _cached_pred=cached
        if cached and cached["_play"] and cached["_next_round"] is not None:
            stats_now=build_global_stats(rewards, order=4)
            _,_,_,_,_,_,_,mp=score_round(rewards, *stats_now)
            _pending_pred={
                "round":cached["_next_round"],"top2":cached["_top2"],"top3":cached["_top3"],
                "bonus_picks":cached["_bonus_picks"],"bonus_triggered":cached.get("_bonus_triggered",False),
                "pattern_scores":cached.get("_pattern_scores",{}),"any_boosted":cached.get("_any_boosted",False),
                "t1s":cached.get("_t1s",0.25),"markov_preds":mp,"penalty_forced":cached.get("_penalty_forced",False),
            }
            print(f"[Startup] Pending → #{_pending_pred['round']} top2={_pending_pred['top2']}")
        action=cached.get("action","N/A") if cached else "N/A"
        regime=(cached.get("reasoning",{}).get("regime",{}).get("mode","normal") if cached else "normal")
        print(f"[Startup] Done — acc={sim_dict.get('accuracy')}% brake={brake} "
              f"action={action} regime={regime} top1_t={round(_get_top1_threshold(),4)}")
    else:
        print("[Startup] No data file. Waiting for API fetch...")

    threading.Thread(target=fetcher_loop, daemon=True).start()
    print(f"[Startup] Fetcher started. IST: {now_ist.strftime('%H:%M:%S')}. "
          f"Schedule: cooldown 05:30-08:30, warmup 08:30-09:00, live 09:00+. Reset at 05:30 IST.")

startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
