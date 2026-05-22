import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.180
SKIP_ENTROPY_THRESHOLD = 2.50
BRAKE_TRIGGER          = 2
BRAKE_PAUSE            = 2
POLL_INTERVAL          = 5

HIGH_MULT_CLASSES      = {2, 3, 4, 7}

# EV multipliers for bonus inclusion
HIGH_MULT_EV        = {2: 10, 3: 19, 4: 13, 7: 30}
HIGH_MULT_EV_TARGET = 1.5

# Minimum score floor for high-mult classes
HIGH_MULT_EV_FLOOR = {2: 0.055, 3: 0.030, 4: 0.045, 7: 0.018}

# Entropy adaptation targets
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

# ── COOLDOWN / WARMUP / LIVE SCHEDULE ─────────────────────────────────────────
COOLDOWN_END_HOUR   = 8
COOLDOWN_END_MINUTE = 30
LIVE_START_HOUR     = 16
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

# ══════════════════════════════════════════════════════════════════════════════
#  ███████╗███╗   ███╗ █████╗ ██████╗ ████████╗    ██████╗ ███████╗ █████╗ ███████╗ ██████╗ ███╗   ██╗███████╗██████╗
#  ██╔════╝████╗ ████║██╔══██╗██╔══██╗╚══██╔══╝    ██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗  ██║██╔════╝██╔══██╗
#  ███████╗██╔████╔██║███████║██████╔╝   ██║       ██████╔╝█████╗  ███████║███████╗██║   ██║██╔██╗ ██║█████╗  ██████╔╝
#  ╚════██║██║╚██╔╝██║██╔══██║██╔══██╗   ██║       ██╔══██╗██╔══╝  ██╔══██║╚════██║██║   ██║██║╚██╗██║██╔══╝  ██╔══██╗
#  ███████║██║ ╚═╝ ██║██║  ██║██║  ██║   ██║       ██║  ██║███████╗██║  ██║███████║╚██████╔╝██║ ╚████║███████╗██║  ██║
#  ╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝       ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝
#
#  SMART REASONER MODULE — Pattern Memory + Regime Detector + Anti-pattern Shift
# ══════════════════════════════════════════════════════════════════════════════

# ── REASONER CONFIG ───────────────────────────────────────────────────────────
REASONER_FINGERPRINT_LEN   = 6      # last N rounds ka fingerprint banao
REASONER_MATCH_MIN_LEN     = 4      # minimum matching prefix to count as "similar"
REASONER_HISTORY_LOOKBACK  = 500    # kitne purane rounds mein dhundho
REASONER_MATCH_WEIGHT      = 0.18   # fingerprint match ka score adjustment weight
REASONER_MAX_MATCHES       = 15     # kitne matches consider karo

# Regime detection
REGIME_WINDOW_SHORT        = 12     # short window for recent behavior
REGIME_WINDOW_LONG         = 50     # long window for baseline
REGIME_HIT_DIFF_THRESHOLD  = 0.20   # hit rate diff > this = weird regime
REGIME_ENTROPY_SPIKE       = 0.45   # entropy diff > this = weird regime
REGIME_DIST_THRESHOLD      = 0.30   # class distribution drift > this = weird regime
REGIME_DIVERSIFY_STRENGTH  = 0.12   # kitna diversify karo weird regime mein

# Anti-pattern / adversarial shift
ANTI_PATTERN_WINDOW        = 8      # last N PLAY rounds check karo
ANTI_PATTERN_MISS_THRESH   = 6      # agar itne miss ho gayein is window mein
ANTI_PATTERN_SHIFT_WEIGHT  = 0.22   # kitna shift karo alternatives ki taraf
ANTI_PATTERN_AVOID_TOP     = 2      # top N predicted classes ko penalize karo

# Reasoner final blend
REASONER_BLEND_WEIGHT      = 0.28   # reasoner ka overall influence on final scores

# ── REASONER STATE ────────────────────────────────────────────────────────────
_reasoner_lock          = threading.Lock()
_reasoner_last_info     = {}        # last reasoning explanation
_reasoner_regime_log    = []        # (timestamp, regime_type) log
_reasoner_anti_active   = False     # anti-pattern mode active?
_reasoner_anti_streak   = 0        # consecutive play-misses count


# ══════════════════════════════════════════════════════════════════════════════
#  PATTERN MEMORY ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _get_fingerprint(rewards, length=REASONER_FINGERPRINT_LEN):
    """Last N rounds ka tuple fingerprint."""
    if len(rewards) < length:
        return tuple(rewards)
    return tuple(rewards[-length:])


def _fingerprint_similarity(fp1, fp2):
    """
    Do fingerprints kitne similar hain 0.0-1.0 scale pe.
    Tail-match: ending match zyada important hai.
    """
    if not fp1 or not fp2:
        return 0.0
    min_len = min(len(fp1), len(fp2))
    score = 0.0
    total_weight = 0.0
    for i in range(min_len):
        # Last elements zyada weight — recency matters
        pos_weight = (i + 1) / min_len
        total_weight += pos_weight
        if fp1[-(i+1)] == fp2[-(i+1)]:
            score += pos_weight
        else:
            # Ek mismatch ke baad aage mat dekho (strict tail match)
            break
    return score / total_weight if total_weight > 0 else 0.0


def _pattern_memory_lookup(rewards):
    """
    Current fingerprint jaisa pattern pehle kab kab aaya,
    aur uss ke baad kya aaya — woh return karo.
    Returns: dict of {class: weighted_count}, explanation string
    """
    if len(rewards) < REASONER_FINGERPRINT_LEN + 1:
        return {}, "Not enough data for pattern memory"

    current_fp = _get_fingerprint(
