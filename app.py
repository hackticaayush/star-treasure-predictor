import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# ─── CONFIG — exact same as simulate_v3.py ────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.220
SKIP_ENTROPY_THRESHOLD = 2.70
BRAKE_TRIGGER          = 3
BRAKE_PAUSE            = 3
TRAIN_ROUNDS           = 50
POLL_INTERVAL          = 5

CLASS_NAMES = {
    1:"Purple", 2:"10×", 3:"25×",
    4:"15×",   5:"Yellow", 6:"Lt. Green",
    7:"50×",   8:"Dk. Green"
}
CLASS_COLORS = {
    1:"#a855f7", 2:"#ef4444", 3:"#f97316",
    4:"#eab308", 5:"#facc15", 6:"#22c55e",
    7:"#06b6d4", 8:"#16a34a"
}

FETCH_BASE = "https://m.starmakerstudios.com/go-v1/ssc/2711/records?start_round="
FETCH_HEADERS = {
    'User-Agent': "sm/9.9.4/Android/13/google play/d48399ffafa2d343/wifi/en-IN/SM-M325F/10977524107285207///India",
    'Accept': "application/json, text/plain, */*",
    'Cookie': "PHPSESSID=pd6mapbqfhbk3e7argj51uh1ts; oauth_token=94le54aFnKy5CrbNzo7s903FOWniysVT"
}

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
_lock           = threading.Lock()
_rewards        = []        # oldest → newest (ascending by round)
_raw_rounds     = []        # full record objects
_sim_stats      = {}        # rebuilt after every fetch
_brake_left     = 0         # live brake counter (synced from simulation)

# ─── FILE HELPERS ─────────────────────────────────────────────────────────────
def _load_file():
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and data:
            return sorted(data, key=lambda x: x["round"])   # always ascending
    except Exception:
        pass
    return []

def _save_file(records):
    with open(DATA_FILE, "w") as f:
        json.dump(records, f, indent=2)

# ─── FETCHER ──────────────────────────────────────────────────────────────────
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
            print(f"[Fetcher] Error: {e}")
            break
        records = data.get("list", [])
        if not records:
            break
        fresh = []; overlap = False
        for r in records:
            rv = r.get("round")
            if rv in known:
                overlap = True; break
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

def fetcher_loop():
    while True:
        try:
            count, records = fetch_new()
            if count > 0:
                rewards = [r["reward_index"] for r in records]
                with _lock:
                    _raw_rounds.clear(); _raw_rounds.extend(records)
                    _rewards.clear();    _rewards.extend(rewards)
                _rebuild_sim()
                print(f"[Fetcher] +{count} new round(s). Latest: {records[-1]['round']}")
        except Exception as e:
            print(f"[Fetcher] Unexpected: {e}")
        time.sleep(POLL_INTERVAL)

# ─── MARKOV ENGINE — exact copy of simulate_v3.py ─────────────────────────────

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

def _build_global_stats(rewards):
    """Build prob, trans1-4, avg_gap, avg_run from full sequence.
    This is called on the FULL rewards list — same as simulate_v3 does
    at the top level before the simulation loop."""
    freq = Counter(rewards)
    prob = {k: v/len(rewards) for k, v in freq.items()}

    t1, tp1 = build_trans(rewards, 1)
    t2, tp2 = build_trans(rewards, 2)
    t3, tp3 = build_trans(rewards, 3)
    t4, tp4 = build_trans(rewards, 4)

    last_seen = {}; gaps = defaultdict(list)
    for i, r in enumerate(rewards):
        if r in last_seen: gaps[r].append(i - last_seen[r])
        last_seen[r] = i
    avg_gap = {k: sum(v)/len(v) if v else 8 for k, v in gaps.items()}

    rl = defaultdict(list); i = 0
    while i < len(rewards):
        j = i
        while j < len(rewards) and rewards[j] == rewards[i]: j += 1
        rl[rewards[i]].append(j - i); i = j
    avg_run = {k: sum(v)/len(v) for k, v in rl.items()}

    return prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap, avg_run


def score_round(history, prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap_g, avg_run_g):
    """Exact copy of simulate_v3's score_round — no changes."""
    n = len(history)
    if n < 4:
        ranked = sorted(prob.items(), key=lambda x: -x[1])
        scores = {k: 1/8 for k in range(1, 9)}
        return [ranked[0][0], ranked[1][0]], scores, 3.0, 0.125, 0.125

    l1, l2, l3, l4 = history[-1], history[-2], history[-3], history[-4]
    k1=(l1,); k2=(l2,l1); k3=(l3,l2,l1); k4=(l4,l3,l2,l1)

    def rel(t, key):
        return min(1.0, sum(t[key].values()) / 30) if key in t else 0

    r2=rel(t2,k2); r3=rel(t3,k3); r4=rel(t4,k4)

    WINDOW=100; WINDOW_SHORT=20
    recent  = history[-WINDOW:]       if n >= WINDOW       else history
    rshort  = history[-WINDOW_SHORT:] if n >= WINDOW_SHORT else history
    rec_cnt = Counter(recent); rs_cnt = Counter(rshort)
    rec_len = len(recent);     rs_len  = len(rshort)

    last_pos = {}
    for i2, r in enumerate(history): last_pos[r] = i2

    run_val = history[-1]; run_len = 1
    for i2 in range(n-2, -1, -1):
        if history[i2] == run_val: run_len += 1
        else: break

    gaps_h = defaultdict(list); lseen_h = {}
    for i2, r in enumerate(history):
        if r in lseen_h: gaps_h[r].append(i2 - lseen_h[r])
        lseen_h[r] = i2
    avg_gap_h = {r: sum(gaps_h[r])/len(gaps_h[r]) if gaps_h.get(r) else avg_gap_g.get(r, 8)
                 for r in range(1, 9)}

    rl_h = defaultdict(list); i2 = 0
    while i2 < n:
        j = i2
        while j < n and history[j] == history[i2]: j += 1
        rl_h[history[i2]].append(j - i2); i2 = j
    avg_run_h = {r: sum(rl_h[r])/len(rl_h[r]) if rl_h.get(r) else avg_run_g.get(r, 1.5)
                 for r in range(1, 9)}

    score = {}
    for idx in range(1, 9):
        base  = prob.get(idx, 0)
        m1    = tp1.get(k1, {}).get(idx, base)
        m2    = tp2.get(k2, {}).get(idx, m1)  if r2 > 0 else m1
        m3    = tp3.get(k3, {}).get(idx, m2)  if r3 > 0 else m2
        m4    = tp4.get(k4, {}).get(idx, m3)  if r4 > 0 else m3
        rec_p  = rec_cnt.get(idx, 0) / rec_len
        recs_p = rs_cnt.get(idx, 0)  / rs_len
        pos    = last_pos.get(idx, 0); ag = avg_gap_h.get(idx, 8)
        od     = (n - 1 - pos) / ag if ag else 0
        od_boost    = min(0.5, max(0.0, (od - 1.0) * 0.1))
        run_penalty = 0.05 if idx == run_val and run_len >= avg_run_h.get(idx, 1.5) else 0
        w_base=0.05; w_m1=0.10; w_m2=0.15*r2; w_m3=0.25*r3; w_m4=0.20*r4
        w_rec=0.10;  w_vs=0.10; w_od=0.05
        total_w = w_base+w_m1+w_m2+w_m3+w_m4+w_rec+w_vs+w_od or 1
        raw = (w_base*base + w_m1*m1 + w_m2*m2 + w_m3*m3 + w_m4*m4
               + w_rec*rec_p + w_vs*recs_p + w_od*od_boost)
        score[idx] = max(0.0, raw / total_w - run_penalty)

    total_s = sum(score.values()) or 1
    score   = {k: v/total_s for k, v in score.items()}
    ranked  = sorted(score.items(), key=lambda x: -x[1])
    entropy    = -sum(v * math.log2(v) for v in score.values() if v > 0)
    top1_score = ranked[0][1]
    top2_score = ranked[1][1]
    return [ranked[0][0], ranked[1][0]], score, entropy, top1_score, top2_score


def should_play(top1_score, entropy, brake_active=False):
    """Exact copy of simulate_v3's should_play."""
    if brake_active:
        return False
    return top1_score > SKIP_TOP1_THRESHOLD and entropy < SKIP_ENTROPY_THRESHOLD


# ─── SIMULATION — exact walk-forward logic from simulate_v3.py ───────────────
def _rebuild_sim():
    """
    This is a line-for-line port of simulate_v3's main loop.
    Key difference from old app.py:
      - NO stats caching (recomputes prob/trans on every round — correct but slower)
      - Global stats are built from the FULL rewards list at start
      - Brake logic is identical
      - recent_played stores ONLY played rounds with correct hit/miss
    """
    global _sim_stats, _brake_left

    with _lock:
        rewards = list(_rewards)

    total = len(rewards)
    if total < TRAIN_ROUNDS + 5:
        return

    # Build global stats from full dataset — same as simulate_v3 top-level
    prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap_g, avg_run_g = \
        _build_global_stats(rewards)

    hits = misses = skipped_total = played_total = 0
    loss_streak = max_loss_streak = win_streak = max_win_streak = 0
    all_loss_streaks = []
    brake_remaining = 0
    recent_played = []   # only PLAYED rounds, with correct hit/miss

    skip_ent_count = skip_top1_count = skip_brake_count = 0

    for i in range(TRAIN_ROUNDS, total - 1):
        history   = rewards[:i+1]
        true_next = rewards[i+1]

        # ── score_round uses history slice + global stats ──
        # (same as simulate_v3 which uses global prob/trans built at top)
        top2, scores, entropy, top1_s, top2_s = score_round(
            history, prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap_g, avg_run_g
        )

        brake_active = brake_remaining > 0
        if brake_remaining > 0:
            brake_remaining -= 1

        play = should_play(top1_s, entropy, brake_active)

        if not play:
            skipped_total += 1
            if brake_active:
                skip_brake_count += 1
            elif entropy >= SKIP_ENTROPY_THRESHOLD:
                skip_ent_count += 1
            else:
                skip_top1_count += 1
            continue   # ← skipped rounds do NOT go into recent_played

        # ── PLAYED round ──
        played_total += 1
        hit = true_next in top2   # true_next is rewards[i+1] — always correct

        if hit:
            hits += 1
            if loss_streak > 0:
                all_loss_streaks.append(loss_streak)
            loss_streak = 0
            win_streak += 1
            max_win_streak = max(max_win_streak, win_streak)
        else:
            misses += 1
            loss_streak += 1
            max_loss_streak = max(max_loss_streak, loss_streak)
            win_streak = 0
            if loss_streak >= BRAKE_TRIGGER:
                brake_remaining = BRAKE_PAUSE

        recent_played.append({
            "round":  i + 1,       # dataset index (not the API round number)
            "top2":   top2,
            "actual": true_next,   # the REAL next value
            "hit":    hit,
            "loss_streak": loss_streak,
        })

    if loss_streak > 0:
        all_loss_streaks.append(loss_streak)

    sim_total = total - TRAIN_ROUNDS - 1
    acc       = hits / played_total * 100 if played_total else 0
    play_pct  = played_total / sim_total * 100 if sim_total else 0

    with _lock:
        _sim_stats = {
            "total":          total,
            "sim_total":      sim_total,
            "played":         played_total,
            "skipped":        skipped_total,
            "hits":           hits,
            "misses":         misses,
            "accuracy":       round(acc, 2),
            "play_pct":       round(play_pct, 1),
            "max_loss":       max_loss_streak,
            "max_win":        max_win_streak,
            "skip_brake":     skip_brake_count,
            "skip_entropy":   skip_ent_count,
            "skip_top1":      skip_top1_count,
            "recent_played":  recent_played[-20:],  # last 20 played rounds
        }
        _brake_left = brake_remaining   # sync live brake state


# ─── LIVE PREDICTION ──────────────────────────────────────────────────────────
def get_prediction():
    with _lock:
        rewards = list(_rewards)
        raw     = list(_raw_rounds)
        brake   = _brake_left

    if len(rewards) < TRAIN_ROUNDS + 5:
        return None

    # Build stats from full sequence — same as simulate_v3
    prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap_g, avg_run_g = \
        _build_global_stats(rewards)

    top2, scores, entropy, t1s, t2s = score_round(
        rewards, prob, t1, tp1, t2, tp2, t3, tp3, t4, tp4, avg_gap_g, avg_run_g
    )

    play = should_play(t1s, entropy, brake > 0)

    skip_reason = None
    if brake > 0:
        skip_reason = f"Loss brake ({brake} rounds left)"
    elif entropy >= SKIP_ENTROPY_THRESHOLD:
        skip_reason = f"High entropy ({entropy:.3f} ≥ {SKIP_ENTROPY_THRESHOLD})"
    elif t1s <= SKIP_TOP1_THRESHOLD:
        skip_reason = f"Low confidence ({t1s:.3f} ≤ {SKIP_TOP1_THRESHOLD})"

    last_round = raw[-1]["round"] if raw else None

    return {
        "next_round":   (last_round + 1) if last_round else None,
        "latest_round": last_round,
        "pred1":        top2[0],
        "pred2":        top2[1],
        "pred1_name":   CLASS_NAMES.get(top2[0], "?"),
        "pred2_name":   CLASS_NAMES.get(top2[1], "?"),
        "pred1_color":  CLASS_COLORS.get(top2[0], "#888"),
        "pred2_color":  CLASS_COLORS.get(top2[1], "#888"),
        "pred1_conf":   round(t1s * 100, 2),
        "pred2_conf":   round(t2s * 100, 2),
        "entropy":      round(entropy, 4),
        "action":       "PLAY" if play else "SKIP",
        "skip_reason":  skip_reason,
        "all_scores":   {k: round(v * 100, 2) for k, v in scores.items()},
        "last_10":      rewards[-10:],
        "total_rounds": len(rewards),
    }


# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", class_names=CLASS_NAMES, class_colors=CLASS_COLORS)

@app.route("/api/predict")
def api_predict():
    pred = get_prediction()
    if pred is None:
        return jsonify({"error": "Not enough data yet"}), 503
    return jsonify(pred)

@app.route("/api/stats")
def api_stats():
    with _lock:
        stats = dict(_sim_stats)
    return jsonify(stats)

@app.route("/api/history")
def api_history():
    with _lock:
        raw = list(_raw_rounds[-50:])
    return jsonify([{
        "round":        r["round"],
        "reward_index": r["reward_index"],
        "name":         CLASS_NAMES.get(r["reward_index"], "?"),
        "color":        CLASS_COLORS.get(r["reward_index"], "#888")
    } for r in reversed(raw)])


# ─── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    records = _load_file()
    if records:
        rewards = [r["reward_index"] for r in records]   # ascending — oldest→newest
        with _lock:
            _raw_rounds.extend(records)
            _rewards.extend(rewards)
        print(f"[Startup] Loaded {len(records)} rounds. Building simulation...")
        _rebuild_sim()
        with _lock:
            s = dict(_sim_stats)
        print(f"[Startup] Sim done — played={s.get('played')}, "
              f"acc={s.get('accuracy')}%, maxloss={s.get('max_loss')}")
    t = threading.Thread(target=fetcher_loop, daemon=True)
    t.start()
    print("[Startup] Fetcher thread started.")

startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
