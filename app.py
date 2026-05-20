import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template, Response

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.220
SKIP_ENTROPY_THRESHOLD = 2.70
BRAKE_TRIGGER          = 3
BRAKE_PAUSE            = 3
TRAIN_ROUNDS           = 50
POLL_INTERVAL          = 5

HIGH_MULT_CLASSES      = {2, 3, 4, 7}

TARGET_PLAY_MIN  = 0.38
TARGET_PLAY_MAX  = 0.45
ENT_THRESH_MIN   = 2.65
ENT_THRESH_MAX   = 2.90

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
# Base weights for distance-decay scoring
PATTERN_BASE_WEIGHTS = {
    7: 3.0,   # 50x
    3: 2.0,   # 25x
    4: 1.5,   # 15x
    2: 1.0,   # 10x
    1: 1.0,   # Purple
    5: 1.0,   # Yellow
    6: 1.0,   # Lt. Green
    8: 1.0,   # Dk. Green
}

# 5 pattern groups
PATTERN_GROUPS = {
    "high_mult": {"classes": {2, 3, 4, 7}, "default_threshold": 5.0},
    "cls1":      {"classes": {1},           "default_threshold": 4.0},
    "cls5":      {"classes": {5},           "default_threshold": 4.0},
    "cls6":      {"classes": {6},           "default_threshold": 4.0},
    "cls8":      {"classes": {8},           "default_threshold": 4.0},
}

PATTERN_LOOKBACK     = 10          # look at last 10 rounds
PATTERN_DECAY        = 0.5         # 50% decay per distance step
PATTERN_HIT_BUFFER   = 100         # rolling buffer of last 100 hits per group
PATTERN_BOOST_MIN    = 0.15        # minimum boost when threshold exceeded
PATTERN_BOOST_MAX    = 0.35        # maximum boost cap
PATTERN_BOOST_SCALE  = 0.04        # how fast boost grows above threshold

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
_lock              = threading.Lock()
_rewards           = []
_raw_rounds        = []
_sim_stats         = {}
_brake_left        = 0
_live_log          = []
_pending_pred      = None
_entropy_threshold = SKIP_ENTROPY_THRESHOLD
_last_reset_date   = None

_cached_pred = None

_fetch_status = {
    "last_attempt": None, "last_success": None,
    "last_error":   None, "total_fetched": 0,
    "status": "starting", "last_reset": None,
}

# ── PATTERN STATE (new) ───────────────────────────────────────────────────────
# Rolling buffer of pattern scores at time of confirmed HIT, per group
_hit_pattern_scores = {g: [] for g in PATTERN_GROUPS}
# Dynamic threshold per group (starts at default, learned from hits)
_dynamic_threshold  = {g: PATTERN_GROUPS[g]["default_threshold"] for g in PATTERN_GROUPS}
# Last computed pattern scores (for logging / API)
_last_pattern_info  = {}

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
        # Reset pattern state
        for g in PATTERN_GROUPS:
            _hit_pattern_scores[g].clear()
            _dynamic_threshold[g] = PATTERN_GROUPS[g]["default_threshold"]
        _last_pattern_info.clear()
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

# ── PATTERN DETECTOR (new) ────────────────────────────────────────────────────
def compute_pattern_scores(rewards):
    """
    For each of the 5 pattern groups, compute a distance-decayed score
    over the last PATTERN_LOOKBACK rounds.

    contribution = base_weight × (PATTERN_DECAY ^ (distance - 1))
    distance 1 = most recent round

    Returns dict: {group_name: score}
    """
    if len(rewards) < PATTERN_LOOKBACK:
        return {g: 0.0 for g in PATTERN_GROUPS}

    last10 = rewards[-PATTERN_LOOKBACK:]  # oldest → newest
    scores = {g: 0.0 for g in PATTERN_GROUPS}

    for dist_from_end, cls in enumerate(reversed(last10)):
        distance = dist_from_end + 1  # 1 = most recent
        decay    = PATTERN_DECAY ** (distance - 1)
        weight   = PATTERN_BASE_WEIGHTS.get(cls, 1.0)
        contrib  = weight * decay

        for group_name, group_cfg in PATTERN_GROUPS.items():
            if cls in group_cfg["classes"]:
                scores[group_name] += contrib

    return scores


def apply_pattern_boost(scores_dict, pattern_scores, dyn_thresholds):
    """
    For each pattern group, if pattern_score >= dynamic_threshold,
    boost the classes in that group proportionally, then renormalize.

    scores_dict: {cls_int: float}  — already normalized to sum=1
    Returns: (boosted_scores, pattern_info)
    """
    boosted = dict(scores_dict)
    pattern_info = {}

    for group_name, group_cfg in PATTERN_GROUPS.items():
        pscore    = pattern_scores.get(group_name, 0.0)
        threshold = dyn_thresholds.get(group_name, group_cfg["default_threshold"])
        triggered = pscore >= threshold

        boost_applied = 0.0
        if triggered:
            excess        = pscore - threshold
            boost_applied = min(PATTERN_BOOST_MAX,
                                PATTERN_BOOST_MIN + excess * PATTERN_BOOST_SCALE)
            for cls in group_cfg["classes"]:
                if cls in boosted:
                    boosted[cls] += boosted[cls] * boost_applied
                else:
                    boosted[cls] = boost_applied * 0.01  # tiny seed if absent

        pattern_info[group_name] = {
            "score":         round(pscore, 4),
            "threshold":     round(threshold, 4),
            "triggered":     triggered,
            "boost_applied": round(boost_applied, 4),
        }

    # Renormalize to sum = 1
    total = sum(boosted.values()) or 1.0
    boosted = {k: v / total for k, v in boosted.items()}

    return boosted, pattern_info


def update_pattern_hit(group_name, pattern_score):
    """
    Called when a prediction HIT is confirmed.
    Adds the pattern_score to the rolling buffer and updates dynamic threshold.
    """
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


# ── SCORE ROUND (updated) ─────────────────────────────────────────────────────
def score_round(h, prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,ag,ar):
    n=len(h)
    if n<4:
        ranked=sorted(prob.items(),key=lambda x:-x[1])
        return [ranked[0][0],ranked[1][0]],{k:1/8 for k in range(1,9)},3.0,0.125,0.125,0.125,ranked[2][0] if len(ranked)>=3 else None

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
        wb=0.05;wm1=0.10;wm2=0.15*r2;wm3=0.25*r3;wm4=0.20*r4;wr=0.10;wv=0.10;wo=0.05
        tw=wb+wm1+wm2+wm3+wm4+wr+wv+wo or 1
        raw=(wb*base+wm1*m1+wm2*m2+wm3*m3+wm4*m4+wr*rp+wv*rsp+wo*ob)
        sc[idx]=max(0.0,raw/tw-rpen)
    ts=sum(sc.values()) or 1
    sc={k:v/ts for k,v in sc.items()}

    # ── PATTERN BOOST (new) ──────────────────────────────────────────────────
    with _lock:
        dyn_thresh = dict(_dynamic_threshold)

    pattern_scores = compute_pattern_scores(h)
    sc, pattern_info = apply_pattern_boost(sc, pattern_scores, dyn_thresh)

    with _lock:
        _last_pattern_info.clear()
        _last_pattern_info.update(pattern_info)
        _last_pattern_info["raw_scores"] = {g: round(v, 4) for g, v in pattern_scores.items()}
    # ────────────────────────────────────────────────────────────────────────

    rk=sorted(sc.items(),key=lambda x:-x[1])
    ent=-sum(v*math.log2(v) for v in sc.values() if v>0)
    # return top3 scores too so caller can decide on 3rd pick
    t3s = rk[2][1] if len(rk) >= 3 else 0.0
    t3c = rk[2][0] if len(rk) >= 3 else None
    return [rk[0][0],rk[1][0]],sc,ent,rk[0][1],rk[1][1],t3s,t3c

def should_play(t1, ent, brake_active=False):
    if brake_active: return False
    return t1 > SKIP_TOP1_THRESHOLD and ent < _entropy_threshold

# ── BONUS PICK LOGIC ──────────────────────────────────────────────────────────
def get_bonus_picks(scores, top2):
    ranked     = sorted(scores.items(), key=lambda x: -x[1])
    ranked_ids = [int(k) for k, _ in ranked]
    in_top4    = bool(set(ranked_ids[:4]).intersection(HIGH_MULT_CLASSES))
    above_10   = any(scores.get(k, scores.get(str(k), 0)) > 0.10 for k in HIGH_MULT_CLASSES)
    if not in_top4 and not above_10:
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
    if total < TRAIN_ROUNDS + 5:
        return {}, 0, 0.0
    stats = build_global_stats(rewards)
    hits=misses=sk=pl=0; ls=mx=ws=mw=0; brake=0; se=st=sb=0
    for i in range(TRAIN_ROUNDS, total - 1):
        h=rewards[:i+1]; tn=rewards[i+1]
        top2,_,ent,t1s,_,_,_=score_round(h,*stats)
        ba=brake>0
        if brake>0: brake-=1
        play=should_play(t1s,ent,ba)
        if not play:
            sk+=1
            if ba: sb+=1
            elif ent>=SKIP_ENTROPY_THRESHOLD: se+=1
            else: st+=1
            continue
        pl+=1; hit=tn in top2
        if hit: hits+=1;ls=0;ws+=1;mw=max(mw,ws)
        else: misses+=1;ls+=1;mx=max(mx,ls);ws=0
        if ls>=BRAKE_TRIGGER: brake=BRAKE_PAUSE
    sim_total = total - TRAIN_ROUNDS - 1
    acc       = hits / pl * 100 if pl else 0
    play_pct  = pl / sim_total if sim_total else 0.0
    return {
        "total":total,"sim_total":sim_total,
        "played":pl,"skipped":sk,"hits":hits,"misses":misses,
        "accuracy":round(acc,2),"play_pct":round(play_pct*100,1),
        "max_loss":mx,"max_win":mw,
        "skip_brake":sb,"skip_entropy":se,"skip_top1":st,
    }, brake, play_pct

# ── BUILD CACHED PRED ─────────────────────────────────────────────────────────
def _build_cached_pred(rewards, raw_rounds, brake):
    if len(rewards) < TRAIN_ROUNDS + 5:
        return None
    stats = build_global_stats(rewards)
    top2, scores, ent, t1s, t2s, t3s, t3c = score_round(rewards, *stats)
    eth  = _entropy_threshold
    play = t1s > SKIP_TOP1_THRESHOLD and ent < eth and brake == 0

    skip_reason = None
    if brake > 0:                    skip_reason = f"Loss brake ({brake} rounds left)"
    elif ent >= eth:                 skip_reason = f"High entropy ({ent:.4f} ≥ {eth:.3f})"
    elif t1s <= SKIP_TOP1_THRESHOLD: skip_reason = f"Low confidence ({t1s:.4f})"

    last_round = raw_rounds[-1]["round"] if raw_rounds else None
    next_round = (last_round + 1) if last_round else None

    # ── 3rd PICK LOGIC ────────────────────────────────────────────────────────
    # If top2 are both small-mult classes AND top3 conf is within 1% of top2,
    # include a 3rd suggestion.
    SMALL_MULT = {1, 5, 6, 8}
    pred3       = None
    pred3_conf  = None
    top2_are_small = all(c in SMALL_MULT for c in top2)
    if (play and top2_are_small and t3c is not None
            and (t2s - t3s) <= 0.01):   # within 1 percentage point
        pred3      = t3c
        pred3_conf = round(t3s * 100, 2)
    # ─────────────────────────────────────────────────────────────────────────

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

    # Snapshot pattern info for API
    with _lock:
        pattern_snap = dict(_last_pattern_info)
        dyn_thresh_snap = dict(_dynamic_threshold)

    pattern_summary = {}
    for g, info in pattern_snap.items():
        if g == "raw_scores":
            continue
        if isinstance(info, dict):
            pattern_summary[g] = {
                "score":         info.get("score", 0.0),
                "threshold":     info.get("threshold", 0.0),
                "triggered":     info.get("triggered", False),
                "boost_applied": info.get("boost_applied", 0.0),
            }

    return {
        "next_round":        next_round,  "latest_round": last_round,
        "pred1":             top2[0],     "pred2":        top2[1],
        "pred3":             pred3,
        "pred1_name":        CLASS_NAMES.get(top2[0],"?"),
        "pred2_name":        CLASS_NAMES.get(top2[1],"?"),
        "pred3_name":        CLASS_NAMES.get(pred3,"?") if pred3 else None,
        "pred1_color":       CLASS_COLORS.get(top2[0],"#888"),
        "pred2_color":       CLASS_COLORS.get(top2[1],"#888"),
        "pred3_color":       CLASS_COLORS.get(pred3,"#888") if pred3 else None,
        "pred1_conf":        round(t1s*100,2),
        "pred2_conf":        round(t2s*100,2),
        "pred3_conf":        pred3_conf,
        "entropy":           round(ent,4),
        "action":            "PLAY" if play else "SKIP",
        "skip_reason":       skip_reason,
        "bonus_picks":       bonus_details,
        "all_scores":        {k: round(v*100,2) for k,v in scores.items()},
        "last_10":           rewards[-10:],
        "total_rounds":      len(rewards),
        "pattern_info":      pattern_summary,
        "dynamic_thresholds": {g: round(v, 4) for g, v in dyn_thresh_snap.items()},
        "_play":             play,
        "_top2":             list(top2),
        "_top3":             ([top2[0], top2[1], pred3] if pred3 else list(top2)),
        "_bonus_picks":      list(bonus_picks) if bonus_picks else None,
        "_next_round":       next_round,
        "_pattern_scores":   {g: pattern_snap.get(g, {}).get("score", 0.0)
                              for g in PATTERN_GROUPS},
    }

# ── FETCHER LOOP ──────────────────────────────────────────────────────────────
def fetcher_loop():
    global _pending_pred, _cached_pred, _entropy_threshold, _brake_left

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
                        actual_val    = actual_rec["reward_index"]
                        # use top3 if available, else top2
                        all_preds     = list(pending.get("top3", pending["top2"]))
                        if pending.get("bonus_picks"):
                            all_preds += [p for p in pending["bonus_picks"] if p not in all_preds]
                        hit = actual_val in all_preds

                        # ── UPDATE PATTERN THRESHOLDS ON HIT (new) ──────────
                        if hit and pending.get("pattern_scores"):
                            for group_name, pscore in pending["pattern_scores"].items():
                                update_pattern_hit(group_name, pscore)
                        # ────────────────────────────────────────────────────

                        entry = {
                            "round": pred_round, "top2": pending["top2"],
                            "top3": pending.get("top3"),
                            "bonus_picks": pending.get("bonus_picks"),
                            "actual": actual_val, "hit": hit, "action": "PLAY",
                        }
                        with _lock:
                            _live_log.append(entry)
                            _pending_pred = None
                        pending = None
                        print(f"[Live] #{pred_round}: pred={entry['top2']} "
                              f"pred3={entry.get('top3')} "
                              f"bonus={entry['bonus_picks']} "
                              f"actual={actual_val} → {'HIT ✓' if hit else 'MISS ✗'}")
                    elif records and pred_round <= records[-1]["round"]:
                        with _lock:
                            _pending_pred = None
                        pending = None
                        print(f"[Fetcher] Stale pending cleared (#{pred_round})")

                sim_dict, brake, play_pct = _run_sim(rewards)

                cur = _entropy_threshold
                if play_pct < TARGET_PLAY_MIN:   new_eth = min(ENT_THRESH_MAX, cur + 0.034)
                elif play_pct > TARGET_PLAY_MAX: new_eth = max(ENT_THRESH_MIN, cur - 0.034)
                else:                            new_eth = cur
                if abs(new_eth - cur) > 0.001:
                    print(f"[Adaptive] Entropy: {cur:.3f}→{new_eth:.3f} (play%={play_pct*100:.1f}%)")
                _entropy_threshold = new_eth
                sim_dict["entropy_threshold"] = round(new_eth, 3)

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
                        np_ = {
                            "round":          nr,
                            "top2":           cached["_top2"],
                            "top3":           cached["_top3"],
                            "bonus_picks":    cached["_bonus_picks"],
                            "pattern_scores": cached.get("_pattern_scores", {}),
                        }
                        with _lock:
                            _pending_pred = np_
                        print(f"[Fetcher] Pending → #{nr} top2={cached['_top2']} "
                              f"bonus={cached['_bonus_picks']} "
                              f"pattern_scores={cached.get('_pattern_scores',{})}")

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
    with open(sw_path, "r") as f:
        content = f.read()
    return Response(
        content,
        mimetype="application/javascript",
        headers={
            "Service-Worker-Allowed": "/",
            "Cache-Control": "no-cache, no-store, must-revalidate",
        }
    )

@app.route("/pwa/manifest.json")
def serve_manifest():
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    with open(manifest_path, "r") as f:
        content = f.read()
    return Response(
        content,
        mimetype="application/manifest+json",
        headers={"Cache-Control": "no-cache"},
    )

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
    cur = 0; mx_live = 0
    for e in reversed(live):
        if not e["hit"]: cur += 1; mx_live = max(mx_live, cur)
        else: break
    stats["live_log"]            = live
    stats["live_cur_streak"]     = cur
    stats["live_max_loss"]       = mx_live
    stats["dynamic_thresholds"]  = {g: round(v, 4) for g, v in dyn_t.items()}
    stats["pattern_hit_counts"]  = hit_bufs
    return jsonify(stats)

@app.route("/api/status")
def api_status():
    with _lock:
        fs     = dict(_fetch_status)
        total  = len(_rewards)
        latest = _raw_rounds[-1]["round"] if _raw_rounds else None
    now_ist = datetime.now(IST)
    fs["total_rounds"]      = total
    fs["latest_round"]      = latest
    fs["server_time_ist"]   = now_ist.strftime("%H:%M:%S IST")
    fs["entropy_threshold"] = round(_entropy_threshold, 3)
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
    """New endpoint — live pattern scores and dynamic thresholds."""
    with _lock:
        pattern_info = dict(_last_pattern_info)
        dyn_t        = dict(_dynamic_threshold)
        hit_bufs     = {g: list(v) for g, v in _hit_pattern_scores.items()}
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
    return jsonify(result)

# ── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    global _last_reset_date, _pending_pred, _cached_pred, _entropy_threshold, _brake_left

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
        print(f"[Startup] Loaded {len(records)} rounds. Building sim...")

        sim_dict, brake, play_pct = _run_sim(rewards)
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
            _pending_pred = {
                "round":          cached["_next_round"],
                "top2":           cached["_top2"],
                "top3":           cached["_top3"],
                "bonus_picks":    cached["_bonus_picks"],
                "pattern_scores": cached.get("_pattern_scores", {}),
            }
            print(f"[Startup] Pending → #{_pending_pred['round']} top2={_pending_pred['top2']} top3={_pending_pred['top3']}")

        action = cached.get("action","N/A") if cached else "N/A"
        print(f"[Startup] Done — acc={sim_dict.get('accuracy')}% brake={brake} action={action}")
    else:
        print("[Startup] No data file. Waiting for API fetch...")

    threading.Thread(target=fetcher_loop, daemon=True).start()
    print(f"[Startup] Fetcher started. IST: {now_ist.strftime('%H:%M:%S')}. Reset at 05:30 IST.")

startup()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
