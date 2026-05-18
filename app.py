import json, math, threading, time, os, requests
from collections import defaultdict, Counter
from datetime import datetime
import pytz
from flask import Flask, jsonify, render_template

app = Flask(__name__)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_FILE              = "round_data.json"
SKIP_TOP1_THRESHOLD    = 0.220
SKIP_ENTROPY_THRESHOLD = 2.70   # base value — gets adapted dynamically
BRAKE_TRIGGER          = 3
BRAKE_PAUSE            = 3
TRAIN_ROUNDS           = 50
POLL_INTERVAL          = 5

# Adaptive threshold config
# Target: play 38-45% of rounds with best possible accuracy
# The threshold auto-adjusts after each simulation rebuild
TARGET_PLAY_MIN  = 0.38   # min play rate we want
TARGET_PLAY_MAX  = 0.45   # max play rate we want
ENT_THRESH_MIN   = 2.65   # never go below this (accuracy drops)
ENT_THRESH_MAX   = 2.90   # never go above this (too many bad rounds played)

# Daily reset: every day at 05:30 AM IST (UTC+5:30)
# Round 1 starts fresh at this time, so we wipe old JSON and start clean
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

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
_lock              = threading.Lock()
_rewards           = []       # ascending: oldest→newest
_raw_rounds        = []
_sim_stats         = {}
_brake_left        = 0
_live_log          = []       # ALL played rounds since server start, no cap
_pending_pred      = None
_entropy_threshold = SKIP_ENTROPY_THRESHOLD  # starts at 2.70, adapts over time
_fetch_status = {
    "last_attempt": None, "last_success": None,
    "last_error": None,   "total_fetched": 0,
    "status": "starting", "last_reset": None,
}
_last_reset_date = None  # tracks which calendar date we last reset on

# ── DAILY RESET LOGIC ─────────────────────────────────────────────────────────
def _should_reset():
    """
    Returns True if current IST time >= 05:30 AM and we haven't
    already reset today (i.e. _last_reset_date != today IST date).
    """
    global _last_reset_date
    now_ist  = datetime.now(IST)
    today    = now_ist.date()
    past_530 = (now_ist.hour > RESET_HOUR_IST or
                (now_ist.hour == RESET_HOUR_IST and now_ist.minute >= RESET_MINUTE_IST))

    if past_530 and _last_reset_date != today:
        return True
    return False

def _do_reset():
    """Wipe JSON file and all in-memory data. New day, new rounds."""
    global _last_reset_date, _pending_pred
    now_ist = datetime.now(IST)
    today   = now_ist.date()

    print(f"[Reset] 5:30 AM IST reached — wiping data for new day ({today})")

    # Delete JSON file
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        print(f"[Reset] {DATA_FILE} deleted")

    # Clear all in-memory state
    with _lock:
        _rewards.clear()
        _raw_rounds.clear()
        _sim_stats.clear()
        _live_log.clear()
        _pending_pred   = None
        _fetch_status["last_reset"] = now_ist.strftime("%d %b %Y %H:%M IST")
        _fetch_status["status"]     = "reset_done"

    _last_reset_date = today
    print(f"[Reset] All data cleared. Waiting for new rounds from API...")

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
                err = "AUTH FAILED (403) — Cookie expired! Update FETCH_HEADERS in app.py"
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

def fetcher_loop():
    global _pending_pred
    while True:
        try:
            # ── Check daily reset FIRST ──────────────────────────────────────
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

                # Resolve pending PLAY prediction
                if pending is not None:
                    pred_round = pending["round"]
                    actual_rec = next((r for r in records if r["round"] == pred_round), None)
                    if actual_rec is not None:
                        actual_val = actual_rec["reward_index"]
                        hit        = actual_val in pending["top2"]
                        entry = {
                            "round":  pred_round,
                            "top2":   pending["top2"],
                            "actual": actual_val,
                            "hit":    hit,
                            "action": "PLAY",
                        }
                        with _lock:
                            _live_log.append(entry)  # no cap — keep ALL history
                            _pending_pred = None
                        print(f"[Live] #{pred_round}: pred={pending['top2']} "
                              f"actual={actual_val} -> {'HIT' if hit else 'MISS'}")

                _rebuild_sim()
                with _lock:
                    lr = _raw_rounds[-1]["round"] if _raw_rounds else "?"
                print(f"[Fetcher] +{count} rounds. Latest: #{lr}. Total: {len(records)}")

        except Exception as e:
            print(f"[Fetcher] Unexpected: {e}")
            with _lock:
                _fetch_status["last_error"] = str(e)
                _fetch_status["status"]     = "error"
        time.sleep(POLL_INTERVAL)

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

def score_round(h, prob,t1,tp1,t2,tp2,t3,tp3,t4,tp4,ag,ar):
    n=len(h)
    if n<4:
        ranked=sorted(prob.items(),key=lambda x:-x[1])
        return [ranked[0][0],ranked[1][0]],{k:1/8 for k in range(1,9)},3.0,0.125,0.125
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
    rk=sorted(sc.items(),key=lambda x:-x[1])
    ent=-sum(v*math.log2(v) for v in sc.values() if v>0)
    return [rk[0][0],rk[1][0]],sc,ent,rk[0][1],rk[1][1]

def should_play(t1, ent, brake_active=False):
    if brake_active: return False
    return t1 > SKIP_TOP1_THRESHOLD and ent < _entropy_threshold

# ── SIMULATION ────────────────────────────────────────────────────────────────
def _rebuild_sim():
    global _sim_stats, _brake_left
    with _lock:
        rewards=list(_rewards)
    total=len(rewards)
    if total<TRAIN_ROUNDS+5: return

    stats=build_global_stats(rewards)
    hits=misses=sk=pl=0; ls=mx=ws=mw=0; brake=0; se=st=sb=0

    for i in range(TRAIN_ROUNDS,total-1):
        h=rewards[:i+1]; tn=rewards[i+1]
        top2,_,ent,t1s,_=score_round(h,*stats)
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

    sim_total=total-TRAIN_ROUNDS-1
    acc=hits/pl*100 if pl else 0
    play_pct = pl/sim_total if sim_total else 0

    # ── ADAPTIVE ENTROPY THRESHOLD ────────────────────────────────────────────
    # After simulation, check if play rate is in target range.
    # If too few rounds played (< 38%) → relax threshold (raise it)
    # If too many rounds played (> 45%) → tighten threshold (lower it)
    # Step size: 0.01 per adjustment, clamped to [ENT_THRESH_MIN, ENT_THRESH_MAX]
    global _entropy_threshold
    cur_thresh = _entropy_threshold
    if play_pct < TARGET_PLAY_MIN:
        # Too many skips — relax entropy filter
        new_thresh = min(ENT_THRESH_MAX, cur_thresh + 0.01)
    elif play_pct > TARGET_PLAY_MAX:
        # Too many plays — tighten entropy filter
        new_thresh = max(ENT_THRESH_MIN, cur_thresh - 0.01)
    else:
        new_thresh = cur_thresh  # already in target range

    if abs(new_thresh - cur_thresh) > 0.001:
        print(f"[Adaptive] Entropy threshold: {cur_thresh:.3f} → {new_thresh:.3f}  "
              f"(play_pct={play_pct*100:.1f}%, target={TARGET_PLAY_MIN*100:.0f}–{TARGET_PLAY_MAX*100:.0f}%)")
    _entropy_threshold = new_thresh

    with _lock:
        _sim_stats={
            "total":total,"sim_total":sim_total,
            "played":pl,"skipped":sk,"hits":hits,"misses":misses,
            "accuracy":round(acc,2),
            "play_pct":round(play_pct*100,1),
            "max_loss":mx,"max_win":mw,
            "skip_brake":sb,"skip_entropy":se,"skip_top1":st,
            "entropy_threshold":round(new_thresh,3),
        }
        _brake_left=brake

# ── LIVE PREDICTION ───────────────────────────────────────────────────────────
def get_prediction():
    global _pending_pred
    with _lock:
        rewards=list(_rewards); raw=list(_raw_rounds); brake=_brake_left
    if len(rewards)<TRAIN_ROUNDS+5: return None

    stats=build_global_stats(rewards)
    top2,scores,ent,t1s,t2s=score_round(rewards,*stats)
    eth  = _entropy_threshold   # adaptive threshold
    play = t1s > SKIP_TOP1_THRESHOLD and ent < eth and brake == 0

    skip_reason=None
    if brake>0:                    skip_reason=f"Loss brake ({brake} rounds left)"
    elif ent>=eth:                 skip_reason=f"High entropy ({ent:.4f} ≥ {eth:.3f})"
    elif t1s<=SKIP_TOP1_THRESHOLD: skip_reason=f"Low confidence ({t1s:.4f})"

    last_round=raw[-1]["round"] if raw else None
    next_round=(last_round+1) if last_round else None

    if play and next_round is not None:
        with _lock:
            cur=_pending_pred
        if cur is None or cur["round"]!=next_round:
            with _lock:
                _pending_pred={"round":next_round,"top2":list(top2)}

    return {
        "next_round":next_round,"latest_round":last_round,
        "pred1":top2[0],"pred2":top2[1],
        "pred1_name":CLASS_NAMES.get(top2[0],"?"),"pred2_name":CLASS_NAMES.get(top2[1],"?"),
        "pred1_color":CLASS_COLORS.get(top2[0],"#888"),"pred2_color":CLASS_COLORS.get(top2[1],"#888"),
        "pred1_conf":round(t1s*100,2),"pred2_conf":round(t2s*100,2),
        "entropy":round(ent,4),"action":"PLAY" if play else "SKIP",
        "skip_reason":skip_reason,
        "all_scores":{k:round(v*100,2) for k,v in scores.items()},
        "last_10":rewards[-10:],"total_rounds":len(rewards),
    }

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",class_names=CLASS_NAMES,class_colors=CLASS_COLORS)

@app.route("/api/predict")
def api_predict():
    pred=get_prediction()
    if pred is None: return jsonify({"error":"Not enough data yet"}),503
    return jsonify(pred)

@app.route("/api/stats")
def api_stats():
    with _lock:
        stats=dict(_sim_stats); live=list(_live_log)
    # Live streak calc
    cur=0; mx_live=0
    for e in reversed(live):
        if not e["hit"]: cur+=1; mx_live=max(mx_live,cur)
        else: break
    stats["live_log"]=live
    stats["live_cur_streak"]=cur
    stats["live_max_loss"]=mx_live
    return jsonify(stats)

@app.route("/api/status")
def api_status():
    with _lock:
        fs=dict(_fetch_status); total=len(_rewards)
        latest=_raw_rounds[-1]["round"] if _raw_rounds else None
    now_ist=datetime.now(IST)
    fs["total_rounds"]=total; fs["latest_round"]=latest
    fs["server_time_ist"]=now_ist.strftime("%H:%M:%S IST")
    fs["entropy_threshold"]=round(_entropy_threshold,3)
    # Time until next reset
    reset_today=now_ist.replace(hour=RESET_HOUR_IST,minute=RESET_MINUTE_IST,second=0,microsecond=0)
    from datetime import timedelta
    if now_ist>=reset_today:
        reset_today+=timedelta(days=1)
    secs=(reset_today-now_ist).seconds
    fs["next_reset_in"]=f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"
    return jsonify(fs)

@app.route("/api/history")
def api_history():
    with _lock: raw=list(_raw_rounds[-50:])
    return jsonify([{
        "round":r["round"],"reward_index":r["reward_index"],
        "name":CLASS_NAMES.get(r["reward_index"],"?"),
        "color":CLASS_COLORS.get(r["reward_index"],"#888")
    } for r in reversed(raw)])

# ── STARTUP ───────────────────────────────────────────────────────────────────
def startup():
    global _last_reset_date

    # Set last_reset_date based on current time so we don't double-reset on start
    now_ist=datetime.now(IST)
    past_530=(now_ist.hour>RESET_HOUR_IST or
              (now_ist.hour==RESET_HOUR_IST and now_ist.minute>=RESET_MINUTE_IST))
    if past_530:
        # We're past today's reset time — mark today as already reset
        # (so we don't wipe data that was just loaded for today)
        _last_reset_date=now_ist.date()

    records=_load_file()
    if records:
        rewards=[r["reward_index"] for r in records]
        with _lock:
            _raw_rounds.extend(records); _rewards.extend(rewards)
        print(f"[Startup] Loaded {len(records)} rounds. Building sim...")
        _rebuild_sim()
        with _lock: s=dict(_sim_stats)
        print(f"[Startup] Done — played={s.get('played')}, acc={s.get('accuracy')}%, maxloss={s.get('max_loss')}")
    else:
        print("[Startup] No data file found. Waiting for API fetch...")

    threading.Thread(target=fetcher_loop,daemon=True).start()
    print(f"[Startup] Fetcher started. IST now: {now_ist.strftime('%H:%M:%S')}. Next reset at 05:30 IST.")

startup()

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=False)
