# ⚡ Treasure AI — StarMaker Prediction Website

Mobile-first prediction site using the v3 Markov engine.
Auto-fetches new rounds every 5s, rebuilds predictions live.

---

## Files

```
treasure/
├── app.py              ← Flask server + prediction engine + auto-fetcher (all-in-one)
├── templates/
│   └── index.html      ← Mobile-optimized prediction UI
├── requirements.txt
└── round_data.json     ← Copy your existing file here before starting
```

---

## Setup & Run

### 1. Copy your data file
```bash
cp /path/to/your/round_data.json ./round_data.json
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start the server
```bash
python app.py
```

### 4. Open in browser
```
http://localhost:5000
```
On your phone (same WiFi): `http://<your-pc-ip>:5000`

---

## How it works

- **Fetcher** runs in background every 5s, hits the StarMaker API, saves new rounds to `round_data.json`
- **Prediction engine** uses v3 Markov logic (order 1–4) + gap/run analysis
- **Skip engine** uses `top1 > 0.220` AND `entropy < 2.70` thresholds (v3 tuned)
- **Loss brake** auto-pauses 3 rounds after 3 consecutive losses
- **UI** auto-refreshes every 5s showing latest prediction

---

## Accuracy (from simulation on 2,566 rounds)
- Play rate: ~40%
- Accuracy on played rounds: ~64.6%
- Max loss streak: 4
- Note: 70% accuracy + 40% play is mathematically not achievable on this dataset

---

## Production (optional — run as background service)

```bash
# Install gunicorn
pip install gunicorn

# Run (replace 5000 with your port)
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

> Use `-w 1` (1 worker) so the background fetcher thread isn't duplicated.
