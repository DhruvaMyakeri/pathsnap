# PathSnap: two-laptop deployment runbook

| Laptop | Runs | Holds |
|---|---|---|
| **Laptop 1: model + API** | `api/main.py` (FastAPI, port 8000) | trained model, HMM/Viterbi, road networks, evaluation trips **including the reference routes** |
| **Laptop 2: UI** | `ui/app.py` (Streamlit, port 8501) | nothing but the UI code; it only talks to the API over HTTP |

Ground truth (`.route` files) exists only on Laptop 1. The API computes accuracy there and returns
**aggregate metrics only**: no reference route, no per-point labels, no per-point correctness.

---

## 1. One-time setup

**Laptop 1** (full project):

```bat
cd path\to\pathsnap
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Make sure `artifacts\model.joblib` and `artifacts\hmm_config.json` are present (committed in the repo).
Without them, the API falls back to the heuristic baseline and logs a warning at startup.

**Laptop 2** (UI only):

```bat
cd path\to\pathsnap
python -m venv .venv
.venv\Scripts\activate
pip install -r ui\requirements.txt
```

## 2. Network

1. Connect **both laptops to the same Wi-Fi or phone hotspot**. A hotspot avoids campus Wi-Fi
   blocking traffic between clients. The map tiles in the UI need internet access on Laptop 2.
2. On Laptop 1, run `ipconfig` and note the **IPv4 Address** of the Wi-Fi adapter, e.g. `192.168.1.20`.

## 3. Start the API (Laptop 1)

```bat
scripts\run_api.bat
```

or, equivalently:

```bat
.venv\Scripts\activate
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- **Windows Firewall:** the first time, Windows asks whether to allow Python to accept connections.
  Tick **Private networks** and click **Allow access**. If you clicked Cancel earlier, re-allow
  `python.exe` in *Windows Security → Firewall & network protection → Allow an app through firewall*.
  Make sure the Wi-Fi network is set to *Private* in Windows settings. Do this yourself on Laptop 1.
- The console shows the startup lines: eval folder, model loaded, HMM ON/OFF, number of trips.

## 4. Check the connection (from Laptop 2)

Open `http://192.168.1.20:8000/health` in Laptop 2's browser (use your IP). You should see
`"status":"ok"` and `"n_trips":10`. Interactive API docs: `http://192.168.1.20:8000/docs`.

## 5. Start the UI (Laptop 2)

```bat
scripts\run_ui.bat http://192.168.1.20:8000
```

Then open **http://localhost:8501**. The sidebar shows *Connected*, plus the model and HMM status.
You can also change the API URL in the sidebar.

## 6. Demo day: evaluator trips

1. Copy the evaluator trip folders onto Laptop 1. Each folder is `<trip_id>\<trip_id>.track/.nodes/.arcs/.route`,
   all inside one parent folder, e.g. `D:\eval_trips\00000015\...`.
2. Stop the API (CTRL+C) and restart it pointing at that folder:
   ```bat
   scripts\run_api.bat "D:\eval_trips"
   ```
   (or `set PATHSNAP_EVAL_DIR=D:\eval_trips` before starting uvicorn).
3. Check that `/health` reports `"n_trips": 10`. Click **Check connection** in the UI to refresh the trip list.
4. The evaluator picks any trip in the UI and clicks **Run map-matching**.
5. **Show the evaluator the API console on Laptop 1.** Every request logs these lines:
   ```
   REQUEST  /predict trip=00000076 from 192.168.1.35
   STAGES   trip=00000076 load+project 0.06s | network 0.14s | candidates+features 0.09s | classify 0.02s | decode 0.43s | score 0.08s
   RESULT   trip=00000076 points=1486 labelled=1486 accuracy=96.23% (classifier-only 83.18%) route_F1=97.68% continuity=99.93% | model=... hmm=on breaks=0
   RESPONSE trip=00000076 41.5 KB in 0.85s
   INFO:     192.168.1.35:50816 - "POST /predict HTTP/1.1" 200 OK
   ```
   The `from <IP>` and the uvicorn access line prove the request came from Laptop 2 through the API.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| UI sidebar: *API not reachable* | Check the URL (http, IP, port 8000). Is the API window still running? Can Laptop 2's browser open `/health`? |
| `/health` works on Laptop 1 but not from Laptop 2 | Firewall (see step 3), the network set to *Public*, or Wi-Fi client isolation. Switch to a phone hotspot. |
| `n_trips: 0` / *degraded* | `PATHSNAP_EVAL_DIR` points at the wrong level. It must be the folder that *contains* the trip folders. |
| Startup log says *heuristic baseline* | `artifacts\model.joblib` missing; pull the repo again or set `PATHSNAP_MODEL`. |
| Startup log says *HMM OFF* | `src\pathsnap\hmm.py` missing or failing to import (the traceback is in the log), or `PATHSNAP_USE_HMM=0` is set. |
| Map is grey / no background | Laptop 2 has no internet, so map tiles cannot load. The route and points still draw. |
| Request times out (180 s) | Very long trip on a slow laptop. Close other programs on Laptop 1; check the STAGES timings in the log. |
| Port already in use | Another API instance is running. Close it, or `set PATHSNAP_PORT=8001` (and use that port in the UI URL). |
