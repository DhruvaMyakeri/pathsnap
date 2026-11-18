"""PathSnap evaluation API (Laptop 1).

Run from the repository root:

    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints
  GET  /health    service status, loaded model, HMM on/off, eval folder, trip count
  GET  /trips     trips available for evaluation: [{trip_id, n_points}]
  POST /predict   {"trip_id": "..."} -> predicted route + server-side metrics

The whole prediction pipeline runs here: candidates, classifier and HMM/Viterbi.
For each request the API:

1. runs ``pipeline.predict_trip``, which reads only the GPS track and road network;
2. then, once the prediction is final, runs ``pipeline.score_trip``, which reads
   the trip's ``.route`` file and computes accuracy.

The response carries aggregate metrics only. It never includes the reference
route, per-point labels or per-point correctness, so ground truth never leaves
this machine.

Configuration (environment variables)
  PATHSNAP_EVAL_DIR    folder of trip sub-folders to serve (default: the student_test partition)
  PATHSNAP_MODEL       trained model bundle (default: artifacts/model.joblib)
  PATHSNAP_HMM_CONFIG  HMM parameters (default: artifacts/hmm_config.json)
  PATHSNAP_USE_HMM     "0" disables HMM decoding (default: enabled when available)
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap import config, pipeline
from pathsnap.data import list_trips, trip_file

MAX_GPS_POINTS = 3000  # GPS fixes returned for display; longer tracks are thinned evenly
METRIC_KEYS = (  # whitelist: only aggregate metrics ever leave the server
    "accuracy", "accuracy_classifier_only", "direction_agnostic_accuracy",
    "route_precision", "route_recall", "route_f1",
    "continuity", "continuity_classifier_only", "n_labelled_points",
)

# ---------------------------------------------------------------- logging
log = logging.getLogger("pathsnap.api")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


# ---------------------------------------------------------------- state
@dataclass
class ServiceState:
    eval_dir: Path
    model_path: Path
    hmm_config_path: Path
    bundle: dict | None = None
    model_name: str = "heuristic_baseline"
    hmm_enabled: bool = False
    hmm_params: object | None = None
    hmm_note: str = ""
    started_at: float = field(default_factory=time.time)
    n_requests: int = 0


def _load_state() -> ServiceState:
    state = ServiceState(
        eval_dir=Path(os.environ.get("PATHSNAP_EVAL_DIR", config.PARTITIONS["student_test"])).resolve(),
        model_path=Path(os.environ.get("PATHSNAP_MODEL", pipeline.MODEL_PATH)),
        hmm_config_path=Path(os.environ.get("PATHSNAP_HMM_CONFIG", pipeline.HMM_CONFIG_PATH)),
    )
    log.info("PathSnap API starting")
    log.info("Eval dir   : %s", state.eval_dir)

    try:
        state.bundle = pipeline.load_bundle(state.model_path)
    except Exception:  # a corrupt artifact must not take the service down
        log.exception("Could not load model bundle %s", state.model_path)
        state.bundle = None
    if state.bundle is not None:
        state.model_name = str(state.bundle.get("name", "model"))
        log.info("Model      : %s (loaded from %s)", state.model_name, state.model_path)
    else:
        log.warning("Model      : no bundle at %s -> heuristic baseline scorer", state.model_path)

    if os.environ.get("PATHSNAP_USE_HMM", "1").strip() == "0":
        state.hmm_note = "disabled by PATHSNAP_USE_HMM=0"
    else:
        try:
            import pathsnap.hmm  # noqa: F401  (availability check)

            state.hmm_params = pipeline.load_hmm_params(state.hmm_config_path)
            state.hmm_enabled = True
            src = state.hmm_config_path if state.hmm_config_path.exists() else "built-in defaults"
            state.hmm_note = f"params from {src}"
        except ImportError:
            state.hmm_note = "pathsnap.hmm not available -> classifier-only decoding"
        except Exception as exc:  # broken module or config: fall back, but say so loudly
            log.exception("HMM could not be initialised")
            state.hmm_note = f"HMM init failed ({type(exc).__name__}) -> classifier-only decoding"
    log.info("HMM        : %s (%s)", "ON" if state.hmm_enabled else "OFF", state.hmm_note)
    log.info("Trips      : %d available", len(_trip_index(state.eval_dir)))
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pathsnap = _load_state()
    log.info("Ready. Waiting for requests on /health, /trips, /predict")
    yield
    log.info("PathSnap API shutting down")


app = FastAPI(title="PathSnap API", version="1.0",
              description="Map-matching of noisy GPS trips with a candidate classifier and HMM/Viterbi decoding.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- helpers
def _trip_index(eval_dir: Path) -> dict[str, Path]:
    """trip_id -> folder, for every trip folder that has a .track file."""
    if not eval_dir.is_dir():
        return {}
    return {p.name: p for p in list_trips(eval_dir)}


def _count_points(trip_dir: Path) -> int:
    with open(trip_file(trip_dir, "track"), "rb") as f:
        return sum(1 for line in f if line.strip())


def _json_safe(o):
    """numpy -> python, NaN/inf -> None, recursively."""
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _pct(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{100 * v:.2f}%"


def _state(request: Request) -> ServiceState:
    return request.app.state.pathsnap


# ---------------------------------------------------------------- endpoints
class PredictRequest(BaseModel):
    trip_id: str


@app.get("/")
def root():
    return {"service": "PathSnap API", "endpoints": ["/health", "/trips", "/predict", "/docs"]}


@app.get("/health")
def health(request: Request):
    s = _state(request)
    trips = _trip_index(s.eval_dir)
    return {
        "status": "ok" if trips else "degraded: no trips found in eval dir",
        "model_name": s.model_name,
        "mode": "trained model" if s.bundle is not None else "heuristic baseline",
        "hmm_enabled": s.hmm_enabled,
        "hmm_note": s.hmm_note,
        "eval_dir": str(s.eval_dir),
        "n_trips": len(trips),
        "uptime_s": round(time.time() - s.started_at, 1),
        "requests_served": s.n_requests,
    }


@app.get("/trips")
def trips(request: Request):
    s = _state(request)
    out = [{"trip_id": tid, "n_points": _count_points(d)} for tid, d in sorted(_trip_index(s.eval_dir).items())]
    log.info("TRIPS    listed %d trips for %s", len(out), request.client.host if request.client else "?")
    return out


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    """Run the full pipeline on one trip and score it server-side (sync def -> runs in a threadpool)."""
    s = _state(request)
    client = request.client.host if request.client else "?"
    trip_id = req.trip_id.strip()
    log.info("REQUEST  /predict trip=%s from %s", trip_id, client)
    trip_dir = _trip_index(s.eval_dir).get(trip_id)
    if trip_dir is None:  # only ids from the listing are accepted (blocks path traversal)
        log.warning("REJECT   unknown trip id %r from %s", trip_id, client)
        raise HTTPException(status_code=404, detail=f"Unknown trip id: {trip_id!r}")

    t0 = time.perf_counter()
    try:
        ctx, pred = pipeline.predict_trip(trip_dir, bundle=s.bundle, hmm_params=s.hmm_params,
                                          use_hmm=s.hmm_enabled)
    except Exception as exc:
        log.exception("FAILED   prediction for trip=%s", trip_id)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")
    t_pred = time.perf_counter()

    # Server-side scoring: reads the reference route AFTER the prediction is final.
    metrics: dict = {k: None for k in METRIC_KEYS}
    try:
        scored = pipeline.score_trip(trip_dir, ctx, pred)
        metrics = {k: scored.get(k) for k in METRIC_KEYS}
    except FileNotFoundError:
        log.warning("SCORE    trip=%s has no reference route -> metrics unavailable", trip_id)
    except Exception:
        log.exception("SCORE    failed for trip=%s -> metrics unavailable", trip_id)
    t_score = time.perf_counter()

    n = len(ctx.lonlat)
    step = max(1, math.ceil(n / MAX_GPS_POINTS))
    idx = np.arange(0, n, step)
    if n and idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    gps = np.round(ctx.lonlat[idx][:, ::-1], 6)  # -> [lat, lon]
    lines = [[[round(lat, 6), round(lon, 6)] for lat, lon in line] for line in pipeline.route_polylines(ctx, pred.route_arcs)]
    route_len_km = float(np.nansum(ctx.net.length[pred.route_arcs]) / 1000) if len(pred.route_arcs) else 0.0
    timings = dict(pred.timings, score_s=t_score - t_pred, request_total_s=t_score - t0)

    payload = _json_safe({
        "trip_id": trip_id,
        "model_name": pred.model_name,
        "used_hmm": pred.used_hmm,
        "n_points": n,
        "n_raw_points": ctx.n_raw,
        "n_breaks": pred.n_breaks,
        "gps": gps,
        "gps_downsample": {"every_nth": step, "n_returned": len(idx),
                           "note": f"every {step}th fix shown (plus the last)" if step > 1 else "all fixes shown"},
        "route": lines,
        "route_summary": {"n_arcs": len(pred.route_arcs), "n_polylines": len(lines), "length_km": route_len_km},
        "bounds": [[float(ctx.lonlat[:, 1].min()), float(ctx.lonlat[:, 0].min())],
                   [float(ctx.lonlat[:, 1].max()), float(ctx.lonlat[:, 0].max())]] if n else None,
        "metrics": metrics,
        "timings": timings,
    })
    body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    s.n_requests += 1

    tm = pred.timings
    log.info("STAGES   trip=%s load+project %.2fs | network %.2fs | candidates+features %.2fs | "
             "classify %.2fs | decode %.2fs | score %.2fs", trip_id, tm.get("load_project_s", 0),
             tm.get("network_s", 0), tm.get("candidates_features_s", 0), tm.get("classify_s", 0),
             tm.get("decode_s", 0), t_score - t_pred)
    log.info("RESULT   trip=%s points=%d labelled=%s accuracy=%s (classifier-only %s) route_F1=%s "
             "continuity=%s | model=%s hmm=%s breaks=%d", trip_id, n, metrics.get("n_labelled_points"),
             _pct(metrics.get("accuracy")), _pct(metrics.get("accuracy_classifier_only")),
             _pct(metrics.get("route_f1")), _pct(metrics.get("continuity")), pred.model_name,
             "on" if pred.used_hmm else "off", pred.n_breaks)
    log.info("RESPONSE trip=%s %.1f KB in %.2fs", trip_id, len(body) / 1024, time.perf_counter() - t0)
    return Response(content=body, media_type="application/json")
