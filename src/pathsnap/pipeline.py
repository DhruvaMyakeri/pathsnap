"""End-to-end inference for one trip, plus server-side scoring.

The prediction path (``prepare_trip`` -> ``score_candidates`` -> ``predict_trip``)
only reads the GPS track and the road network. The reference route is read in
``score_trip`` alone, after the prediction is final; the API runs that step
server-side and only returns aggregate metrics, never the reference route.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .candidates import generate_candidates
from .data import clean_track, load_route, load_trip
from .features import FEATURES, build_features
from .geo import Projector
from .labels import point_labels
from .metrics import collapse, direction_agnostic_accuracy, pointwise_accuracy, route_continuity, route_overlap
from .network import RoadNetwork
from .routing import Router

MODEL_PATH = config.ARTIFACTS_DIR / "model.joblib"
HMM_CONFIG_PATH = config.ARTIFACTS_DIR / "hmm_config.json"


@dataclass
class TripContext:
    trip_id: str
    n_raw: int  # rows in the .track file
    keep: np.ndarray  # indices of usable rows (into the raw track)
    lonlat: np.ndarray  # (n, 2) usable fixes
    t: np.ndarray  # (n,)
    xy: np.ndarray  # (n, 2) projected metres
    projector: Projector
    nodes_lonlat: np.ndarray
    net: RoadNetwork
    router: Router
    cands: pd.DataFrame  # candidate rows with FEATURES
    timings: dict = field(default_factory=dict)


@dataclass
class TripPrediction:
    trip_id: str
    point_arc_clf: np.ndarray  # classifier-only arc per usable point (-1 = none)
    point_arc: np.ndarray  # final arc per usable point (HMM output when used)
    route_arcs: np.ndarray  # connected arc sequence of the final route
    used_hmm: bool
    n_breaks: int
    model_name: str
    timings: dict = field(default_factory=dict)


def prepare_trip(trip_dir: Path | str) -> TripContext:
    """Load GPS + road network, project to metres, generate candidates and features."""
    t0 = time.perf_counter()
    trip = load_trip(trip_dir)
    keep = clean_track(trip.track)
    track = trip.track[keep]
    lonlat, t = track[:, :2], track[:, 2]
    projector = Projector.for_points(lonlat)
    xy = projector.to_xy(lonlat)
    t1 = time.perf_counter()
    net = RoadNetwork(projector.to_xy(trip.nodes), trip.arcs)
    router = Router(net)
    t2 = time.perf_counter()
    cands = generate_candidates(net, xy)
    cands = build_features(net, xy, t, cands)
    t3 = time.perf_counter()
    return TripContext(
        trip_id=trip.trip_id, n_raw=len(trip.track), keep=keep, lonlat=lonlat, t=t, xy=xy,
        projector=projector, nodes_lonlat=trip.nodes, net=net, router=router, cands=cands,
        timings={"load_project_s": t1 - t0, "network_s": t2 - t1, "candidates_features_s": t3 - t2},
    )


# ---------------------------------------------------------------- scoring
def heuristic_prob(cands: pd.DataFrame, sigma_m: float = 10.0) -> np.ndarray:
    """Non-learned baseline score: Gaussian distance emission x heading agreement.

    This is the classic map-matching emission (Newson & Krumm 2009) plus a
    penalty for driving against the arc direction. It is used as a baseline
    and as a fallback when no trained model artifact exists yet.
    """
    d = cands["dist"].to_numpy(np.float64)
    hd = cands["heading_diff"].to_numpy(np.float64)
    return np.exp(-0.5 * (d / sigma_m) ** 2) * np.where(hd <= 90.0, 1.0, 0.1)


def load_bundle(path: Path | str = MODEL_PATH) -> dict | None:
    """Trained model bundle: {"model", "features", "name", ...} or None if absent."""
    path = Path(path)
    return joblib.load(path) if path.exists() else None


def load_hmm_params(path: Path | str = HMM_CONFIG_PATH):
    from .hmm import HMMParams  # noqa: WPS433 (optional dependency inside the package)

    path = Path(path)
    if path.exists():
        return HMMParams.from_dict(json.loads(path.read_text()))
    return HMMParams()


def score_candidates(ctx: TripContext, bundle: dict | None) -> np.ndarray:
    if bundle is None:
        return heuristic_prob(ctx.cands)
    X = ctx.cands[bundle.get("features", FEATURES)]
    return bundle["model"].predict_proba(X)[:, 1]


def argmax_per_point(cands: pd.DataFrame, prob: np.ndarray, n_points: int) -> np.ndarray:
    """Classifier-only prediction: best-scoring arc per point (-1 if no candidate)."""
    df = pd.DataFrame({"pt": cands["pt"].to_numpy(), "arc": cands["arc"].to_numpy(),
                       "prob": prob, "rank": cands["rank"].to_numpy()})
    best = df.sort_values(["pt", "prob", "rank"], ascending=[True, False, True], kind="mergesort")
    best = best.drop_duplicates("pt", keep="first")
    out = np.full(n_points, -1, dtype=np.int64)
    out[best["pt"].to_numpy()] = best["arc"].to_numpy()
    return out


# ---------------------------------------------------------------- predict
def predict_trip(trip_dir: Path | str, bundle: dict | None = None, hmm_params=None,
                 use_hmm: bool = True, ctx: TripContext | None = None) -> tuple[TripContext, TripPrediction]:
    """Full prediction for one trip. Never reads the reference route."""
    ctx = ctx or prepare_trip(trip_dir)
    n = len(ctx.xy)
    t0 = time.perf_counter()
    prob = score_candidates(ctx, bundle)
    ctx.cands["prob"] = prob
    clf = argmax_per_point(ctx.cands, prob, n)
    t1 = time.perf_counter()

    used_hmm, n_breaks = False, 0
    point_arc, route_arcs = clf, collapse(clf)
    if use_hmm:
        try:
            from . import hmm
        except ImportError:
            hmm = None
        if hmm is not None:
            params = hmm_params if hmm_params is not None else load_hmm_params()
            res = hmm.decode(ctx.net, ctx.router, ctx.xy, ctx.t, ctx.cands, params)
            point_arc, route_arcs, n_breaks, used_hmm = res.point_arc, res.route_arcs, res.n_breaks, True
    t2 = time.perf_counter()
    timings = dict(ctx.timings, classify_s=t1 - t0, decode_s=t2 - t1)
    timings["total_s"] = sum(timings.values())
    name = bundle.get("name", "model") if bundle else "heuristic_baseline"
    return ctx, TripPrediction(ctx.trip_id, clf, point_arc, np.asarray(route_arcs, dtype=np.int64),
                               used_hmm, int(n_breaks), name, timings)


def route_polylines(ctx: TripContext, route_arcs) -> list[list[list[float]]]:
    """Route as [[lat, lon], ...] polylines; a new polyline starts wherever arcs do not connect."""
    lines: list[list[list[float]]] = []
    prev_dst = None
    for a in np.asarray(route_arcs, dtype=np.int64):
        s, d = int(ctx.net.src[a]), int(ctx.net.dst[a])
        s_ll, d_ll = ctx.nodes_lonlat[s], ctx.nodes_lonlat[d]
        if prev_dst is None or s != prev_dst:
            lines.append([[float(s_ll[1]), float(s_ll[0])]])
        lines[-1].append([float(d_ll[1]), float(d_ll[0])])
        prev_dst = d
    return lines


# ---------------------------------------------------------------- scoring (server side)
def score_trip(trip_dir: Path | str, ctx: TripContext, pred: TripPrediction) -> dict:
    """Compare a finished prediction with the reference route. Server-side only."""
    route = load_route(trip_dir)
    lab = point_labels(ctx.net, ctx.xy, route)
    valid = lab["label_valid"]
    ref_route = route[(route >= 0) & (route < ctx.net.n_arcs)]
    out = {
        "n_points": int(len(ctx.xy)),
        "n_labelled_points": int(valid.sum()),
        "accuracy": pointwise_accuracy(pred.point_arc, lab["label_arc"], valid),
        "accuracy_classifier_only": pointwise_accuracy(pred.point_arc_clf, lab["label_arc"], valid),
        "direction_agnostic_accuracy": direction_agnostic_accuracy(pred.point_arc, lab["label_arc"], valid, ctx.net),
    }
    out.update(route_overlap(pred.route_arcs, ref_route, ctx.net))
    out["continuity"] = route_continuity(pred.point_arc, ctx.xy, ctx.net, ctx.router)["continuity"]
    out["continuity_classifier_only"] = route_continuity(pred.point_arc_clf, ctx.xy, ctx.net, ctx.router)["continuity"]
    return out
