"""Classifier-only vs classifier + HMM on the 10 student_test trips, for every model.

For each model with ``data_cache/preds/test_<model>.parquet``, plus the
non-learned distance x heading score (``pipeline.heuristic_prob``), every
student_test trip is decoded twice:

* classifier-only: arg-max candidate per point; the route is the collapsed sequence of those arcs
* classifier + HMM: Viterbi decoding (``pathsnap.hmm``); the route includes shortest-path fills

Then both are compared with the reference route. The ``baseline_*`` prediction
files are skipped: their scores are hard 0/1 decisions, so they carry no
emission information for the HMM.

Metrics (definitions in pathsnap.metrics): pointwise accuracy and
direction-agnostic accuracy over labelled points, route continuity,
length-weighted route precision / recall / F1, number of chain breaks, and decode time.

Outputs
  results/hmm/sequence_eval.csv            one row per model x mode (accuracy micro-averaged over points,
                                           continuity over point pairs, route metrics averaged over trips)
  results/hmm/sequence_eval_per_trip.csv   one row per trip x model x mode

HMM parameters come from artifacts/hmm_config.json when present (else the
defaults); the column ``hmm_params_source`` records which.

Usage:  python scripts/eval_sequences.py [--jobs 6]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.data import list_trips, load_route
from pathsnap.hmm import HMMParams, decode
from pathsnap.labels import point_labels
from pathsnap.metrics import collapse, route_continuity, route_overlap
from pathsnap.pipeline import HMM_CONFIG_PATH, argmax_per_point, heuristic_prob, prepare_trip

OUT = config.RESULTS_DIR / "hmm"
HEURISTIC = "heuristic_distance_heading"


def available_models() -> list[str]:
    files = sorted((config.CACHE_DIR / "preds").glob("test_*.parquet"))
    return [f.stem[len("test_"):] for f in files if not f.stem.startswith("test_baseline_")]


def load_test_probs(models: list[str]) -> dict[str, dict[str, pd.DataFrame]]:
    out = {}
    for m in models:
        df = pd.read_parquet(config.CACHE_DIR / "preds" / f"test_{m}.parquet")
        df["trip_id"] = df["trip_id"].astype(str)
        out[m] = {tid: g[["pt", "arc", "prob"]] for tid, g in df.groupby("trip_id", observed=True)}
    return out


def eval_trip(trip_dir, probs_by_model: dict[str, pd.DataFrame | None], params: HMMParams) -> list[dict]:
    warnings.filterwarnings("ignore")
    ctx = prepare_trip(trip_dir)
    route = load_route(trip_dir)
    ref_route = route[(route >= 0) & (route < ctx.net.n_arcs)]
    lab = point_labels(ctx.net, ctx.xy, route)
    v, la = lab["label_valid"], lab["label_arc"]
    twin_of_label = ctx.net.twin[np.maximum(la, 0)]
    n = len(ctx.xy)
    rows = []
    for model, probs in probs_by_model.items():
        c = ctx.cands
        if probs is None:
            c["prob"] = heuristic_prob(c)
        else:
            merged = c[["pt", "arc"]].merge(probs, on=["pt", "arc"], how="left")
            if merged["prob"].isna().any():
                raise ValueError(f"{model}/{ctx.trip_id}: candidate rows without probability")
            c["prob"] = merged["prob"].to_numpy()
        clf = argmax_per_point(c, c["prob"].to_numpy(), n)
        t0 = time.perf_counter()
        res = decode(ctx.net, ctx.router, ctx.xy, ctx.t, c, params)
        decode_s = time.perf_counter() - t0
        for mode, point_arc, route_arcs, breaks, sec in [
            ("classifier_only", clf, collapse(clf), 0, 0.0),
            ("classifier_hmm", res.point_arc, res.route_arcs, res.n_breaks, decode_s),
        ]:
            pa = point_arc[v]
            cont = route_continuity(point_arc, ctx.xy, ctx.net, ctx.router)
            rows.append({
                "trip_id": ctx.trip_id, "model": model, "mode": mode, "n_points": n, "n_lab": int(v.sum()),
                "correct": int((pa == la[v]).sum()),
                "correct_dir_agnostic": int(((pa == la[v]) | ((pa >= 0) & (pa == twin_of_label[v]))).sum()),
                "n_pairs": cont["n_pairs"], "n_jumps": cont["n_jumps"],
                **route_overlap(route_arcs, ref_route, ctx.net),
                "route_arcs": int(len(route_arcs)), "n_breaks": int(breaks), "decode_s": sec,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if HMM_CONFIG_PATH.exists():
        cfg = json.loads(HMM_CONFIG_PATH.read_text())
        params, source = HMMParams.from_dict(cfg), f"artifacts/hmm_config.json (tuned with {cfg.get('model', '?')})"
    else:
        params, source = HMMParams(), "defaults"
    models = available_models()
    probs = load_test_probs(models)
    trips = list_trips(config.PARTITIONS["student_test"])

    def per_trip(d):
        return {HEURISTIC: None, **{m: probs[m][d.name] for m in models}}

    res = Parallel(n_jobs=args.jobs)(delayed(eval_trip)(d, per_trip(d), params) for d in trips)
    per = pd.DataFrame([r for rows in res for r in rows])
    per["accuracy"] = per["correct"] / per["n_lab"]
    per["continuity"] = 1 - per["n_jumps"] / per["n_pairs"]
    per.to_csv(OUT / "sequence_eval_per_trip.csv", index=False)

    g = per.groupby(["model", "mode"], sort=False)
    summary = g.agg(trips=("trip_id", "nunique"), labelled_points=("n_lab", "sum"), correct=("correct", "sum"),
                    correct_dir_agnostic=("correct_dir_agnostic", "sum"), pairs=("n_pairs", "sum"),
                    jumps=("n_jumps", "sum"), route_precision=("route_precision", "mean"),
                    route_recall=("route_recall", "mean"), route_f1=("route_f1", "mean"),
                    n_breaks=("n_breaks", "sum"), decode_s_total=("decode_s", "sum")).reset_index()
    summary["accuracy"] = summary["correct"] / summary["labelled_points"]
    summary["direction_agnostic_accuracy"] = summary["correct_dir_agnostic"] / summary["labelled_points"]
    summary["continuity"] = 1 - summary["jumps"] / summary["pairs"]
    summary["decode_s_per_trip"] = summary["decode_s_total"] / summary["trips"]
    summary["hmm_params_source"] = source
    summary["hmm_params"] = json.dumps(params.to_dict())
    cols = ["model", "mode", "trips", "labelled_points", "accuracy", "direction_agnostic_accuracy", "continuity",
            "route_precision", "route_recall", "route_f1", "n_breaks", "decode_s_total", "decode_s_per_trip",
            "hmm_params_source", "hmm_params"]
    summary[cols].to_csv(OUT / "sequence_eval.csv", index=False)
    print(f"HMM params: {source}")
    print(summary[cols[:13]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
