"""Tune the HMM decoder on TRAINING trips using out-of-fold classifier probabilities.

Why out-of-fold (OOF) probabilities: a classifier is over-confident on the
trips it was fitted on. Tuning the decoder on in-sample scores would teach it
to trust the emissions too much and under-weight the transitions. OOF scores
(each training trip scored by a model that never saw it) behave like the
scores the decoder will get on unseen trips.

What is tuned: ``beta`` (transition scale) and ``max_states`` (candidates kept
per point). ``emission_weight`` stays 1: the decoded path only depends on the
product ``emission_weight * beta``, because scaling every log-score by the
same constant does not change the Viterbi arg-max. Searching both would
duplicate the same configurations.

Protocol
  1. Grid search on a fixed random subset of 30 training trips (seed 42).
     Objective: pointwise accuracy over labelled points (micro-averaged).
     Ties are broken by route continuity.
  2. The best and the default configuration are then evaluated on all 80
     training trips (OOF probabilities) and on the 10 student_test trips
     (probabilities of the model refit on all training trips), next to the
     classifier-only arg-max. No evaluator data is involved anywhere.

Outputs
  results/hmm/hmm_tuning_<model>.csv   one row per configuration (subset objective)
  results/hmm/hmm_best_<model>.csv     classifier-only vs default vs tuned HMM, train(80) and student_test(10)
  artifacts/hmm_config.json            tuned parameters + provenance (read by pathsnap.pipeline)
  results/hmm/hmm_config_<model>.json  the same, kept per model

Usage
  python scripts/tune_hmm.py --model lightgbm [--jobs 6] [--n-trips 30]
  python scripts/tune_hmm.py --model heuristic      # non-learned emission scores
"""
from __future__ import annotations

import argparse
import itertools
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
from pathsnap.metrics import route_continuity
from pathsnap.pipeline import argmax_per_point, heuristic_prob, prepare_trip

BETAS = [0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 100]
MAX_STATES = [4, 8, 12]
OUT = config.RESULTS_DIR / "hmm"


def load_probs(model: str, kind: str) -> dict[str, pd.DataFrame] | None:
    """{trip_id: (pt, arc, prob)} from data_cache/preds/<kind>_<model>.parquet (None for the heuristic)."""
    if model == "heuristic":
        return None
    df = pd.read_parquet(config.CACHE_DIR / "preds" / f"{kind}_{model}.parquet")
    df["trip_id"] = df["trip_id"].astype(str)
    return {tid: g[["pt", "arc", "prob"]] for tid, g in df.groupby("trip_id", observed=True)}


def context_with_probs(trip_dir, probs: pd.DataFrame | None):
    """Rebuild the trip context (GPS + network only) and attach classifier probabilities."""
    ctx = prepare_trip(trip_dir)
    c = ctx.cands
    if probs is None:
        c["prob"] = heuristic_prob(c)
    else:
        merged = c[["pt", "arc"]].merge(probs, on=["pt", "arc"], how="left")
        if merged["prob"].isna().any():
            raise ValueError(f"{ctx.trip_id}: {int(merged['prob'].isna().sum())} candidate rows have no probability")
        c["prob"] = merged["prob"].to_numpy()
    lab = point_labels(ctx.net, ctx.xy, load_route(trip_dir))
    return ctx, lab


def eval_trip(trip_dir, probs, configs: list[tuple[str, HMMParams]]) -> list[dict]:
    warnings.filterwarnings("ignore")
    ctx, lab = context_with_probs(trip_dir, probs)
    v, la = lab["label_valid"], lab["label_arc"]
    base = {"trip_id": ctx.trip_id, "n_points": len(ctx.xy), "n_lab": int(v.sum())}
    clf = argmax_per_point(ctx.cands, ctx.cands["prob"].to_numpy(), len(ctx.xy))
    cont = route_continuity(clf, ctx.xy, ctx.net, ctx.router)
    rows = [{**base, "config": "classifier_only", "correct": int((clf[v] == la[v]).sum()),
             "n_pairs": cont["n_pairs"], "n_jumps": cont["n_jumps"], "n_breaks": 0, "decode_s": 0.0}]
    for name, params in configs:
        t0 = time.perf_counter()
        res = decode(ctx.net, ctx.router, ctx.xy, ctx.t, ctx.cands, params)
        sec = time.perf_counter() - t0
        cont = route_continuity(res.point_arc, ctx.xy, ctx.net, ctx.router)
        rows.append({**base, "config": name, "correct": int((res.point_arc[v] == la[v]).sum()),
                     "n_pairs": cont["n_pairs"], "n_jumps": cont["n_jumps"],
                     "n_breaks": res.n_breaks, "decode_s": sec})
    return rows


def aggregate(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby("config", sort=False).agg(
        trips=("trip_id", "nunique"), labelled_points=("n_lab", "sum"), correct=("correct", "sum"),
        pairs=("n_pairs", "sum"), jumps=("n_jumps", "sum"), n_breaks=("n_breaks", "sum"),
        decode_s=("decode_s", "sum")).reset_index()
    g["accuracy"] = g["correct"] / g["labelled_points"]
    g["continuity"] = 1 - g["jumps"] / g["pairs"]
    return g


def run(trip_dirs, probs, configs, jobs) -> pd.DataFrame:
    res = Parallel(n_jobs=jobs)(delayed(eval_trip)(d, None if probs is None else probs[d.name], configs)
                                for d in trip_dirs)
    return aggregate([r for rows in res for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-trips", type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    train_dirs = list_trips(config.PARTITIONS["train"])
    test_dirs = list_trips(config.PARTITIONS["student_test"])
    rng = np.random.default_rng(config.SEED)
    subset = sorted(rng.choice(len(train_dirs), size=min(args.n_trips, len(train_dirs)), replace=False))
    subset_dirs = [train_dirs[i] for i in subset]
    oof, test = load_probs(args.model, "oof"), load_probs(args.model, "test")

    # 1. grid search on the training subset
    grid = [(f"beta={b}|K={k}", HMMParams(beta=float(b), max_states=k)) for b, k in itertools.product(BETAS, MAX_STATES)]
    tuning = run(subset_dirs, oof, grid, args.jobs)
    params_of = dict(grid)
    tuning["beta"] = tuning["config"].map(lambda c: params_of[c].beta if c in params_of else np.nan)
    tuning["max_states"] = tuning["config"].map(lambda c: params_of[c].max_states if c in params_of else np.nan)
    tuning.insert(0, "model", args.model)
    tuning.insert(1, "partition", f"train subset ({len(subset_dirs)} trips, OOF)")
    tuning.to_csv(OUT / f"hmm_tuning_{args.model}.csv", index=False)
    hmm_rows = tuning[tuning["config"] != "classifier_only"]
    best_name = hmm_rows.sort_values(["accuracy", "continuity"], ascending=False).iloc[0]["config"]
    best = params_of[best_name]
    print(tuning.sort_values("accuracy", ascending=False).head(8)[
        ["config", "accuracy", "continuity", "n_breaks", "decode_s"]].to_string(index=False), flush=True)

    # 2. best vs default vs classifier-only on all train trips and on student_test
    final_cfgs = [("hmm_default", HMMParams()), ("hmm_tuned", best)]
    summary = []
    for part, dirs, probs in [("train (80 trips, OOF)", train_dirs, oof), ("student_test (10 trips)", test_dirs, test)]:
        agg = run(dirs, probs, final_cfgs, args.jobs)
        agg.insert(0, "partition", part)
        summary.append(agg)
    summary = pd.concat(summary, ignore_index=True)
    summary.insert(0, "model", args.model)
    summary.to_csv(OUT / f"hmm_best_{args.model}.csv", index=False)
    print(summary[["partition", "config", "accuracy", "continuity", "n_breaks", "decode_s"]].to_string(index=False))

    def acc(part_prefix, cfg):
        s = summary[summary["partition"].str.startswith(part_prefix) & (summary["config"] == cfg)]
        return float(s["accuracy"].iloc[0])

    artifact = {
        **best.to_dict(),
        "model": args.model,
        "tuned_on": f"{len(subset_dirs)} training trips, out-of-fold probabilities",
        "objective": "pointwise accuracy over labelled points",
        "objective_value": float(hmm_rows.loc[hmm_rows["config"] == best_name, "accuracy"].iloc[0]),
        "train80_accuracy": {c: acc("train", c) for c in ["classifier_only", "hmm_default", "hmm_tuned"]},
        "student_test_accuracy": {c: acc("student_test", c) for c in ["classifier_only", "hmm_default", "hmm_tuned"]},
        "grid": {"beta": BETAS, "max_states": MAX_STATES},
        "seconds": time.perf_counter() - t_start,
    }
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    (config.ARTIFACTS_DIR / "hmm_config.json").write_text(json.dumps(artifact, indent=2))
    # per-model copy, so the configurations of competing classifiers can be compared later
    (config.RESULTS_DIR / "hmm" / f"hmm_config_{args.model}.json").write_text(json.dumps(artifact, indent=2))
    print(json.dumps(artifact, indent=1))


if __name__ == "__main__":
    main()
