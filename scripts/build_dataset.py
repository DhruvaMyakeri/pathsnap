"""Build candidate / feature / label tables for the train and student_test partitions.

For every trip: clean + project the GPS track, build the road network,
generate candidates, compute features, align the reference route to the
points (labels), and mark each candidate row y = 1 if it is the point's
reference arc.

Outputs
  data_cache/<partition>_cands.parquet   one row per (trip, point, candidate arc)   [git-ignored]
  data_cache/<partition>_points.parquet  one row per usable GPS point                [git-ignored]
  results/dataset/trip_summary.csv       per-trip counts, label coverage, candidate miss rate
  results/dataset/dataset_summary.json   partition-level totals

Usage:  python scripts/build_dataset.py [--partitions train student_test] [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap import config
from pathsnap.data import list_trips, load_route
from pathsnap.labels import point_labels
from pathsnap.pipeline import prepare_trip


def process_trip(trip_dir, partition: str):
    t0 = time.perf_counter()
    ctx = prepare_trip(trip_dir)
    lab = point_labels(ctx.net, ctx.xy, load_route(trip_dir))
    n = len(ctx.xy)

    points = pd.DataFrame({
        "trip_id": ctx.trip_id, "pt": np.arange(n), "raw_idx": ctx.keep,
        "x": ctx.xy[:, 0], "y": ctx.xy[:, 1], "t": ctx.t,
        "label_arc": lab["label_arc"], "label_dist": lab["label_dist"],
        "label_valid": lab["label_valid"], "route_pos": lab["route_pos"],
    })
    cands = ctx.cands
    cands.insert(0, "trip_id", ctx.trip_id)
    pt = cands["pt"].to_numpy()
    cands["y"] = (cands["arc"].to_numpy() == lab["label_arc"][pt]).astype(np.int8)
    cands["label_valid"] = lab["label_valid"][pt]

    has_cand = np.zeros(n, dtype=bool)
    has_cand[pt] = True
    hit = np.zeros(n, dtype=bool)
    hit[pt[cands["y"].to_numpy() == 1]] = True
    valid = lab["label_valid"]
    summary = {
        "partition": partition, "trip_id": ctx.trip_id,
        "n_raw_points": ctx.n_raw, "n_points": n, "n_dropped_points": ctx.n_raw - n,
        "n_nodes": ctx.net.n_nodes, "n_arcs": ctx.net.n_arcs,
        "n_cand_rows": len(cands), "mean_cands_per_point": len(cands) / max(n, 1),
        "points_without_candidates": int((~has_cand).sum()),
        "labelled_share": float(valid.mean()) if n else float("nan"),
        "median_label_dist_m": float(np.median(lab["label_dist"][valid])) if valid.any() else float("nan"),
        "p95_label_dist_m": float(np.percentile(lab["label_dist"][valid], 95)) if valid.any() else float("nan"),
        "candidate_miss_rate": float(1 - hit[valid].mean()) if valid.any() else float("nan"),
        "utm_epsg": ctx.projector.epsg, "seconds": time.perf_counter() - t0,
    }
    return points, cands, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    config.CACHE_DIR.mkdir(exist_ok=True)
    out_dir = config.RESULTS_DIR / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries, totals = [], {}
    for part in args.partitions:
        trips = list_trips(config.PARTITIONS[part])
        t0 = time.perf_counter()
        results = Parallel(n_jobs=args.jobs, verbose=0)(delayed(process_trip)(d, part) for d in trips)
        points = pd.concat([r[0] for r in results], ignore_index=True)
        cands = pd.concat([r[1] for r in results], ignore_index=True)
        for df in (points, cands):
            df["trip_id"] = df["trip_id"].astype("category")
        points.to_parquet(config.CACHE_DIR / f"{part}_points.parquet", index=False)
        cands.to_parquet(config.CACHE_DIR / f"{part}_cands.parquet", index=False)
        summaries += [r[2] for r in results]
        valid = points["label_valid"].to_numpy()
        pos_pts = cands.loc[cands["y"] == 1, ["trip_id", "pt"]].drop_duplicates()
        totals[part] = {
            "trips": len(trips), "points": len(points), "labelled_points": int(valid.sum()),
            "labelled_share": float(valid.mean()), "candidate_rows": len(cands),
            "positive_rows": int(cands["y"].sum()),
            "candidate_miss_rate": 1 - len(pos_pts) / max(int(valid.sum()), 1),
            "build_seconds": time.perf_counter() - t0,
        }
        print(part, json.dumps(totals[part], indent=1), flush=True)

    pd.DataFrame(summaries).to_csv(out_dir / "trip_summary.csv", index=False)
    totals["config"] = {
        "CAND_RADIUS_M": config.CAND_RADIUS_M, "CAND_MAX_RADIUS_M": config.CAND_MAX_RADIUS_M,
        "CAND_MAX_K": config.CAND_MAX_K, "LABEL_MAX_DIST_M": config.LABEL_MAX_DIST_M,
        "LABEL_COST_CAP_M": config.LABEL_COST_CAP_M,
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
