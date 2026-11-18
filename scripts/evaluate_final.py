"""Final end-to-end evaluation of the deployed pipeline on the 10 student_test trips.

Runs exactly what the API runs: ``pipeline.predict_trip`` with the exported
model (artifacts/model.joblib) and HMM settings (artifacts/hmm_config.json),
then ``pipeline.score_trip``. It also writes one interactive Folium map per trip:
GPS observations, nearby road network, reference route, classifier-only route
and the final predicted route.

Outputs
  results/final/student_test_per_trip.csv
  results/final/student_test_summary.json
  results/maps/final_<trip_id>.html

Usage:  python scripts/evaluate_final.py [--partition student_test]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from pathsnap import config
from pathsnap.data import list_trips, load_route
from pathsnap.metrics import collapse
from pathsnap.pipeline import load_bundle, load_hmm_params, predict_trip, score_trip
from pathsnap.viz import trip_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test", choices=["student_test"])
    ap.add_argument("--no-maps", action="store_true")
    args = ap.parse_args()
    out = config.RESULTS_DIR / "final"
    maps = config.RESULTS_DIR / "maps"
    out.mkdir(parents=True, exist_ok=True)
    maps.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle()
    params = load_hmm_params()
    rows = []
    for trip_dir in list_trips(config.PARTITIONS[args.partition]):
        ctx, pred = predict_trip(trip_dir, bundle=bundle, hmm_params=params)
        s = score_trip(trip_dir, ctx, pred)
        rows.append({"trip_id": ctx.trip_id, "model": pred.model_name, "used_hmm": pred.used_hmm,
                     "n_breaks": pred.n_breaks, **s, **{f"time_{k}": v for k, v in pred.timings.items()}})
        print(f"{ctx.trip_id}: acc={s['accuracy']:.4f} (clf {s['accuracy_classifier_only']:.4f}) "
              f"routeF1={s['route_f1']:.4f} cont={s['continuity']:.4f} t={pred.timings['total_s']:.1f}s", flush=True)
        if not args.no_maps:
            ref = load_route(trip_dir)  # reference shown on report maps only (never sent to the UI)
            m = trip_map(ctx, predicted_route=pred.route_arcs, reference_route=ref,
                         classifier_route=collapse(pred.point_arc_clf),
                         title=f"Trip {ctx.trip_id} ({args.partition}) - accuracy {s['accuracy']:.1%}")
            m.save(str(maps / f"final_{ctx.trip_id}.html"))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{args.partition}_per_trip.csv", index=False)
    w = df["n_labelled_points"].to_numpy()
    summary = {
        "partition": args.partition, "model": df["model"].iloc[0], "used_hmm": bool(df["used_hmm"].all()),
        "hmm_params": params.to_dict() if hasattr(params, "to_dict") else None,
        "trips": len(df), "labelled_points": int(w.sum()),
        "pointwise_accuracy": float(np.average(df["accuracy"], weights=w)),
        "pointwise_accuracy_classifier_only": float(np.average(df["accuracy_classifier_only"], weights=w)),
        "direction_agnostic_accuracy": float(np.average(df["direction_agnostic_accuracy"], weights=w)),
        "mean_route_precision": float(df["route_precision"].mean()),
        "mean_route_recall": float(df["route_recall"].mean()),
        "mean_route_f1": float(df["route_f1"].mean()),
        "mean_continuity": float(df["continuity"].mean()),
        "mean_continuity_classifier_only": float(df["continuity_classifier_only"].mean()),
        "mean_seconds_per_trip": float(df["time_total_s"].mean()),
        "max_seconds_per_trip": float(df["time_total_s"].max()),
    }
    (out / f"{args.partition}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
