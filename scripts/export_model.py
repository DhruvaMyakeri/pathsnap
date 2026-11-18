"""Export the selected classifier as the deployment artifact used by the API.

The model was already refit on all 80 training trips by ``train_models.py``
(``data_cache/models/<run>.joblib``). This script wraps it with the
metadata the pipeline needs and writes ``artifacts/model.joblib``:

    {"model": estimator, "name": run, "model_type": ..., "features": FEATURES,
     "params": {...}, "candidate_config": {...}, "trained_on": "80 training trips",
     "selection_metrics": {...}}

Usage:  python scripts/export_model.py --run lightgbm_tuned
"""
from __future__ import annotations

import argparse
import json

import joblib

from pathsnap import config
from pathsnap.features import FEATURES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name in results/models/runs/")
    args = ap.parse_args()
    run = json.loads((config.RESULTS_DIR / "models" / "runs" / f"{args.run}.json").read_text())
    model = joblib.load(config.CACHE_DIR / "models" / f"{args.run}.joblib")
    bundle = {
        "model": model,
        "name": args.run,
        "model_type": run.get("model_type", args.run),
        "features": FEATURES,
        "params": run["params"],
        "seed": run["seed"],
        "candidate_config": {"radius_m": config.CAND_RADIUS_M, "max_radius_m": config.CAND_MAX_RADIUS_M,
                             "max_k": config.CAND_MAX_K},
        "trained_on": "all 80 training trips (labelled points)",
        "selection_metrics": {"cv": run["cv"], "student_test": run["student_test"]},
    }
    config.ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.ARTIFACTS_DIR / "model.joblib", compress=3)
    meta = {k: v for k, v in bundle.items() if k != "model"}
    (config.ARTIFACTS_DIR / "model_card.json").write_text(json.dumps(meta, indent=2))
    size_mb = (config.ARTIFACTS_DIR / "model.joblib").stat().st_size / 1e6
    print(f"exported {args.run} -> artifacts/model.joblib ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
