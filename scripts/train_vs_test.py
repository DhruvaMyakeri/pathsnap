"""Training (in-sample) vs cross-validated vs student-test Top-1 accuracy per model.

In-sample = the model refit on all 80 training trips, scored on the very rows it
was fitted on (the same every-n-th-fix subsample it was trained on). Comparing it
with the cross-validated score (each trip scored by a model that never saw it)
shows how much each model memorises its training data.

Output: results/models/train_vs_test.csv
Usage:  python scripts/train_vs_test.py
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from pathsnap import config
from pathsnap.experiment import DEFAULT_SUBSAMPLE, every_nth_point, labelled, load_partition, score
from pathsnap.features import FEATURES

MODELS = ["lightgbm", "random_forest", "mlp", "logistic_regression", "knn", "gaussian_nb"]


def main():
    train, train_pts = load_partition("train")
    rows = []
    for name in MODELS:
        model = joblib.load(config.CACHE_DIR / "models" / f"{name}.joblib")
        step = DEFAULT_SUBSAMPLE.get(name, 1)
        tr = every_nth_point(labelled(train), step)
        pts = every_nth_point(train_pts, step)  # score exactly the fixes the model was fitted on
        m = score(tr, pts, model.predict_proba(tr[FEATURES])[:, 1])
        run = json.loads((config.RESULTS_DIR / "models" / "runs" / f"{name}.json").read_text())
        rows.append({
            "model": name,
            "train_in_sample_top1": m["top1"], "train_in_sample_macro_f1": m["macro_f1"],
            "train_in_sample_log_loss": m["log_loss"],
            "cv_top1": run["cv"]["top1"][0], "cv_macro_f1": run["cv"]["macro_f1"][0],
            "test_top1": run["student_test"]["top1"], "test_macro_f1": run["student_test"]["macro_f1"],
            "train_rows_scored": len(tr),
        })
        print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out["gap_in_sample_minus_test_top1"] = out["train_in_sample_top1"] - out["test_top1"]
    out.to_csv(config.RESULTS_DIR / "models" / "train_vs_test.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
