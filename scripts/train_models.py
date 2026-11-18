"""Train and compare candidate classifiers (6 model families + 2 non-learned baselines).

Protocol (no student_test or evaluator data is used for fitting or selection):
  * Training partition (80 trips): 5-fold GroupKFold grouped by trip, giving
    out-of-fold (OOF) probabilities. Metrics are computed per fold and
    reported as mean +/- std.
  * Each model is then refit on all 80 training trips and scored on the 10
    student_test trips (development-stage comparison).
  * Only labelled points are used. All candidates of a point stay together.
  * Slow models may train on every s-th GPS point per trip (consecutive 1 Hz
    fixes are near-duplicates). Evaluation always uses every labelled point.

Outputs
  results/models/runs/<model>.json   config, seed, timings, CV + test metrics (experiment log)
  results/models/cv_folds.csv        per-fold metrics of every model
  results/models/model_comparison.csv  one row per model (written by --summarize)
  results/models/curves/<model>_loss_curve.csv  epoch/iteration-wise losses (MLP, LightGBM)
  data_cache/preds/oof_<model>.parquet, test_<model>.parquet   (trip_id, pt, arc, prob)
  data_cache/models/<model>.joblib   model refit on all training trips

Usage
  python scripts/train_models.py --models logistic_regression gaussian_nb
  python scripts/train_models.py --summarize
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from pathsnap import config
from pathsnap.experiment import DEFAULT_SUBSAMPLE, every_nth_point, labelled, load_partition, score, trip_folds
from pathsnap.features import FEATURES
from pathsnap.models import DEFAULT_PARAMS, FAMILIES, make_model
from pathsnap.pipeline import heuristic_prob

OUT = config.RESULTS_DIR / "models"
PRED_DIR = config.CACHE_DIR / "preds"
MODEL_DIR = config.CACHE_DIR / "models"
BASELINES = {"baseline_nearest_arc": "Non-learned rule", "baseline_distance_heading": "Non-learned rule"}


def baseline_prob(name: str, cands: pd.DataFrame) -> np.ndarray:
    if name == "baseline_nearest_arc":
        return (cands["rank"].to_numpy() == 0).astype(np.float64)
    s = heuristic_prob(cands)
    # hard decision: the best-scoring candidate of each point is "positive"
    df = pd.DataFrame({"trip_id": cands["trip_id"].to_numpy(), "pt": cands["pt"].to_numpy(), "s": s})
    best = df.groupby(["trip_id", "pt"], observed=True)["s"].transform("max").to_numpy()
    return (s >= best).astype(np.float64)


def fit(model_type: str, params: dict | None, model, X, y, groups, record_curve: bool):
    """Fit a model; MLP/LightGBM also record epoch/iteration-wise losses on an internal
    trip-grouped 90/10 split of the training data (never on the evaluation fold)."""
    curve = None
    if model_type in ("mlp", "lightgbm"):
        tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=config.SEED)
                      .split(X, y, groups))
        if model_type == "mlp":
            model.fit(X.iloc[tr], y[tr], X.iloc[va], y[va])
            curve = pd.DataFrame(model.history)
        else:
            if record_curve:
                ev = {}
                curve_model = make_model("lightgbm", params)
                import lightgbm as lgb
                curve_model.fit(X.iloc[tr], y[tr], eval_X=(X.iloc[tr], X.iloc[va]), eval_y=(y[tr], y[va]),
                                eval_names=["train", "valid"], eval_metric="binary_logloss",
                                callbacks=[lgb.record_evaluation(ev)])
                curve = pd.DataFrame({"iteration": np.arange(1, len(ev["train"]["binary_logloss"]) + 1),
                                      "train_loss": ev["train"]["binary_logloss"],
                                      "val_loss": ev["valid"]["binary_logloss"]})
            model.fit(X, y)
    else:
        model.fit(X, y)
    return model, curve


def run_model(name: str, train, train_pts, test, test_pts, subsample: int, params: dict | None,
              model_type: str | None = None):
    """Run the full CV + student_test protocol. ``name`` labels the run (e.g. ``lightgbm_tuned``);
    ``model_type`` picks the estimator (defaults to ``name``)."""
    model_type = model_type or name
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    is_baseline = name in BASELINES
    folds = trip_folds(train)
    oof = np.zeros(len(train))
    fold_rows = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        va = train.iloc[va_idx]
        t0 = time.perf_counter()
        if is_baseline:
            p = baseline_prob(name, va)
            fit_s = 0.0
        else:
            tr = every_nth_point(labelled(train.iloc[tr_idx]), subsample)
            model, _ = fit(model_type, params, make_model(model_type, params), tr[FEATURES], tr["y"].to_numpy(),
                           tr["trip_id"].astype(str).to_numpy(), record_curve=False)
            fit_s = time.perf_counter() - t0
            p = model.predict_proba(va[FEATURES])[:, 1]
        oof[va_idx] = p
        m = score(va, train_pts, p, probabilistic=not is_baseline)
        m.update({"model": name, "fold": k, "fit_s": fit_s, "n_train_rows": 0 if is_baseline else len(tr)})
        fold_rows.append(m)
        print(f"[{name}] fold {k}: top1={m['top1']:.4f} macroF1={m['macro_f1']:.4f} "
              f"logloss={m['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)

    # refit on all training trips, evaluate on student_test
    t0 = time.perf_counter()
    curve = None
    if is_baseline:
        model, fit_s = None, 0.0
    else:
        tr = every_nth_point(labelled(train), subsample)
        model, curve = fit(model_type, params, make_model(model_type, params), tr[FEATURES], tr["y"].to_numpy(),
                           tr["trip_id"].astype(str).to_numpy(), record_curve=True)
        fit_s = time.perf_counter() - t0
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    t1 = time.perf_counter()
    test_p = baseline_prob(name, test) if is_baseline else model.predict_proba(test[FEATURES])[:, 1]
    infer_s = time.perf_counter() - t1
    test_m = score(test, test_pts, test_p, probabilistic=not is_baseline)
    if curve is not None:
        curve.to_csv(OUT / "curves" / f"{name}_loss_curve.csv", index=False)

    cols = ["trip_id", "pt", "arc"]
    pd.concat([train[cols].reset_index(drop=True), pd.Series(oof, name="prob")], axis=1) \
        .to_parquet(PRED_DIR / f"oof_{name}.parquet", index=False)
    pd.concat([test[cols].reset_index(drop=True), pd.Series(test_p, name="prob")], axis=1) \
        .to_parquet(PRED_DIR / f"test_{name}.parquet", index=False)

    folds_df = pd.DataFrame(fold_rows)
    n_test_points = len(test_pts)  # inference runs on every candidate of every point
    run = {
        "model": name,
        "model_type": model_type,
        "family": FAMILIES.get(model_type, BASELINES.get(name)),
        "params": {} if is_baseline else {**DEFAULT_PARAMS.get(model_type, {}), **(params or {})},
        "seed": config.SEED, "features": FEATURES, "train_subsample_every_nth_point": subsample,
        "cv": {m: [float(folds_df[m].mean()), float(folds_df[m].std())]
               for m in ["top1", "macro_f1", "f1", "precision", "recall", "accuracy", "log_loss"]},
        "student_test": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in test_m.items()},
        "fit_seconds_full_train": fit_s,
        "inference_seconds_student_test": infer_s,
        "inference_ms_per_1000_points": 1e6 * infer_s / max(n_test_points, 1),
        "n_train_rows_full": 0 if is_baseline else int(len(every_nth_point(labelled(train), subsample))),
        "n_test_rows_labelled": int(test["label_valid"].sum()),
    }
    (OUT / "runs" / f"{name}.json").write_text(json.dumps(run, indent=2))
    folds_df.to_csv(OUT / "runs" / f"{name}_folds.csv", index=False)
    print(f"[{name}] student_test top1={test_m['top1']:.4f} macroF1={test_m['macro_f1']:.4f} "
          f"logloss={test_m['log_loss']:.4f} infer={infer_s:.2f}s", flush=True)


def summarize():
    rows, folds = [], []
    for f in sorted((OUT / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        rows.append({
            "model": r["model"], "family": r["family"],
            "key_config": json.dumps(r["params"]) if r["params"] else "rule-based",
            "train_subsample": r["train_subsample_every_nth_point"],
            "cv_top1_mean": r["cv"]["top1"][0], "cv_top1_std": r["cv"]["top1"][1],
            "cv_macro_f1_mean": r["cv"]["macro_f1"][0], "cv_macro_f1_std": r["cv"]["macro_f1"][1],
            "cv_log_loss_mean": r["cv"]["log_loss"][0],
            "test_top1": r["student_test"]["top1"], "test_macro_f1": r["student_test"]["macro_f1"],
            "test_precision": r["student_test"]["precision"], "test_recall": r["student_test"]["recall"],
            "test_f1": r["student_test"]["f1"], "test_accuracy": r["student_test"]["accuracy"],
            "test_log_loss": r["student_test"]["log_loss"],
            "fit_s": r["fit_seconds_full_train"], "infer_s_student_test": r["inference_seconds_student_test"],
        })
        folds.append(pd.read_csv(OUT / "runs" / f"{r['model']}_folds.csv"))
    pd.DataFrame(rows).sort_values("cv_top1_mean", ascending=False).to_csv(OUT / "model_comparison.csv", index=False)
    pd.concat(folds).to_csv(OUT / "cv_folds.csv", index=False)
    print(pd.DataFrame(rows).sort_values("cv_top1_mean", ascending=False).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="model types to run with default params")
    ap.add_argument("--params", help="JSON file of hyperparameters (e.g. Optuna best params) for --models[0]")
    ap.add_argument("--name", help="run name for --params runs, e.g. lightgbm_tuned")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.models:
        train, train_pts = load_partition("train")
        test, test_pts = load_partition("student_test")
        print(f"train rows={len(train):,} test rows={len(test):,}", flush=True)
        if args.params:
            model_type = args.models[0]
            params = json.loads(open(args.params).read())["params"]
            run_model(args.name or f"{model_type}_tuned", train, train_pts, test, test_pts,
                      DEFAULT_SUBSAMPLE.get(model_type, 1), params, model_type=model_type)
        else:
            for name in args.models:
                run_model(name, train, train_pts, test, test_pts, DEFAULT_SUBSAMPLE.get(name, 1), None)
    if args.summarize:
        summarize()


if __name__ == "__main__":
    main()
