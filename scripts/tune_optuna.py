"""Bayesian hyperparameter optimisation of the selected candidate classifier (Optuna).

Search strategy: Optuna's Tree-structured Parzen Estimator (TPE), a sequential
model-based (Bayesian) optimiser. After a few random start-up trials it models
the density of good and bad configurations from all completed trials, and
proposes the configuration that maximises the expected improvement. The
multivariate variant models parameter interactions (e.g. learning rate x
number of trees).

Objective: mean observation-level Top-1 accuracy over trip-grouped folds of
the 80 TRAINING trips. Top-1 is the primary selection metric because the
system outputs one arc per GPS point. Macro F1 and log loss of every trial are
stored as trial attributes. Student-test and evaluator data are never touched.

Search budget, to keep the search affordable:
* 3 GroupKFold folds;
* fitting on every 2nd labelled GPS point (validation always uses every
  labelled point of the held-out trips);
* MedianPruner stops trials clearly below the median after the first fold.

The best configuration is then re-run through the exact protocol of
``train_models.py`` (5 folds, same subsampling, refit on all 80 trips, scored
on student_test), so tuned and untuned are compared on identical splits.

Outputs (results/optuna/)
  <model>_study.db              Optuna SQLite storage: the full search record
  <model>_trials.csv            one row per trial: params, value, state, duration, metrics
  best_params_<model>.json
  <model>_history.png, <model>_importance.png

Usage:  python scripts/tune_optuna.py --model lightgbm --trials 40
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import optuna  # noqa: E402

from pathsnap import config  # noqa: E402
from pathsnap.experiment import every_nth_point, labelled, load_partition, score, trip_folds  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.models import make_model  # noqa: E402

OUT = config.RESULTS_DIR / "optuna"
SEARCH_FOLDS = 3
SEARCH_SUBSAMPLE = 2


def space_lightgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 1200, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def space_random_forest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 8, 40),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 100, log=True),
        "max_features": trial.suggest_float("max_features", 0.2, 0.9),
    }


SPACES = {"lightgbm": space_lightgbm, "random_forest": space_random_forest}
# Why these ranges (lightgbm): num_leaves / min_child_samples bound tree complexity
# (over- vs under-fitting on ~1.7M rows); learning_rate x n_estimators trade
# optimisation speed for accuracy; subsample / colsample_bytree / L1 / L2 add
# regularisation. All scale-type parameters are searched on a log scale.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lightgbm", choices=sorted(SPACES))
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    train, train_pts = load_partition("train")
    folds = trip_folds(train, n_splits=SEARCH_FOLDS)
    fold_data = []
    for tr_idx, va_idx in folds:
        tr = every_nth_point(labelled(train.iloc[tr_idx]), SEARCH_SUBSAMPLE)
        fold_data.append((tr[FEATURES], tr["y"].to_numpy(), train.iloc[va_idx]))

    def objective(trial: optuna.Trial) -> float:
        params = SPACES[args.model](trial)
        tops, f1s, lls = [], [], []
        t0 = time.perf_counter()
        for k, (X, y, va) in enumerate(fold_data):
            model = make_model(args.model, params, n_jobs=args.n_jobs)
            model.fit(X, y)
            m = score(va, train_pts, model.predict_proba(va[FEATURES])[:, 1])
            tops.append(m["top1"])
            f1s.append(m["macro_f1"])
            lls.append(m["log_loss"])
            trial.report(float(np.mean(tops)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1", float(np.mean(f1s)))
        trial.set_user_attr("cv_log_loss", float(np.mean(lls)))
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(tops))

    storage = f"sqlite:///{(OUT / f'{args.model}_study.db').as_posix()}"
    study = optuna.create_study(
        study_name=f"{args.model}_top1", storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.SEED, multivariate=True, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0),
    )
    remaining = args.trials - len([t for t in study.trials if t.state.is_finished()])
    t0 = time.perf_counter()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    search_s = time.perf_counter() - t0

    df = study.trials_dataframe()
    df.to_csv(OUT / f"{args.model}_trials.csv", index=False)
    best = {
        "model": args.model, "objective": "mean CV Top-1 accuracy (3 trip-grouped folds, train)",
        "best_value": study.best_value, "best_trial": study.best_trial.number,
        "params": study.best_params, "n_trials": len(study.trials),
        "n_complete": int((df["state"] == "COMPLETE").sum()), "n_pruned": int((df["state"] == "PRUNED").sum()),
        "search_seconds_this_run": search_s,
        "search_setup": {"folds": SEARCH_FOLDS, "train_every_nth_point": SEARCH_SUBSAMPLE,
                         "sampler": "TPE (multivariate, 8 random start-up trials)", "pruner": "Median"},
    }
    (OUT / f"best_params_{args.model}.json").write_text(json.dumps(best, indent=2))

    ax = optuna.visualization.matplotlib.plot_optimization_history(study)
    ax.set_title(f"Optuna optimisation history ({args.model}, objective = CV Top-1)")
    plt.gcf().set_size_inches(9, 4.5)
    plt.tight_layout()
    plt.savefig(OUT / f"{args.model}_history.png", dpi=130)
    plt.close("all")
    try:
        ax = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.gcf().set_size_inches(8, 4.5)
        plt.tight_layout()
        plt.savefig(OUT / f"{args.model}_importance.png", dpi=130)
        plt.close("all")
    except Exception as exc:  # importance needs enough completed trials
        print("param importance plot skipped:", exc)
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
