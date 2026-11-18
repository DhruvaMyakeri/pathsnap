"""Candidate classifiers compared in the experiments (one per algorithm family).

All models solve the same binary task: given the features of one (GPS point,
candidate arc) pair, estimate P(this arc is the point's reference arc).
Probabilities feed both the Top-1 choice (argmax per point) and the HMM
emissions, so models must expose ``predict_proba``.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

from .config import SEED

FAMILIES = {
    "logistic_regression": "Linear",
    "gaussian_nb": "Probabilistic",
    "knn": "Instance-based",
    "random_forest": "Ensemble (bagging)",
    "lightgbm": "Boosting",
    "mlp": "Neural network",
}

DEFAULT_PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 2000},
    "gaussian_nb": {"var_smoothing": 1e-9},
    "knn": {"n_neighbors": 50, "weights": "distance"},
    "random_forest": {"n_estimators": 200, "min_samples_leaf": 20, "max_features": "sqrt"},
    "lightgbm": {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "mlp": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "batch_size": 1024,
            "learning_rate_init": 1e-3, "max_epochs": 40, "patience": 5},
}


class EpochMLP:
    """MLP trained epoch by epoch so training and validation loss can be recorded.

    Wraps sklearn's MLPClassifier: inputs are standardised, each epoch is
    one ``partial_fit`` pass over shuffled data, and training stops early
    when the validation log loss has not improved for ``patience`` epochs
    (the best epoch's weights are restored).
    """

    def __init__(self, hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024,
                 learning_rate_init=1e-3, max_epochs=40, patience=5, random_state=SEED):
        self.params = dict(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha, batch_size=batch_size,
                           learning_rate_init=learning_rate_init)
        self.max_epochs, self.patience, self.random_state = max_epochs, patience, random_state
        self.history: list[dict] = []

    def fit(self, X, y, X_val=None, y_val=None):
        import copy

        rng = np.random.default_rng(self.random_state)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X).astype(np.float32)
        y = np.asarray(y)
        Xv = self.scaler_.transform(X_val).astype(np.float32) if X_val is not None else None
        self.model_ = MLPClassifier(solver="adam", random_state=self.random_state, **self.params)
        best, best_state, stale = np.inf, None, 0
        self.history = []
        for epoch in range(1, self.max_epochs + 1):
            order = rng.permutation(len(Xs))
            self.model_.partial_fit(Xs[order], y[order], classes=np.array([0, 1]))
            row = {"epoch": epoch, "train_loss": log_loss(y, self.model_.predict_proba(Xs)[:, 1], labels=[0, 1])}
            if Xv is not None:
                row["val_loss"] = log_loss(y_val, self.model_.predict_proba(Xv)[:, 1], labels=[0, 1])
                if row["val_loss"] < best - 1e-5:
                    best, best_state, stale = row["val_loss"], copy.deepcopy(self.model_), 0
                else:
                    stale += 1
            self.history.append(row)
            if Xv is not None and stale >= self.patience:
                break
        if best_state is not None:
            self.model_ = best_state
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(X).astype(np.float32))


def make_model(name: str, params: dict | None = None, n_jobs: int = 12):
    p = dict(DEFAULT_PARAMS[name])
    p.update(params or {})
    if name == "logistic_regression":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(random_state=SEED, **p))])
    if name == "gaussian_nb":
        return GaussianNB(**p)
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=n_jobs, **p))])
    if name == "random_forest":
        return RandomForestClassifier(random_state=SEED, n_jobs=n_jobs, **p)
    if name == "lightgbm":
        return LGBMClassifier(random_state=SEED, n_jobs=n_jobs, verbose=-1, **p)
    if name == "mlp":
        return EpochMLP(**p)
    raise ValueError(f"unknown model {name}")
