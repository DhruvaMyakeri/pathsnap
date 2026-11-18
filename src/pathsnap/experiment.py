"""Shared experiment helpers so training, tuning and evaluation use one protocol.

* Data: cached candidate tables from ``scripts/build_dataset.py``.
* Fitting uses labelled rows only; prediction covers every row, so the HMM has
  scores for every point.
* Slow models can train on every s-th GPS point per trip. Consecutive 1 Hz
  fixes are near-duplicates, so this removes redundancy rather than information.
* Scoring is always on all labelled rows / points.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config
from .metrics import classification_metrics, top1_accuracy

N_FOLDS = 5
# Train on every s-th GPS point (per trip) for models whose cost grows quickly with data size.
DEFAULT_SUBSAMPLE = {"logistic_regression": 1, "gaussian_nb": 1, "knn": 12, "random_forest": 2,
                     "lightgbm": 1, "mlp": 4}


def load_partition(part: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All candidate rows (labelled or not) + the per-point label table of a partition."""
    cands = pd.read_parquet(config.CACHE_DIR / f"{part}_cands.parquet")
    points = pd.read_parquet(config.CACHE_DIR / f"{part}_points.parquet",
                             columns=["trip_id", "pt", "label_arc", "label_valid"])
    # identical categories on both tables so merges on trip_id stay categorical
    cats = sorted(set(points["trip_id"].astype(str)) | set(cands["trip_id"].astype(str)))
    for df in (cands, points):
        df["trip_id"] = pd.Categorical(df["trip_id"].astype(str), categories=cats)
    return cands, points


def trip_folds(cands: pd.DataFrame, n_splits: int = N_FOLDS):
    """GroupKFold by trip: all rows of a trip land in the same fold (no leakage between folds)."""
    groups = cands["trip_id"].astype(str).to_numpy()
    return list(GroupKFold(n_splits=n_splits).split(cands, groups=groups))


def every_nth_point(df: pd.DataFrame, s: int) -> pd.DataFrame:
    return df if s <= 1 else df[df["pt"].to_numpy() % s == 0]


def labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["label_valid"].to_numpy()]


def score(cands: pd.DataFrame, points: pd.DataFrame, prob: np.ndarray, probabilistic: bool = True) -> dict:
    """Metrics on the labelled rows of ``cands`` (``prob`` is aligned with all rows)."""
    lab = cands["label_valid"].to_numpy()
    c, p = cands[lab], np.asarray(prob)[lab]
    m = classification_metrics(c["y"].to_numpy(), p)
    if not probabilistic:
        m["log_loss"] = float("nan")
    df = c[["trip_id", "pt", "arc", "rank"]].copy()
    df["prob"] = p
    m["top1"] = top1_accuracy(df, points[points["trip_id"].isin(df["trip_id"].unique())])
    m["n_labelled_points"] = int(df[["trip_id", "pt"]].drop_duplicates().shape[0])
    return m
