"""Evaluation metrics: candidate classification and route reconstruction.

Definitions (used consistently in experiments, API and report):

* Top-1 accuracy: share of labelled GPS points whose highest-scoring candidate
  (or decoded arc) equals the reference arc. A point with no prediction counts as wrong.
* Candidate miss rate: share of labelled points whose reference arc is not in
  the candidate set. ``1 - miss rate`` is the best possible Top-1 accuracy.
* Classification metrics: binary candidate labels (1 = reference arc), with
  probability threshold 0.5. Macro F1 averages the F1 of both classes.
* Route continuity: share of consecutive point pairs whose predicted arcs are
  connected: the same arc, or the next arc reachable within
  ``50 m + 2 x`` the GPS step through the directed network.
* Route precision / recall / F1: length-weighted overlap between the arcs of
  the predicted route and the reference route.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from .network import RoadNetwork
from .routing import INF, Router


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(np.int8)
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-7, 1 - 1e-7)
    pred = (p >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_candidates(cands: pd.DataFrame, score_col: str = "prob",
                    keys: tuple[str, ...] = ("trip_id", "pt")) -> pd.DataFrame:
    """Highest-scoring candidate per point (ties go to the closer candidate)."""
    keys = [k for k in keys if k in cands.columns]
    order = cands.sort_values(keys + [score_col, "rank"], ascending=[True] * len(keys) + [False, True],
                              kind="mergesort")
    return order.drop_duplicates(keys, keep="first")[keys + ["arc"]]


def top1_accuracy(cands: pd.DataFrame, points: pd.DataFrame, score_col: str = "prob") -> float:
    """Top-1 accuracy over labelled points (``points`` has trip_id, pt, label_arc, label_valid)."""
    keys = [k for k in ("trip_id", "pt") if k in points.columns]
    best = best_candidates(cands, score_col, tuple(keys))
    lab = points.loc[points["label_valid"], keys + ["label_arc"]]
    merged = lab.merge(best, on=keys, how="left")
    return float((merged["arc"] == merged["label_arc"]).mean()) if len(merged) else float("nan")


def candidate_miss_rate(cands: pd.DataFrame, points: pd.DataFrame) -> float:
    keys = [k for k in ("trip_id", "pt") if k in points.columns]
    lab = points.loc[points["label_valid"], keys + ["label_arc"]]
    hit = lab.merge(cands[keys + ["arc"]], left_on=keys + ["label_arc"], right_on=keys + ["arc"], how="inner")
    return 1.0 - len(hit.drop_duplicates(keys)) / max(len(lab), 1)


def pointwise_accuracy(pred_arc: np.ndarray, label_arc: np.ndarray, valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    if not valid.any():
        return float("nan")
    return float((np.asarray(pred_arc)[valid] == np.asarray(label_arc)[valid]).mean())


def direction_agnostic_accuracy(pred_arc, label_arc, valid, net: RoadNetwork) -> float:
    """Accuracy that also accepts the reverse-direction twin (diagnostic only)."""
    valid = np.asarray(valid, dtype=bool)
    p, l = np.asarray(pred_arc)[valid], np.asarray(label_arc)[valid]
    twin_ok = (p >= 0) & (net.twin[np.maximum(l, 0)] == p)
    return float(((p == l) | twin_ok).mean()) if valid.any() else float("nan")


def route_continuity(pred_arc: np.ndarray, xy: np.ndarray, net: RoadNetwork, router: Router,
                     base_m: float = 50.0, factor: float = 2.0) -> dict:
    """Share of consecutive predicted arc pairs that are connected in the road network."""
    pred_arc = np.asarray(pred_arc)
    idx = np.flatnonzero((pred_arc[:-1] >= 0) & (pred_arc[1:] >= 0))
    if len(idx) == 0:
        return {"continuity": float("nan"), "n_pairs": 0, "n_jumps": 0}
    a, b = pred_arc[idx], pred_arc[idx + 1]
    step = np.hypot(*(xy[idx + 1] - xy[idx]).T)
    jumps = 0
    for k in np.flatnonzero(a != b):
        limit = base_m + factor * step[k]
        if router.node_distance(int(net.dst[a[k]]), int(net.src[b[k]]), limit) == INF:
            jumps += 1
    return {"continuity": 1.0 - jumps / len(idx), "n_pairs": int(len(idx)), "n_jumps": int(jumps)}


def collapse(arc_seq) -> np.ndarray:
    """Drop -1 entries and consecutive duplicates: per-point arcs -> arc sequence."""
    s = np.asarray(arc_seq)
    s = s[s >= 0]
    if len(s) == 0:
        return s
    return s[np.concatenate([[True], s[1:] != s[:-1]])]


def route_overlap(pred_route, ref_route, net: RoadNetwork) -> dict:
    """Length-weighted precision / recall / F1 between predicted and reference arc sets."""
    p = set(int(a) for a in np.asarray(pred_route) if a >= 0)
    r = set(int(a) for a in np.asarray(ref_route) if 0 <= a < net.n_arcs)
    length = np.nan_to_num(net.length, nan=0.0)
    lp = float(sum(length[a] for a in p))
    lr = float(sum(length[a] for a in r))
    lm = float(sum(length[a] for a in p & r))
    prec = lm / lp if lp > 0 else float("nan")
    rec = lm / lr if lr > 0 else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
