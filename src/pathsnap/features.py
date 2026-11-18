"""Features for each (GPS point, candidate arc) pair.

Groups and why they are there:

* Geometry: how close the point is to the arc, absolutely and relative to
  the other candidates of the same point (the nearest arc is often right but not always).
* Direction: the angle between the vehicle's direction of travel and the
  arc's direction. This is the only signal that separates a two-way road's
  twin arcs, since they share identical geometry. Heading is undefined when the
  vehicle does not move, so the angle is set to a neutral 90 deg when the net
  displacement over the window is below ``geo.HEADING_MIN_DISP_M``. Path
  straightness is given as a feature, so the model can learn to discount
  heading in stationary jitter clouds. (A hard straightness gate was tried and
  hurt Top-1: it also removed the useful heading of slow, noisy driving.)
* Kinematics: speed, displacement, path straightness (heading reliability)
  and sampling gaps.
* Context: how well the same arc explains the neighbouring GPS points.
  A single noisy fix can sit closer to a side street, but its neighbours usually do not.
* Topology: arc length, where along the arc the point projects, whether the
  road is two-way, and how many arcs meet at its ends (intersection complexity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .geo import HEADING_MIN_DISP_M, angle_diff, compass_bearing, straightness, window_motion
from .network import RoadNetwork

FEATURES = [
    "dist", "rank", "dist_minus_min", "dist_ratio", "n_cands", "n_close",
    "heading_diff", "cos_heading_diff", "heading_diff_long",
    "speed", "disp", "straightness", "straightness_long", "dt_prev", "dt_next",
    "arc_length", "frac", "beyond_m", "has_twin", "out_degree_end", "in_degree_start",
    "ctx_mean_dist", "ctx_max_dist",
]

CONTEXT_OFFSETS = (-2, -1, 1, 2)
CLOSE_MARGIN_M = 10.0


def _kinematics(xy: np.ndarray, t: np.ndarray, k: int):
    """Heading, displacement, speed, straightness and 'heading defined' over [i-k, i+k]."""
    n = len(xy)
    idx = np.arange(n)
    dt = t[np.clip(idx + k, 0, n - 1)] - t[np.clip(idx - k, 0, n - 1)]
    chord, disp, path = window_motion(xy, k)
    speed = np.where(dt > 0, disp / np.where(dt > 0, dt, 1.0), 0.0)
    return (compass_bearing(chord[:, 0], chord[:, 1]), disp, speed,
            straightness(disp, path), disp >= HEADING_MIN_DISP_M)


def point_features(xy: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    heading, disp, speed, straight, reliable = _kinematics(xy, t, 2)
    heading_long, _, _, straight_long, reliable_long = _kinematics(xy, t, 5)
    dt = np.diff(t)
    return pd.DataFrame({
        "heading": heading, "disp": disp, "speed": speed, "straightness": straight, "reliable": reliable,
        "heading_long": heading_long, "straightness_long": straight_long, "reliable_long": reliable_long,
        "dt_prev": np.concatenate([[1.0], dt]), "dt_next": np.concatenate([dt, [1.0]]),
    })


def build_features(net: RoadNetwork, xy: np.ndarray, t: np.ndarray, cands: pd.DataFrame) -> pd.DataFrame:
    """Candidate table + one column per name in ``FEATURES`` (float32)."""
    pf = point_features(xy, t)
    n = len(xy)
    pt = cands["pt"].to_numpy()
    arc = cands["arc"].to_numpy()
    dist = cands["dist"].to_numpy()

    by_pt = cands.groupby("pt")["dist"]
    min_d = by_pt.transform("min").to_numpy()
    n_cands = by_pt.transform("size").to_numpy()
    close = (dist <= min_d + CLOSE_MARGIN_M).astype(np.int32)
    n_close = pd.Series(close).groupby(pt).transform("sum").to_numpy()

    bearing = net.bearing[arc]
    heading_diff = angle_diff(pf["heading"].to_numpy()[pt], bearing)
    heading_diff = np.where(pf["reliable"].to_numpy()[pt], heading_diff, 90.0)
    heading_diff_long = angle_diff(pf["heading_long"].to_numpy()[pt], bearing)
    heading_diff_long = np.where(pf["reliable_long"].to_numpy()[pt], heading_diff_long, 90.0)

    ctx = np.empty((len(CONTEXT_OFFSETS), len(pt)))
    for k, o in enumerate(CONTEXT_OFFSETS):
        q = np.clip(pt + o, 0, n - 1)
        ctx[k], _, _ = net.project(xy[q, 0], xy[q, 1], arc)

    length = net.length[arc]
    feats = pd.DataFrame({
        "dist": dist,
        "rank": cands["rank"].to_numpy(),
        "dist_minus_min": dist - min_d,
        "dist_ratio": dist / (min_d + 1.0),
        "n_cands": n_cands,
        "n_close": n_close,
        "heading_diff": heading_diff,
        "cos_heading_diff": np.cos(np.radians(heading_diff)),
        "heading_diff_long": heading_diff_long,
        "speed": pf["speed"].to_numpy()[pt],
        "disp": pf["disp"].to_numpy()[pt],
        "straightness": pf["straightness"].to_numpy()[pt],
        "straightness_long": pf["straightness_long"].to_numpy()[pt],
        "dt_prev": pf["dt_prev"].to_numpy()[pt],
        "dt_next": pf["dt_next"].to_numpy()[pt],
        "arc_length": length,
        "frac": cands["frac"].to_numpy(),
        "beyond_m": np.abs(cands["frac_raw"].to_numpy() - cands["frac"].to_numpy()) * length,
        "has_twin": (net.twin[arc] >= 0).astype(np.float64),
        "out_degree_end": net.out_degree[net.dst[arc]],
        "in_degree_start": net.in_degree[net.src[arc]],
        "ctx_mean_dist": ctx.mean(axis=0),
        "ctx_max_dist": ctx.max(axis=0),
    })[FEATURES].astype(np.float32)
    out = pd.concat([cands[["pt", "arc"]].reset_index(drop=True), feats], axis=1)
    out["frac_exact"] = cands["frac"].to_numpy()  # float64 position along the arc, used by the HMM
    return out
