"""Candidate generation: which arcs could each GPS point lie on?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .network import RoadNetwork

CAND_COLUMNS = ["pt", "arc", "dist", "frac", "frac_raw", "rank"]


def generate_candidates(net: RoadNetwork, xy: np.ndarray,
                        radius: float = config.CAND_RADIUS_M,
                        max_radius: float = config.CAND_MAX_RADIUS_M,
                        max_k: int = config.CAND_MAX_K) -> pd.DataFrame:
    """Candidate arcs for every point of a projected track.

    1. Take every arc within ``radius`` metres of the point (R-tree query).
    2. Points with no arc in range retry with a doubled radius, up to ``max_radius``.
    3. Keep the ``max_k`` closest arcs per point.

    Returns one row per (point, candidate arc), sorted by point then distance.
    ``frac`` is the position of the closest point along the arc (0 = start, 1 = end).
    """
    n = len(xy)
    pt, arc = net.query(xy, radius)
    has = np.zeros(n, dtype=bool)
    has[pt] = True
    r = radius
    while (~has).any() and r < max_radius:
        r = min(r * 2.0, max_radius)
        miss = np.flatnonzero(~has)
        p2, a2 = net.query(xy[miss], r)
        if len(p2):
            pt = np.concatenate([pt, miss[p2]])
            arc = np.concatenate([arc, a2])
            has[miss[p2]] = True

    dist, frac, frac_raw = net.project(xy[pt, 0], xy[pt, 1], arc)
    df = pd.DataFrame({"pt": pt, "arc": arc, "dist": dist, "frac": frac, "frac_raw": frac_raw})
    df = df.sort_values(["pt", "dist", "arc"], kind="mergesort")
    df["rank"] = df.groupby("pt").cumcount()
    df = df[df["rank"] < max_k].reset_index(drop=True)
    return df[CAND_COLUMNS]
