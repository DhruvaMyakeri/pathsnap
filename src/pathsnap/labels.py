"""Turning the reference route into one label per GPS point.

The ``.route`` file is an ordered list of arcs, with far fewer rows than the
GPS track, so there is no per-point label. We derive one with a monotone
alignment: every point is assigned to one route position, positions never
move backwards in time, and the summed point-to-arc distance is minimal.
Solved exactly by dynamic programming:

    D[0, j] = c[0, j]
    D[i, j] = c[i, j] + min_{j' <= j} D[i-1, j']

where c[i, j] = min(distance(point i, route arc j), cap). Skipping route arcs
is free (short arcs may get no GPS fix at 1 Hz). Monotonicity resolves loops
and U-turns, since the route lists the same street twice at different
positions. The cap stops GPS glitches from distorting the alignment.

Points farther than ``LABEL_MAX_DIST_M`` from their aligned arc get no label
(-1). They are excluded from training and scoring rather than given a label
we do not trust.
"""
from __future__ import annotations

import numpy as np

from . import config
from .geo import heading_reliable, window_motion
from .network import RoadNetwork


def _segment_dist(px, py, ax, ay, dx, dy):
    l2 = dx * dx + dy * dy
    safe = np.where(l2 > 0, l2, 1.0)
    f = np.clip(np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0), 0.0, 1.0)
    return np.hypot(px - (ax + f * dx), py - (ay + f * dy))


def motion_direction(xy: np.ndarray, k: int = 2) -> np.ndarray:
    """Unit direction of travel over the window [i-k, i+k].

    Returns a zero vector where the heading is unreliable (``geo.heading_reliable``):
    stops, stationary GPS jitter, and the apex of a U-turn.
    """
    chord, chord_len, path_len = window_motion(xy, k)
    ok = heading_reliable(chord_len, path_len)
    return np.where(ok[:, None], chord / np.maximum(chord_len, 1e-9)[:, None], 0.0)


def align_to_route(net: RoadNetwork, xy: np.ndarray, route: np.ndarray,
                   cost_cap: float = config.LABEL_COST_CAP_M,
                   heading_penalty_m: float = config.LABEL_HEADING_PENALTY_M, block: int = 256):
    """Monotone DP alignment of points to route positions.

    The per-point cost is the capped point-to-arc distance plus a direction
    penalty ``heading_penalty_m * max(0, -cos(angle))``, where the angle lies
    between the direction of travel and the arc. Twin arcs (two directions of
    one road) have identical geometry. Without this term, an out-and-back trip
    would tie between them, and the outbound fixes near the turn could be
    labelled with the return arc. The penalty is zero up to 90 deg, so ordinary
    turns (where the travel direction lies between the two arcs) are not
    affected. It is also zero when the vehicle is stationary (direction unknown).

    Returns ``(route_pos, dist)``: index into ``route`` per point and the
    distance (m) from the point to that arc.
    """
    route = np.asarray(route, dtype=np.int64)
    n, m = len(xy), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1, np.int64), np.full(n, np.inf)
    ax, ay = net.ax[route][None, :], net.ay[route][None, :]
    dx, dy = net.dx[route][None, :], net.dy[route][None, :]
    length = np.hypot(dx, dy)
    ux, uy = np.where(length > 0, dx / np.where(length > 0, length, 1), 0), \
        np.where(length > 0, dy / np.where(length > 0, length, 1), 0)
    motion = motion_direction(xy)
    ar = np.arange(m)
    back = np.empty((n, m), dtype=np.int16 if m < 32_000 else np.int32)
    prev = None
    for s in range(0, n, block):
        px, py = xy[s:s + block, 0:1], xy[s:s + block, 1:2]
        c = np.minimum(np.nan_to_num(_segment_dist(px, py, ax, ay, dx, dy), nan=cost_cap), cost_cap)
        if heading_penalty_m > 0:
            cos = motion[s:s + block, 0:1] * ux + motion[s:s + block, 1:2] * uy
            c = c + heading_penalty_m * np.maximum(0.0, -np.nan_to_num(cos))
        for r in range(c.shape[0]):
            i = s + r
            if prev is None:
                cur = c[r].copy()
                back[i] = ar
            else:
                run_min = np.minimum.accumulate(prev)
                # Index of the running minimum: last position where prev hits it.
                arg = np.maximum.accumulate(np.where(prev <= run_min, ar, 0))
                back[i] = arg
                cur = c[r] + run_min
            prev = cur
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    arcs = route[pos]
    dist = _segment_dist(xy[:, 0], xy[:, 1], net.ax[arcs], net.ay[arcs], net.dx[arcs], net.dy[arcs])
    return pos, dist


def point_labels(net: RoadNetwork, xy: np.ndarray, route: np.ndarray,
                 max_dist: float = config.LABEL_MAX_DIST_M) -> dict[str, np.ndarray]:
    """Per-point reference arc (``label_arc``, -1 = unlabeled) plus alignment diagnostics."""
    route = np.asarray(route, dtype=np.int64)
    ok = (route >= 0) & (route < net.n_arcs)
    route = route[ok]
    if len(route):
        route = route[net.valid[route]]
    pos, dist = align_to_route(net, xy, route)
    label = np.where(pos >= 0, route[np.maximum(pos, 0)] if len(route) else -1, -1)
    valid = (pos >= 0) & np.isfinite(dist) & (dist <= max_dist)
    return {
        "label_arc": np.where(valid, label, -1).astype(np.int64),
        "label_dist": dist,
        "route_pos": pos,
        "label_valid": valid,
    }
