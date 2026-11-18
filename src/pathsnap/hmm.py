"""HMM + Viterbi route reconstruction.

Why an HMM on top of the classifier
-----------------------------------
The classifier scores each (GPS point, candidate arc) pair independently. Its
per-point argmax can therefore jump: onto a parallel side street for one noisy
fix, or onto the reverse-direction twin of the road (identical geometry). The
result is a route no vehicle could drive. A hidden Markov model adds the two
facts the classifier cannot see: consecutive fixes are only seconds apart, and
the vehicle must move along the directed road network between them.

Model
-----
* Hidden state at GPS point i: the arc the vehicle is on, restricted to the
  ``max_states`` candidates with the highest classifier probability (states
  are (arc, position along the arc) pairs, since each candidate carries the
  projection ``frac_exact`` of the fix onto the arc).
* Emission score: ``emission_weight * log(max(prob, min_prob))``. The classifier
  posterior is used as a scaled likelihood, as in hybrid NN/HMM decoders; any
  per-point constant cancels in the arg-max, so the probabilities need not be
  normalised per point.
* Transition score (Newson & Krumm, 2009): if the vehicle really went from
  candidate a (at frac_a) to candidate b (at frac_b), the driving distance
  ``d_route`` through the network should be close to the straight-line
  distance ``d_gps`` between the two fixes. Hence

      log p(a -> b) = -|d_route - d_gps| / beta_eff

  ``d_route`` follows the same semantics as ``Router.route_distance``: the rest
  of arc a, plus the shortest node path, plus the start of arc b. Moving
  backwards along the same arc by up to ``back_tolerance_m`` is treated as GPS
  jitter; a larger backward move requires driving around. A transition is
  impossible (-inf) when no path exists within the search limit, or when it
  implies a speed above ``max_speed_mps``. This is what rules out jumps
  between disconnected streets and instant U-turns onto the twin arc.
* GPS gaps: with ``dt > gap_s`` the vehicle may have driven a long detour that
  the straight line does not reflect, and |d_route - d_gps| grows with the
  distance driven. Both beta and the Dijkstra search radius are scaled by
  ``g = dt / gap_s``, so legitimate detours after a dropout stay feasible and
  are not over-penalised, while dense 1 Hz steps keep the strict beta.
  Search limit: ``min(1.5 * d_gps + search_slack_m * g, max_speed_mps * dt)``.
  Here 1.5 is a generous road-circuity factor, and anything beyond the speed
  bound is infeasible anyway.

Note: only the ratio ``emission_weight * beta`` matters for the decoded path
(scaling every score by a constant does not change the arg-max). Tuning
therefore fixes ``emission_weight = 1`` and searches over ``beta``.

Decoding
--------
Viterbi in log space with back-pointers, vectorised over the K x K
transition matrix of each step:

    delta_i(b) = max_a [delta_{i-1}(a) + log p(a -> b)] + emission_i(b)

Outliers and breaks: if no transition into point i is finite, point i is most
likely a GPS outlier (e.g. a multipath jump of 100 m in one second). It is
skipped: the chain tries to connect the last accepted point directly to the
following points, with the time difference and limits growing accordingly.
Skipped points are later filled from their neighbours. Only when more than
``max_skip`` consecutive points are infeasible is the chain broken: decoding
restarts at the first of them with ``delta = emission``, and ``n_breaks`` is
incremented (e.g. a jump between disconnected parts of the network).
Back-tracking then picks the best final state of the previous segment
independently.

Points without candidates are skipped by the chain and receive the arc of the
nearest decoded neighbour (previous, else next), so every usable GPS point gets
an arc. Finally, the decoded arcs are collapsed into a route; consecutive arcs
that do not touch are joined with the shortest network path, so the returned
route is connected wherever the network allows it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields

import numpy as np
import pandas as pd

from .network import RoadNetwork
from .routing import Router

INF = math.inf
_LIMIT_QUANTUM_M = 25.0  # search limits are rounded up so cached Dijkstra trees are reused


@dataclass
class HMMParams:
    emission_weight: float = 1.0
    beta: float = 5.0
    max_states: int = 8
    back_tolerance_m: float = 15.0
    max_speed_mps: float = 70.0
    search_slack_m: float = 150.0
    gap_s: float = 10.0
    min_prob: float = 1e-6
    max_skip: int = 3  # consecutive infeasible (outlier) points bridged before breaking the chain

    @classmethod
    def from_dict(cls, d: dict | None) -> "HMMParams":
        names = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (d or {}).items() if k in names}
        for name in ("max_states", "max_skip"):
            if name in kw:
                kw[name] = int(kw[name])
        return cls(**kw)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HMMResult:
    point_arc: np.ndarray  # (n_points,) int64 decoded arc per usable GPS point
    route_arcs: np.ndarray  # int64 connected arc sequence
    n_breaks: int


def _step_limits(d_gps: float, dt: float, p: HMMParams) -> tuple[float, float, float]:
    """(search limit, beta_eff, max route distance) for one transition."""
    g = max(1.0, dt / p.gap_s)
    max_route = p.max_speed_mps * max(dt, 1.0)
    limit = min(1.5 * d_gps + p.search_slack_m * g, max_route)
    limit = math.ceil(limit / _LIMIT_QUANTUM_M) * _LIMIT_QUANTUM_M
    return limit, p.beta * g, max_route


def _select_states(cands: pd.DataFrame, max_states: int):
    """Top ``max_states`` candidates per point by probability (ties -> closer arc)."""
    pt = cands["pt"].to_numpy(np.int64)
    prob = cands["prob"].to_numpy(np.float64)
    rank = cands["rank"].to_numpy(np.float64) if "rank" in cands else np.zeros(len(pt))
    frac_col = "frac_exact" if "frac_exact" in cands else "frac"
    order = np.lexsort((rank, -prob, pt))
    pt_s = pt[order]
    new = np.concatenate([[True], pt_s[1:] != pt_s[:-1]])
    starts = np.flatnonzero(new)
    within = np.arange(len(order)) - starts[np.cumsum(new) - 1]
    keep = order[within < max_states]
    pt_k = pt[keep]
    first = np.concatenate([[True], pt_k[1:] != pt_k[:-1]])
    pts = pt_k[first]
    bounds = np.concatenate([np.flatnonzero(first), [len(keep)]])
    return (pts, bounds, cands["arc"].to_numpy(np.int64)[keep],
            cands[frac_col].to_numpy(np.float64)[keep], prob[keep])


def decode(net: RoadNetwork, router: Router, xy: np.ndarray, t: np.ndarray,
           cands: pd.DataFrame, params: HMMParams | None = None) -> HMMResult:
    """Most likely arc sequence for one trip (see module docstring)."""
    p = params or HMMParams()
    n = len(xy)
    if cands is None or len(cands) == 0 or n == 0:
        return HMMResult(np.full(n, -1, np.int64), np.empty(0, np.int64), 0)

    pts, bounds, arcs, fracs, probs = _select_states(cands, p.max_states)
    emit_all = p.emission_weight * np.log(np.maximum(probs, p.min_prob))
    length = np.nan_to_num(net.length, nan=INF)
    src, dst = net.src, net.dst
    m = len(pts)

    def transition_logp(k_from: int, k_to: int, alive: np.ndarray) -> np.ndarray:
        """log p(a -> b) between the states of chain steps k_from and k_to."""
        s0, s1 = bounds[k_from], bounds[k_from + 1]
        s2, s3 = bounds[k_to], bounds[k_to + 1]
        A, FA = arcs[s0:s1], fracs[s0:s1]
        B, FB = arcs[s2:s3], fracs[s2:s3]
        i, j = pts[k_from], pts[k_to]
        d_gps = math.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1])
        limit, beta_eff, max_route = _step_limits(d_gps, float(t[j] - t[i]), p)
        LA = length[A]
        head = (1.0 - FA) * LA
        tail = FB * length[B]
        src_b = src[B].tolist()
        mid = np.full((len(A), len(B)), INF)
        dst_a = dst[A]
        for r in np.flatnonzero(alive):  # unreachable states cannot continue the path
            dist, _ = router.tree_from(int(dst_a[r]), limit)
            mid[r] = [dist.get(v, INF) for v in src_b]
        mid[mid > limit] = INF  # deterministic regardless of what the cache already holds
        D = head[:, None] + mid + tail[None, :]
        same = A[:, None] == B[None, :]
        if same.any():
            fwd = (FB[None, :] - FA[:, None]) * LA[:, None]
            direct = same & ((fwd >= 0) | (-fwd <= p.back_tolerance_m)) & alive[:, None]
            D = np.where(direct, np.abs(fwd), D)
        logp = -np.abs(D - d_gps) / beta_eff
        logp[D > max_route] = -INF
        return logp

    deltas: list[np.ndarray | None] = [None] * m
    backs: list[np.ndarray | None] = [None] * m
    prev_step = np.full(m, -1, dtype=np.int64)  # previous accepted chain step (-1 = chain start)
    deltas[0] = emit_all[bounds[0]:bounds[1]].copy()
    last, skipped, n_breaks = 0, [], 0
    k = 1
    while k < m or skipped:
        if k >= m or len(skipped) > p.max_skip:
            # Too many consecutive infeasible points (or the trip ends on them):
            # treat it as a genuine break and restart the chain at the first one.
            n_breaks += 1
            r = skipped[0]
            deltas[r] = emit_all[bounds[r]:bounds[r + 1]].copy()
            backs[r], prev_step[r] = None, last
            last, skipped = r, []
            k = r + 1
            continue
        delta_prev = deltas[last]
        score = delta_prev[:, None] + transition_logp(last, k, np.isfinite(delta_prev))
        bp = np.argmax(score, axis=0)
        best = score[bp, np.arange(score.shape[1])]
        if np.isfinite(best).any():
            deltas[k] = best + emit_all[bounds[k]:bounds[k + 1]]
            backs[k], prev_step[k] = bp, last
            last, skipped = k, []
            k += 1
            continue
        skipped.append(k)  # point k looks like an outlier: try to bridge over it
        k += 1

    point_arc = np.full(n, -1, dtype=np.int64)
    k = last
    idx = int(np.argmax(deltas[k]))
    while k >= 0:
        point_arc[pts[k]] = arcs[bounds[k] + idx]
        pk = int(prev_step[k])
        if pk >= 0:
            idx = int(np.argmax(deltas[pk])) if backs[k] is None else int(backs[k][idx])
        k = pk
    point_arc = _fill_from_neighbours(point_arc)
    return HMMResult(point_arc, _build_route(net, router, xy, t, point_arc, p), n_breaks)


def _fill_from_neighbours(point_arc: np.ndarray) -> np.ndarray:
    """Points without a decoded arc take the previous decoded arc, else the next one."""
    has = point_arc >= 0
    if has.all() or not has.any():
        return point_arc
    n = len(point_arc)
    ar = np.arange(n)
    prev = np.maximum.accumulate(np.where(has, ar, -1))
    nxt = np.minimum.accumulate(np.where(has, ar, n)[::-1])[::-1]
    src_idx = np.where(prev >= 0, prev, nxt)
    return point_arc[src_idx]


def _build_route(net: RoadNetwork, router: Router, xy: np.ndarray, t: np.ndarray,
                 point_arc: np.ndarray, p: HMMParams) -> np.ndarray:
    """Collapse per-point arcs and join non-touching consecutive arcs with shortest paths."""
    valid = np.flatnonzero(point_arc >= 0)
    if len(valid) == 0:
        return np.empty(0, dtype=np.int64)
    seq = point_arc[valid]
    change = np.flatnonzero(seq[1:] != seq[:-1]) + 1
    route = [int(seq[0])]
    for c in change:
        a, b = int(seq[c - 1]), int(seq[c])
        if net.dst[a] != net.src[b]:
            i, j = valid[c - 1], valid[c]
            d_gps = math.hypot(xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1])
            limit, _, _ = _step_limits(d_gps, float(t[j] - t[i]), p)
            path = router.path_arcs(int(net.dst[a]), int(net.src[b]), limit)
            if path:
                route.extend(path)
        route.append(b)
    out = np.asarray(route, dtype=np.int64)
    return out[np.concatenate([[True], out[1:] != out[:-1]])]
