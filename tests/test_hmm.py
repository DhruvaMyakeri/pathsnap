"""Unit tests for the HMM / Viterbi decoder on tiny synthetic road networks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.hmm import HMMParams, decode
from pathsnap.network import RoadNetwork
from pathsnap.routing import Router


def build_net(polylines):
    """Two-way roads along the given polylines. Returns (net, forward arc ids per polyline)."""
    nodes, arcs, forward = [], [], []
    for line in polylines:
        base, fwd = len(nodes), []
        nodes.extend(line)
        for k in range(len(line) - 1):
            fwd.append(len(arcs))
            arcs.append((base + k, base + k + 1))
            arcs.append((base + k + 1, base + k))
        forward.append(fwd)
    return RoadNetwork(np.array(nodes, dtype=float), np.array(arcs)), forward


def candidates(net, xy, radius=30.0):
    c = generate_candidates(net, xy, radius=radius, max_radius=radius, max_k=16)
    return c.rename(columns={"frac": "frac_exact"})


def assign_prob(c, expected, net, base=0.6, twin=0.3, other=0.05, overrides=None):
    """Probability per candidate: expected arc > its twin > anything else, plus noisy overrides."""
    exp = np.asarray(expected)[c["pt"].to_numpy()]
    arc = c["arc"].to_numpy()
    prob = np.where(arc == exp, base, np.where(arc == net.twin[exp], twin, other))
    for (pt, a), pr in (overrides or {}).items():
        prob[(c["pt"].to_numpy() == pt) & (arc == a)] = pr
    c = c.copy()
    c["prob"] = prob
    return c


def is_connected_sequence(net, route):
    return all(net.dst[a] == net.src[b] for a, b in zip(route[:-1], route[1:]))


def straight_with_parallel():
    road1 = [(10.0 * k, 0.0) for k in range(11)]
    road2 = [(10.0 * k, 20.0) for k in range(11)]
    net, (fwd1, fwd2) = build_net([road1, road2])
    xs = 5.0 + 10.0 * np.arange(10)
    xy = np.column_stack([xs, np.full(10, 3.0)])
    t = np.arange(10, dtype=float)
    expected = np.array([fwd1[int(x // 10)] for x in xs])
    return net, fwd1, fwd2, xy, t, expected


def argmax_arc(c, n):
    best = c.sort_values(["pt", "prob"], ascending=[True, False]).drop_duplicates("pt")
    out = np.full(n, -1)
    out[best["pt"].to_numpy()] = best["arc"].to_numpy()
    return out


def test_hmm_fixes_noisy_argmax_on_two_way_road_with_parallel_street():
    net, fwd1, fwd2, xy, t, expected = straight_with_parallel()
    c = candidates(net, xy)
    noisy = {(3, int(net.twin[expected[3]])): 0.7, (3, int(expected[3])): 0.2,  # wrong direction
             (6, fwd2[6]): 0.9, (6, int(expected[6])): 0.2,  # parallel street
             (8, int(net.twin[expected[8]])): 0.8}  # wrong direction again
    c = assign_prob(c, expected, net, overrides=noisy)
    clf = argmax_arc(c, len(xy))
    assert (clf != expected).sum() == 3  # the classifier alone jumps three times

    res = decode(net, Router(net), xy, t, c, HMMParams())
    np.testing.assert_array_equal(res.point_arc, expected)
    assert res.n_breaks == 0
    np.testing.assert_array_equal(res.route_arcs, fwd1)
    assert is_connected_sequence(net, res.route_arcs)


def test_time_gap_with_detour_stays_feasible_and_route_is_filled():
    corner = [(10.0 * k, 0.0) for k in range(11)] + [(100.0, 10.0 * k) for k in range(1, 11)]
    net, (fwd,) = build_net([corner])
    xy = np.array([[5, 3], [15, 3], [25, 3], [97, 75], [97, 85]], dtype=float)
    t = np.array([0, 1, 2, 22, 23], dtype=float)  # 20 s GPS dropout around the corner
    expected = np.array([fwd[0], fwd[1], fwd[2], fwd[17], fwd[18]])
    c = assign_prob(candidates(net, xy), expected, net, twin=0.35)
    res = decode(net, Router(net), xy, t, c, HMMParams())
    np.testing.assert_array_equal(res.point_arc, expected)
    assert res.n_breaks == 0
    np.testing.assert_array_equal(res.route_arcs, fwd[:19])  # gap filled with the driven arcs
    assert is_connected_sequence(net, res.route_arcs)


def test_points_without_candidates_take_neighbour_arc():
    net, fwd1, _, xy, t, expected = straight_with_parallel()
    c = assign_prob(candidates(net, xy), expected, net)
    c = c[~c["pt"].isin([0, 4])].reset_index(drop=True)
    res = decode(net, Router(net), xy, t, c, HMMParams())
    assert res.point_arc[4] == res.point_arc[3] == expected[3]  # previous neighbour
    assert res.point_arc[0] == res.point_arc[1] == expected[1]  # no previous -> next neighbour
    assert (res.point_arc >= 0).all()


def test_disconnected_components_force_a_break():
    road_a = [(10.0 * k, 0.0) for k in range(6)]
    road_b = [(1000.0 + 10.0 * k, 0.0) for k in range(6)]
    net, (fa, fb) = build_net([road_a, road_b])
    xy = np.array([[5, 2], [15, 2], [25, 2], [1005, 2], [1015, 2]], dtype=float)
    t = np.arange(5, dtype=float)
    expected = np.array([fa[0], fa[1], fa[2], fb[0], fb[1]])
    c = assign_prob(candidates(net, xy), expected, net)
    res = decode(net, Router(net), xy, t, c, HMMParams())
    assert res.n_breaks == 1
    np.testing.assert_array_equal(res.point_arc, expected)
    assert set(fa[:3]) <= set(res.route_arcs.tolist()) and set(fb[:2]) <= set(res.route_arcs.tolist())


def test_single_gps_outlier_is_bridged_not_a_break():
    road = [(10.0 * k, 0.0) for k in range(30)]
    net, (fwd,) = build_net([road])
    xs = 5.0 + 10.0 * np.arange(10)
    xs[5] = 205.0  # multipath jump: 160 m ahead in one second, then back
    xy = np.column_stack([xs, np.full(10, 2.0)])
    t = np.arange(10, dtype=float)
    expected = np.array([fwd[int(x // 10)] for x in xs])
    c = assign_prob(candidates(net, xy), expected, net)
    res = decode(net, Router(net), xy, t, c, HMMParams())
    assert res.n_breaks == 0
    assert res.point_arc[5] == res.point_arc[4] == fwd[4]  # outlier takes its neighbour's arc
    np.testing.assert_array_equal(np.delete(res.point_arc, 5), np.delete(expected, 5))
    np.testing.assert_array_equal(res.route_arcs, fwd[:10])


def test_empty_candidates_and_params_roundtrip():
    net, *_ = straight_with_parallel()
    xy = np.zeros((3, 2))
    res = decode(net, Router(net), xy, np.arange(3.0), pd.DataFrame(columns=["pt", "arc", "prob"]))
    assert (res.point_arc == -1).all() and len(res.route_arcs) == 0
    p = HMMParams.from_dict({"beta": 12.5, "max_states": 6.0, "model": "lightgbm", "objective_value": 0.9})
    assert p.beta == 12.5 and p.max_states == 6 and HMMParams.from_dict(p.to_dict()) == p
