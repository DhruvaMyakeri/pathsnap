"""Unit tests for the core library on tiny synthetic data."""
import math

import numpy as np
import pandas as pd

from pathsnap.candidates import generate_candidates
from pathsnap.data import clean_track
from pathsnap.features import FEATURES, build_features
from pathsnap.geo import angle_diff, compass_bearing, heading_reliable, utm_epsg, window_motion
from pathsnap.labels import point_labels
from pathsnap.metrics import collapse, route_continuity, route_overlap
from pathsnap.network import RoadNetwork
from pathsnap.routing import Router


def two_way_line():
    """Nodes on the x-axis every 100 m; arcs i->i+1 (east) and i+1->i (west) for each segment."""
    nodes = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
    arcs = []
    for i in range(3):
        arcs += [[i, i + 1], [i + 1, i]]
    return RoadNetwork(nodes, np.array(arcs))  # even ids eastbound, odd ids westbound


def test_utm_zone_and_hemisphere():
    assert utm_epsg(8.6, 49.8) == 32632  # Germany, north
    assert utm_epsg(-58.4, -34.6) == 32721  # Buenos Aires, south
    assert utm_epsg(-78.5, -0.2) == 32717  # just south of the equator


def test_bearings_and_angle_diff():
    assert compass_bearing(np.array([1.0]), np.array([0.0]))[0] == 90.0  # east
    assert compass_bearing(np.array([0.0]), np.array([-1.0]))[0] == 180.0  # south
    assert angle_diff(np.array([350.0]), np.array([10.0]))[0] == 20.0


def test_heading_reliability_rejects_jitter_and_u_turn_apex():
    rng = np.random.default_rng(0)
    driving = np.column_stack([np.arange(0, 100, 10.0), np.zeros(10)])
    jitter = rng.normal(scale=5.0, size=(10, 2))  # stationary receiver, ~7 m steps
    u_turn = np.array([[0, 0], [10, 0], [20, 0], [10, 1], [0, 1]], dtype=float)
    for xy, expected in [(driving, True), (jitter, False)]:
        _, chord, path = window_motion(xy, 2)
        assert heading_reliable(chord, path)[2:-2].all() == expected
    _, chord, path = window_motion(u_turn, 2)
    assert not heading_reliable(chord, path)[2]


def test_clean_track_drops_invalid_and_non_increasing_rows():
    track = np.array([[8.0, 49.0, 1], [8.0, 49.0, 2], [np.nan, 49.0, 3], [0.0, 0.0, 4],
                      [8.0, 49.0, 4], [8.0, 49.0, 3.5], [8.0, 49.0, 5]], dtype=float)
    assert clean_track(track).tolist() == [0, 1, 4, 6]


def test_network_twins_geometry_and_projection():
    net = two_way_line()
    assert net.twin[0] == 1 and net.twin[1] == 0
    assert net.length[0] == 100.0
    assert net.bearing[0] == 90.0 and net.bearing[1] == 270.0
    dist, frac, frac_raw = net.project(np.array([50.0, -20.0]), np.array([5.0, 0.0]), np.array([0, 0]))
    assert np.allclose(dist, [5.0, 20.0]) and np.allclose(frac, [0.5, 0.0]) and frac_raw[1] < 0


def test_router_distances():
    net = two_way_line()
    router = Router(net)
    assert router.node_distance(0, 3, 1000) == 300.0
    assert router.node_distance(0, 3, 250) == math.inf  # beyond the search limit
    assert router.path_arcs(0, 2, 1000) == [0, 2]
    # forward along the same arc, small backward jitter, and arc-to-next-arc
    assert math.isclose(router.route_distance(0, 0.2, 0, 0.6, 1000), 40.0)
    assert math.isclose(router.route_distance(0, 0.6, 0, 0.5, 1000), 10.0)
    assert math.isclose(router.route_distance(0, 0.5, 2, 0.5, 1000), 100.0)


def test_candidates_expand_radius_for_isolated_points():
    net = two_way_line()
    xy = np.array([[50.0, 3.0], [150.0, 120.0]])  # second point is 120 m from the road
    c = generate_candidates(net, xy, radius=50, max_radius=200, max_k=4)
    assert set(c.loc[c["pt"] == 0, "arc"]) == {0, 1}  # both directions of the nearest segment
    assert (c["pt"] == 1).any()  # found after expanding the radius


def test_alignment_separates_out_and_back_on_a_two_way_road():
    """Monotonicity assigns the outbound fixes to the eastbound arcs and the return to the westbound twins."""
    net = two_way_line()
    out = np.column_stack([np.arange(5, 300, 10.0), np.full(30, 2.0)])
    back = out[::-1] + np.array([0.0, -4.0])
    xy = np.vstack([out, back])
    route = np.array([0, 2, 4, 5, 3, 1])
    lab = point_labels(net, xy, route)
    assert lab["label_valid"].all()
    # At the apex of the U-turn the heading is unreliable and both twins fit equally,
    # so only fixes outside the 2-fix apex window are checked.
    assert set(lab["label_arc"][:28]) == {0, 2, 4}
    assert set(lab["label_arc"][32:]) == {5, 3, 1}
    assert (np.diff(lab["route_pos"]) >= 0).all()


def test_features_heading_separates_twins():
    net = two_way_line()
    xy = np.column_stack([np.arange(10, 100, 10.0), np.full(9, 3.0)])  # driving east
    t = np.arange(9, dtype=float)
    f = build_features(net, xy, t, generate_candidates(net, xy))
    assert list(f.columns[2:2 + len(FEATURES)]) == FEATURES
    mid = f[f["pt"] == 4].set_index("arc")
    assert mid.loc[0, "heading_diff"] < 1 and mid.loc[1, "heading_diff"] > 179


def test_route_metrics():
    net = two_way_line()
    router = Router(net)
    xy = np.column_stack([np.arange(10, 300, 20.0), np.zeros(15)])
    pred = np.array([0] * 5 + [2] * 5 + [4] * 5)
    assert route_continuity(pred, xy, net, router)["continuity"] == 1.0
    jumpy = pred.copy()
    jumpy[7] = 5  # westbound arc of the last segment: not reachable from arc 2 within the limit
    assert route_continuity(jumpy, xy, net, router)["n_jumps"] >= 1
    assert collapse(np.array([-1, 0, 0, 2, 2, -1, 4])).tolist() == [0, 2, 4]
    ov = route_overlap(np.array([0, 2]), np.array([0, 2, 4]), net)
    assert math.isclose(ov["route_precision"], 1.0) and math.isclose(ov["route_recall"], 2 / 3)
