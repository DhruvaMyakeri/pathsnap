"""API contract tests: endpoints work and ground truth never leaves the server.

Run from the repository root:  python -m pytest tests/test_api.py -q
Uses the default eval dir (the student_test partition) unless PATHSNAP_EVAL_DIR is set.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `api` importable

from api.main import METRIC_KEYS, app  # noqa: E402

FORBIDDEN_KEY_PARTS = ("label", "reference", "ref_route", "ground_truth", "truth", "correct", "route_pos")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # context manager runs the lifespan (model/HMM loading)
        yield c


def _all_keys(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield k
            yield from _all_keys(v)
    elif isinstance(o, list):
        for v in o:
            yield from _all_keys(v)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    h = r.json()
    for key in ("status", "model_name", "hmm_enabled", "eval_dir", "n_trips"):
        assert key in h
    assert h["n_trips"] > 0


def test_trips_lists_student_test(client):
    r = client.get("/trips")
    assert r.status_code == 200
    trips = r.json()
    assert len(trips) == 10
    assert all(set(t) == {"trip_id", "n_points"} and t["n_points"] > 0 for t in trips)


def test_predict_smallest_trip_has_metrics_and_no_ground_truth(client):
    trips = client.get("/trips").json()
    smallest = min(trips, key=lambda t: t["n_points"])["trip_id"]
    r = client.post("/predict", json={"trip_id": smallest})
    assert r.status_code == 200
    res = r.json()
    assert res["trip_id"] == smallest
    assert res["n_points"] > 0 and len(res["gps"]) > 0 and len(res["route"]) > 0
    assert set(res["metrics"]) == set(METRIC_KEYS)
    acc = res["metrics"]["accuracy"]
    assert acc is not None and 0.0 <= acc <= 1.0
    # aggregate metric names (e.g. n_labelled_points) are allowed; anything else label-like is not
    bad = [k for k in _all_keys(res)
           if k not in METRIC_KEYS and any(p in k.lower() for p in FORBIDDEN_KEY_PARTS)]
    assert not bad, f"ground-truth-like keys leaked: {bad}"
    # predicted route is coordinates only: no arc ids that could be compared with .route
    assert all(len(pt) == 2 for line in res["route"] for pt in line)


@pytest.mark.parametrize("trip_id", ["99999999", "../train/00000001", "..\\train\\00000001", "", "00000076/../00000000"])
def test_predict_rejects_unknown_or_traversal(client, trip_id):
    r = client.post("/predict", json={"trip_id": trip_id})
    assert r.status_code == 404
