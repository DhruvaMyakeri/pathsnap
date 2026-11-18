"""Reading the raw PathSnap trip files.

Each trip folder ``<id>/`` holds four whitespace-separated, header-less files:

* ``<id>.track``  lon lat t   GPS observations (t in seconds)
* ``<id>.nodes``  lon lat     road-network nodes (row index = node id)
* ``<id>.arcs``   from to     directed road segments (row index = arc id)
* ``<id>.route``  arc_id      reference route (ordered arc ids)

``load_trip`` deliberately never reads ``.route``: the prediction pipeline only
ever sees GPS + road network. Reference routes are read separately with
``load_route`` by training / evaluation code (server-side only in the API).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray  # (N, 3) float64: lon, lat, t
    nodes: np.ndarray  # (M, 2) float64: lon, lat
    arcs: np.ndarray  # (A, 2) int64: from_node, to_node


def trip_file(trip_dir: Path | str, ext: str) -> Path:
    trip_dir = Path(trip_dir)
    return trip_dir / f"{trip_dir.name}.{ext}"


def _read_table(path: Path, dtype) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, dtype=dtype, engine="c")
    return df.to_numpy()


def list_trips(partition_dir: Path | str) -> list[Path]:
    """Trip folders (sorted by id) that contain a ``.track`` file."""
    d = Path(partition_dir)
    return sorted(p for p in d.iterdir() if p.is_dir() and trip_file(p, "track").exists())


def load_trip(trip_dir: Path | str) -> Trip:
    """GPS track + road network of one trip. Never touches the reference route."""
    trip_dir = Path(trip_dir)
    track = _read_table(trip_file(trip_dir, "track"), np.float64).reshape(-1, 3)
    nodes = _read_table(trip_file(trip_dir, "nodes"), np.float64).reshape(-1, 2)
    arcs = _read_table(trip_file(trip_dir, "arcs"), np.int64).reshape(-1, 2)
    return Trip(trip_dir.name, track, nodes, arcs)


def load_route(trip_dir: Path | str) -> np.ndarray:
    """Reference route (ordered arc ids). Only training/evaluation code may call this."""
    return _read_table(trip_file(trip_dir, "route"), np.int64).ravel()


def clean_track(track: np.ndarray) -> np.ndarray:
    """Indices of usable GPS rows.

    A row is kept when lon/lat/t are finite, the coordinates are plausible
    (inside [-180, 180] x [-90, 90] and not the (0, 0) null fix) and its
    timestamp is strictly later than every kept row before it (drops
    duplicated or out-of-order fixes).
    """
    lon, lat, t = track[:, 0], track[:, 1], track[:, 2]
    ok = np.isfinite(track).all(axis=1)
    ok &= (np.abs(lon) <= 180) & (np.abs(lat) <= 90)
    ok &= ~((lon == 0) & (lat == 0))
    keep = []
    last_t = -np.inf
    for i in np.flatnonzero(ok):
        if t[i] > last_t:
            keep.append(i)
            last_t = t[i]
    return np.asarray(keep, dtype=np.int64)
