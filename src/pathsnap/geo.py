"""Per-trip metric projection and angle helpers.

Trips are spread over five continents (including the southern hemisphere and
one on the equator), so no single projected CRS fits all of them. Each trip is
projected into the UTM zone that contains the median of its GPS fixes. UTM is
conformal with < 0.1 % scale error inside a zone, so Euclidean distances and
angles in metres are reliable at the city scale of a trip. GPS headings and arc
bearings are both measured in the same projected frame, so the small angle
between grid north and true north cancels out when they are compared.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_008.8


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the WGS84 / UTM zone containing (lon, lat)."""
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


class Projector:
    """Forward/inverse transform between lon/lat (EPSG:4326) and one UTM zone."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._inv = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    @classmethod
    def for_points(cls, lonlat: np.ndarray) -> "Projector":
        return cls(float(np.median(lonlat[:, 0])), float(np.median(lonlat[:, 1])))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


# Two levels of trust in the GPS heading (direction of the net displacement
# over a +-k fix window):
# * HEADING_MIN_DISP_M: below this net displacement the heading is undefined
#   (the vehicle is not moving). Used for the classifier features, which also
#   receive the path straightness so the model learns how far to trust it.
# * heading_reliable(): the strict test used by the label alignment, where a
#   wrong label is worse than an unresolved tie. A stationary receiver can
#   jitter by ~7-8 m per fix (EDA: off-network stops in trips 00000054 and
#   00000033), which looks like motion but wanders in random directions.
#   Straightness = net displacement / travelled path: ~1 when driving
#   steadily, ~0.3 in such jitter clouds, ~0 at the apex of a U-turn.
#   As a feature gate it lowered train-CV Top-1 (results/experiments/
#   heading_gate_ablation.md): it also discards the useful heading of slow,
#   noisy driving. So it is used for labels only.
HEADING_MIN_DISP_M = 1.0
HEADING_MIN_CHORD_M = 3.0
HEADING_MIN_STRAIGHTNESS = 0.6


def window_motion(xy: np.ndarray, k: int):
    """Motion over the centred window [i-k, i+k] of every point.

    Returns ``(chord, chord_len, path_len)``: the net displacement vector, its
    length, and the summed step length between the window ends.
    """
    n = len(xy)
    idx = np.arange(n)
    lo, hi = np.clip(idx - k, 0, n - 1), np.clip(idx + k, 0, n - 1)
    chord = xy[hi] - xy[lo]
    steps = np.concatenate([[0.0], np.hypot(*np.diff(xy, axis=0).T)]) if n > 1 else np.zeros(n)
    cum = np.cumsum(steps)
    return chord, np.hypot(chord[:, 0], chord[:, 1]), cum[hi] - cum[lo]


def straightness(chord_len: np.ndarray, path_len: np.ndarray) -> np.ndarray:
    return np.where(path_len > 0, chord_len / np.where(path_len > 0, path_len, 1.0), 0.0)


def heading_reliable(chord_len: np.ndarray, path_len: np.ndarray) -> np.ndarray:
    return (chord_len >= HEADING_MIN_CHORD_M) & (straightness(chord_len, path_len) >= HEADING_MIN_STRAIGHTNESS)


def compass_bearing(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Direction of (dx, dy) in degrees clockwise from grid north, in [0, 360)."""
    return np.mod(np.degrees(np.arctan2(dx, dy)), 360.0)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest absolute difference between two bearings, in [0, 180] degrees."""
    d = np.abs(np.mod(np.asarray(a) - np.asarray(b), 360.0))
    return np.minimum(d, 360.0 - d)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Great-circle distance in metres (used for sanity checks and EDA)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
