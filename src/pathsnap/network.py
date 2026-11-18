"""Road-network geometry, spatial index and topology for one trip.

Every arc is a straight directed segment ``nodes[from] -> nodes[to]``. About
92 % of arcs have a reverse-direction twin with identical geometry (two-way
roads), so geometry alone cannot tell the two directions apart; the arc
bearing (direction of travel) is stored for the heading features.
"""
from __future__ import annotations

import numpy as np
import shapely

from .geo import compass_bearing


class RoadNetwork:
    def __init__(self, nodes_xy: np.ndarray, arcs: np.ndarray):
        nodes_xy = np.asarray(nodes_xy, dtype=np.float64)
        arcs = np.asarray(arcs, dtype=np.int64).reshape(-1, 2)
        self.nodes_xy = nodes_xy
        self.arcs = arcs
        self.n_nodes = len(nodes_xy)
        self.n_arcs = len(arcs)

        # Arcs that reference a non-existent node can never be used.
        self.valid = ((arcs >= 0) & (arcs < self.n_nodes)).all(axis=1)
        src = np.where(self.valid, arcs[:, 0], 0)
        dst = np.where(self.valid, arcs[:, 1], 0)
        self.src, self.dst = src, dst

        a, b = nodes_xy[src], nodes_xy[dst]
        nan = ~self.valid
        a[nan] = np.nan
        b[nan] = np.nan
        self.ax, self.ay = a[:, 0], a[:, 1]
        self.bx, self.by = b[:, 0], b[:, 1]
        self.dx, self.dy = self.bx - self.ax, self.by - self.ay
        self.length = np.hypot(self.dx, self.dy)
        self.bearing = compass_bearing(self.dx, self.dy)

        # Spatial index over the valid segments.
        self._tree_ids = np.flatnonzero(self.valid)
        coords = np.stack([a[self._tree_ids], b[self._tree_ids]], axis=1)
        self.tree = shapely.STRtree(shapely.linestrings(coords))

        # Reverse-direction twin of each arc (-1 if none).
        key = src * self.n_nodes + dst
        rkey = dst * self.n_nodes + src
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        pos = np.minimum(np.searchsorted(sorted_key, rkey), max(self.n_arcs - 1, 0))
        found = (sorted_key[pos] == rkey) & self.valid & (src != dst)
        self.twin = np.where(found, order[pos], -1)

        # CSR adjacency: outgoing valid arcs of each node.
        va = np.flatnonzero(self.valid)
        self.out_arcs = va[np.argsort(src[va], kind="stable")]
        self.out_degree = np.bincount(src[va], minlength=self.n_nodes)
        self.in_degree = np.bincount(dst[va], minlength=self.n_nodes)
        self.out_ptr = np.concatenate([[0], np.cumsum(self.out_degree)])

    # ------------------------------------------------------------------
    def query(self, points_xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """All (point index, arc id) pairs whose segment is within ``radius`` metres."""
        if len(points_xy) == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pairs = self.tree.query(shapely.points(points_xy), predicate="dwithin", distance=radius)
        return pairs[0].astype(np.int64), self._tree_ids[pairs[1]]

    def project(self, px: np.ndarray, py: np.ndarray, arc_ids: np.ndarray):
        """Orthogonal projection of points onto arcs (vectorised, paired element-wise).

        Returns ``(dist, frac, frac_raw)``: distance in metres to the closest
        point of the segment, its position along the arc clamped to [0, 1], and
        the unclamped position (outside [0, 1] when the foot of the
        perpendicular falls beyond an end of the segment).
        """
        ax, ay = self.ax[arc_ids], self.ay[arc_ids]
        dx, dy = self.dx[arc_ids], self.dy[arc_ids]
        l2 = dx * dx + dy * dy
        safe = np.where(l2 > 0, l2, 1.0)
        frac_raw = np.where(l2 > 0, ((px - ax) * dx + (py - ay) * dy) / safe, 0.0)
        frac = np.clip(frac_raw, 0.0, 1.0)
        dist = np.hypot(px - (ax + frac * dx), py - (ay + frac * dy))
        return dist, frac, frac_raw

    def arcs_lonlat(self, arc_ids, nodes_lonlat: np.ndarray) -> list[list[list[float]]]:
        """[[lat, lon], [lat, lon]] pairs per arc, for map display."""
        out = []
        for a in np.asarray(arc_ids, dtype=np.int64):
            s, d = nodes_lonlat[self.src[a]], nodes_lonlat[self.dst[a]]
            out.append([[float(s[1]), float(s[0])], [float(d[1]), float(d[0])]])
        return out
