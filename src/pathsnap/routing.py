"""Bounded shortest paths on the directed road network.

GPS is sampled at ~1 Hz, so consecutive observations are usually a few tens of
metres apart. Dijkstra searches are therefore bounded by a distance limit and
cached per source node (consecutive points share most candidate arcs).
"""
from __future__ import annotations

import heapq
import math
from collections import OrderedDict

import numpy as np

from .network import RoadNetwork

INF = math.inf


class Router:
    def __init__(self, net: RoadNetwork, cache_size: int = 50_000):
        self.net = net
        self._ptr = net.out_ptr.tolist()
        self._out = net.out_arcs.tolist()
        self._src = net.src.tolist()
        self._dst = net.dst.tolist()
        self._len = np.nan_to_num(net.length, nan=INF).tolist()
        self._cache: OrderedDict[int, tuple[float, dict, dict]] = OrderedDict()
        self._cache_size = cache_size

    def _search(self, source: int, limit: float):
        ptr, out, dst, length = self._ptr, self._out, self._dst, self._len
        dist = {source: 0.0}
        pred = {source: -1}  # arc used to reach each node
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for k in range(ptr[u], ptr[u + 1]):
                a = out[k]
                nd = d + length[a]
                if nd <= limit:
                    v = dst[a]
                    if nd < dist.get(v, INF):
                        dist[v] = nd
                        pred[v] = a
                        heapq.heappush(heap, (nd, v))
        return dist, pred

    def tree_from(self, source: int, limit: float) -> tuple[dict, dict]:
        """Shortest-path tree from ``source`` covering every node within ``limit`` metres."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit:
            self._cache.move_to_end(source)
            return hit[1], hit[2]
        dist, pred = self._search(source, limit)
        self._cache[source] = (limit, dist, pred)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return dist, pred

    def node_distance(self, u: int, v: int, limit: float) -> float:
        """Network distance from node ``u`` to node ``v`` (``inf`` if farther than ``limit``)."""
        if u == v:
            return 0.0
        dist, _ = self.tree_from(u, limit)
        d = dist.get(v, INF)
        return d if d <= limit else INF

    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arc ids of the shortest path from node ``u`` to node ``v`` (None if not within ``limit``)."""
        if u == v:
            return []
        dist, pred = self.tree_from(u, limit)
        if dist.get(v, INF) > limit:
            return None
        path, node = [], v
        while node != u:
            a = pred[node]
            path.append(a)
            node = self._src[a]
        path.reverse()
        return path

    def route_distance(self, a: int, frac_a: float, b: int, frac_b: float, limit: float,
                       back_tolerance_m: float = 15.0) -> float:
        """Driving distance from position ``frac_a`` on arc ``a`` to ``frac_b`` on arc ``b``.

        Moving slightly backwards along the same arc (up to ``back_tolerance_m``)
        is treated as GPS jitter rather than as a loop around the block.
        """
        la, lb = self._len[a], self._len[b]
        if a == b:
            delta = (frac_b - frac_a) * la
            if delta >= 0:
                return delta
            if -delta <= back_tolerance_m:
                return -delta
        head = (1.0 - frac_a) * la
        tail = frac_b * lb
        if head + tail > limit:
            return INF
        mid = self.node_distance(self._dst[a], self._src[b], limit - head - tail)
        return head + mid + tail
