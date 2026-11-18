"""PathSnap dataset exploration (EDA).

Recomputes every EDA table, figure and map from the raw trip files.

Partitions analysed: ``train`` (80 trips) and ``student_test`` (10 trips).
The evaluator partition is not on disk and is never touched.

Outputs
  results/eda/*.csv              inventory, per-trip statistics, partition summaries, issue lists
  results/eda/fig*.png           figures used in the report
  results/eda/trip_<id>_projected.png   static maps (projected metres) of the representative trips
  results/eda/key_numbers.json   headline numbers (read by EDA.ipynb)
  results/eda/data_quality.md    data-quality log: issues, assumptions, decisions
  results/maps/eda_world_overview.html, results/maps/eda_trip_<id>.html   interactive Folium maps

Usage (from the repo root):  python eda/run_eda.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import folium
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from pathsnap import config  # noqa: E402
from pathsnap.data import clean_track, list_trips, load_route, load_trip, trip_file  # noqa: E402
from pathsnap.geo import Projector  # noqa: E402
from pathsnap.labels import point_labels  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.routing import INF, Router  # noqa: E402

OUT = config.RESULTS_DIR / "eda"
MAPS = config.RESULTS_DIR / "maps"
N_JOBS = 6
PARTS = ["train", "student_test"]
PART_COLORS = {"train": "#1f77b4", "student_test": "#ff7f0e"}
# Column names as documented in "Dataset Exploration" §3 (files have no header).
FILE_COLUMNS = {
    "track": ["lon", "lat", "t"],
    "nodes": ["lon", "lat"],
    "arcs": ["from_node", "to_node"],
    "route": ["arc_id"],
}
SPEED_LIMIT_MPS = 70.0
FAR_M = config.LABEL_MAX_DIST_M  # 50 m
REPRESENTATIVE = {
    "00000001": ("train", "clean urban trip"),
    "00000019": ("train", "reference route with broken links"),
    "00000054": ("train", "long stationary periods off the reference route"),
}
DPI = 130


# ----------------------------------------------------------------------------- per-trip analysis
def analyse_trip(trip_dir: Path, partition: str):
    """All per-trip statistics. Returns (row, dtype_rows, broken_link_rows, arrays)."""
    tid = trip_dir.name
    row: dict = {"partition": partition, "trip_id": tid}
    dtype_rows = []

    # --- files, raw dtypes and missing values (files read without forcing a dtype)
    for ext, cols in FILE_COLUMNS.items():
        f = trip_file(trip_dir, ext)
        row[f"has_{ext}"] = f.exists()
        row[f"{ext}_bytes"] = f.stat().st_size if f.exists() else 0
        raw = pd.read_csv(f, sep=r"\s+", header=None)
        row[f"{ext}_rows"] = len(raw)
        row[f"{ext}_cols"] = raw.shape[1]
        row[f"{ext}_missing_values"] = int(raw.isna().sum().sum())
        for j, c in enumerate(raw.columns):
            dtype_rows.append({"partition": partition, "trip_id": tid, "file": f".{ext}",
                               "column": cols[j] if j < len(cols) else f"extra_{j}",
                               "dtype": str(raw[c].dtype)})

    trip = load_trip(trip_dir)
    route = load_route(trip_dir)

    # --- GPS validity (raw rows)
    tr = trip.track
    lon, lat, t_raw = tr[:, 0], tr[:, 1], tr[:, 2]
    finite = np.isfinite(tr).all(axis=1)
    dt_raw = np.diff(t_raw)
    row["gps_nonfinite_rows"] = int((~finite).sum())
    row["gps_out_of_range_rows"] = int((finite & ((np.abs(lon) > 180) | (np.abs(lat) > 90))).sum())
    row["gps_null_island_rows"] = int(((lon == 0) & (lat == 0)).sum())
    row["t_duplicate_steps"] = int((dt_raw == 0).sum())
    row["t_backward_steps"] = int((dt_raw < 0).sum())
    row["t_non_integer"] = int((np.abs(t_raw - np.round(t_raw)) > 1e-9).sum())
    row["dup_consecutive_fixes"] = int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum())
    keep = clean_track(tr)
    row["usable_points"] = int(len(keep))
    row["dropped_points"] = int(len(tr) - len(keep))

    # --- sampling, extent, kinematics (usable rows, projected to the trip's UTM zone)
    tc = tr[keep]
    t = tc[:, 2]
    proj = Projector.for_points(tc[:, :2])
    xy = proj.to_xy(tc[:, :2])
    dt = np.diff(t)
    step = np.hypot(*np.diff(xy, axis=0).T)
    speed = step / dt
    row.update({
        "utm_epsg": proj.epsg,
        "hemisphere": "S" if np.median(tc[:, 1]) < 0 else "N",
        "t_first_s": float(t[0]), "t_last_s": float(t[-1]),
        "duration_min": float((t[-1] - t[0]) / 60),
        "share_1s_steps": float(np.isclose(dt, 1.0).mean()),
        "median_dt_s": float(np.median(dt)),
        "max_gap_s": float(dt.max()),
        "gaps_gt_10s": int((dt > 10).sum()),
        "gaps_gt_60s": int((dt > 60).sum()),
        "length_km": float(step.sum() / 1000),
        "median_speed_mps": float(np.median(speed)),
        "p95_speed_mps": float(np.percentile(speed, 95)),
        "max_speed_mps": float(speed.max()),
        "steps_gt_70mps": int((speed > SPEED_LIMIT_MPS).sum()),
        "max_step_m": float(step.max()),
        "lon_min": float(tc[:, 0].min()), "lon_max": float(tc[:, 0].max()),
        "lat_min": float(tc[:, 1].min()), "lat_max": float(tc[:, 1].max()),
        "centroid_lon": float(np.median(tc[:, 0])), "centroid_lat": float(np.median(tc[:, 1])),
        "extent_x_km": float(np.ptp(xy[:, 0]) / 1000), "extent_y_km": float(np.ptp(xy[:, 1]) / 1000),
    })
    row["mean_speed_kmh"] = row["length_km"] / max(row["duration_min"] / 60, 1e-9)

    # --- road network
    net = RoadNetwork(proj.to_xy(trip.nodes), trip.arcs)
    v = net.valid
    n = net.n_nodes
    key = net.src[v] * n + net.dst[v]
    g = csr_matrix((np.ones(int(v.sum())), (net.src[v], net.dst[v])), shape=(n, n))
    n_wcc, wcc = connected_components(g, directed=True, connection="weak")
    n_scc, scc = connected_components(g, directed=True, connection="strong")
    big_scc = int(np.bincount(scc).argmax())
    in_big = scc == big_scc
    lengths = net.length[v]
    row.update({
        "nodes": n, "arcs": net.n_arcs,
        "invalid_arc_refs": int((~v).sum()),
        "zero_length_arcs": int((lengths == 0).sum()),
        "self_loops": int((net.src[v] == net.dst[v]).sum()),
        "duplicate_arcs": int(len(key) - len(np.unique(key))),
        "twin_share": float((net.twin[v] >= 0).mean()),
        "isolated_nodes": int(((net.out_degree + net.in_degree) == 0).sum()),
        "weak_components": int(n_wcc),
        "largest_wcc_share": float(np.bincount(wcc).max() / n),
        "strong_components": int(n_scc),
        "largest_scc_share": float(in_big.mean()),
        "arc_length_median_m": float(np.median(lengths)),
        "arc_length_p95_m": float(np.percentile(lengths, 95)),
        "network_km": float(lengths.sum() / 1000),
        "mean_out_degree": float(net.out_degree.mean()),
    })

    # --- reference route
    ok = (route >= 0) & (route < net.n_arcs)
    r = route[ok]
    r = r[net.valid[r]]
    link_ok = net.dst[r[:-1]] == net.src[r[1:]]
    broken_idx = np.flatnonzero(~link_ok)
    row.update({
        "route_arcs": int(len(route)),
        "route_invalid_ids": int((~ok).sum()),
        "route_km": float(net.length[r].sum() / 1000),
        "route_broken_links": int(len(broken_idx)),
        "route_repeated_arcs": int(len(r) - len(np.unique(r))),
        "route_uturns": int((net.twin[r[:-1]] == r[1:]).sum()),
        "route_arcs_in_largest_scc": float((in_big[net.src[r]] & in_big[net.dst[r]]).mean()),
    })
    broken_rows = []
    if len(broken_idx):
        router = Router(net)
        for i in broken_idx:
            a, b = int(r[i]), int(r[i + 1])
            gap = float(np.hypot(net.bx[a] - net.ax[b], net.by[a] - net.ay[b]))
            nd = router.node_distance(int(net.dst[a]), int(net.src[b]), 5000.0)
            broken_rows.append({"partition": partition, "trip_id": tid, "route_pos": int(i),
                                "arc_from": a, "arc_to": b, "straight_gap_m": gap,
                                "network_bridge_m": None if nd == INF else float(nd),
                                "arc_to_is_twin": bool(net.twin[a] == b)})

    # --- GPS noise relative to the reference route (monotone alignment, no distance cut-off)
    lab = point_labels(net, xy, route, max_dist=np.inf)
    d = lab["label_dist"]
    far = d > FAR_M
    far_steps = far[1:] & far[:-1]
    fidx = np.flatnonzero(far)
    row.update({
        "label_dist_median_m": float(np.median(d)),
        "label_dist_p90_m": float(np.percentile(d, 90)),
        "share_points_gt_50m": float(far.mean()),
        "labelled_points": int((~far).sum()),
        "longest_far_run_points": int(_longest_run(far)),
        # where / how the far fixes move (separates GPS glitches from unrouted driving)
        "far_span_h": float((t[fidx[-1]] - t[fidx[0]]) / 3600) if len(fidx) else 0.0,
        "far_bbox_km": float(max(np.ptp(xy[fidx, 0]), np.ptp(xy[fidx, 1])) / 1000) if len(fidx) else 0.0,
        "far_median_speed_mps": float(np.median(speed[far_steps])) if far_steps.any() else float("nan"),
    })
    # Stationarity: net displacement over 60 fixes vs the summed step length over the same window.
    # A parked vehicle with GPS jitter has a small net displacement but a long summed path.
    w = 60
    if len(xy) > w:
        i = np.arange(len(xy) - w)
        net_disp = np.hypot(*(xy[i + w] - xy[i]).T)
        csum = np.concatenate([[0.0], np.cumsum(step)])
        path = csum[i + w] - csum[i]
        still = net_disp < 30.0
        fw = far[i] & far[i + w]
        row.update({
            "stationary_share_60s": float(still.mean()),
            "stationary_path_m_per_60s": float(np.median(path[still])) if still.any() else float("nan"),
            "far_net_disp_60s_m": float(np.median(net_disp[fw])) if fw.any() else float("nan"),
            "far_path_60s_m": float(np.median(path[fw])) if fw.any() else float("nan"),
        })
    arrays = {
        "partition": partition, "trip_id": tid, "dt": dt, "speed": speed, "label_dist": d,
        "lon": tc[:, 0], "lat": tc[:, 1],
        "offset_km": np.hypot(*(xy - np.median(xy, axis=0)).T) / 1000,
        "arc_len_sample": np.random.default_rng(0).choice(lengths, size=min(3000, len(lengths)), replace=False),
    }
    return row, dtype_rows, broken_rows, arrays


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


# ----------------------------------------------------------------------------- trip geometry
def trip_geometry(trip_dir: Path, buffer_m: float = 300.0) -> dict:
    """Everything needed to draw one trip: GPS, clipped network, reference route, far points."""
    trip = load_trip(trip_dir)
    route = load_route(trip_dir)
    tc = trip.track[clean_track(trip.track)]
    proj = Projector.for_points(tc[:, :2])
    xy = proj.to_xy(tc[:, :2])
    net = RoadNetwork(proj.to_xy(trip.nodes), trip.arcs)
    step = max(1, len(xy) // 6000)
    dist, _ = cKDTree(xy[::step]).query(net.nodes_xy, distance_upper_bound=buffer_m)
    near = np.isfinite(dist)
    sel = net.valid & (near[net.src] | near[net.dst])
    sel &= (net.twin < 0) | (net.src < net.dst)  # draw each two-way road once
    ok = (route >= 0) & (route < net.n_arcs)
    r = route[ok]
    r = r[net.valid[r]]
    broken = np.flatnonzero(net.dst[r[:-1]] != net.src[r[1:]])
    lab = point_labels(net, xy, route, max_dist=np.inf)
    return {"trip_id": trip_dir.name, "lonlat": tc[:, :2], "xy": xy, "net": net,
            "nodes_lonlat": trip.nodes, "net_arcs": np.flatnonzero(sel), "route": r,
            "broken": broken, "far": lab["label_dist"] > FAR_M, "center_xy": np.median(xy, axis=0)}


def _route_chains(net: RoadNetwork, r: np.ndarray) -> list[list[int]]:
    """Split the arc sequence into head-to-tail connected chains of node ids."""
    chains: list[list[int]] = []
    for k, a in enumerate(r):
        s, d = int(net.src[a]), int(net.dst[a])
        if k == 0 or net.dst[r[k - 1]] != s:
            chains.append([s])
        chains[-1].append(d)
    return chains


def _add_legend(m: folium.Map, title: str, items: list[tuple[str, str, str]]):
    rows = "".join(
        f'<div><span style="display:inline-block;width:{14 if shape == "line" else 10}px;'
        f'height:{4 if shape == "line" else 10}px;background:{color};margin:0 6px 2px 0;'
        f'border-radius:{"50%" if shape == "dot" else "0"};vertical-align:middle"></span>{label}</div>'
        for color, label, shape in items)
    html = (f'<div style="position:fixed;bottom:28px;left:28px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #888;border-radius:4px;font:13px sans-serif">'
            f'<b>{title}</b>{rows}</div>')
    m.get_root().html.add_child(folium.Element(html))


def trip_map(trip_dir: Path, note: str = "", max_gps_markers: int = 2500) -> folium.Map:
    """Interactive Folium map: GPS fixes, clipped road network, reference route, issues."""
    geo = trip_geometry(Path(trip_dir))
    ll, net, nodes_ll = geo["lonlat"], geo["net"], geo["nodes_lonlat"]
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[float(ll[:, 1].min()), float(ll[:, 0].min())], [float(ll[:, 1].max()), float(ll[:, 0].max())]])

    a = geo["net_arcs"]
    s_ll, d_ll = nodes_ll[net.src[a]], nodes_ll[net.dst[a]]
    lines = np.round(np.stack([s_ll, d_ll], axis=1), 6).tolist()
    fg_net = folium.FeatureGroup(name=f"Road network within 300 m ({len(a):,} roads)")
    folium.GeoJson({"type": "Feature", "properties": {},
                    "geometry": {"type": "MultiLineString", "coordinates": lines}},
                   style_function=lambda _: {"color": "#9a9a9a", "weight": 1.2, "opacity": 0.8},
                   control=False).add_to(fg_net)
    fg_net.add_to(m)

    fg_route = folium.FeatureGroup(name="Reference route (.route)")
    for chain in _route_chains(net, geo["route"]):
        folium.PolyLine([[float(nodes_ll[c, 1]), float(nodes_ll[c, 0])] for c in chain],
                        color="#1a9850", weight=5, opacity=0.75).add_to(fg_route)
    fg_route.add_to(m)

    step = max(1, len(ll) // max_gps_markers)
    fg_gps = folium.FeatureGroup(name=f"GPS fixes (every {step}. of {len(ll):,})" if step > 1
                                 else f"GPS fixes ({len(ll):,})")
    far = geo["far"]
    for i in range(0, len(ll), step):
        folium.CircleMarker([float(ll[i, 1]), float(ll[i, 0])], radius=2,
                            color="#e66101" if far[i] else "#2b57d6", fill=True, fill_opacity=0.9,
                            weight=0).add_to(fg_gps)
    fg_gps.add_to(m)

    fg_issues = folium.FeatureGroup(name="Route issues and start/end")
    r = geo["route"]
    for i in geo["broken"]:
        p = nodes_ll[net.dst[r[i]]]
        q = nodes_ll[net.src[r[i + 1]]]
        for pt, lbl in ((p, "end of arc"), (q, "start of next arc")):
            folium.CircleMarker([float(pt[1]), float(pt[0])], radius=7, color="#d7191c", weight=3, fill=False,
                                popup=f"Broken route link at position {int(i)}: {lbl}").add_to(fg_issues)
    folium.Marker([float(ll[0, 1]), float(ll[0, 0])], popup="GPS start",
                  icon=folium.Icon(color="green", icon="play")).add_to(fg_issues)
    folium.Marker([float(ll[-1, 1]), float(ll[-1, 0])], popup="GPS end",
                  icon=folium.Icon(color="red", icon="stop")).add_to(fg_issues)
    fg_issues.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, f"Trip {geo['trip_id']} (train){' - ' + note if note else ''}", [
        ("#2b57d6", "GPS fix", "dot"),
        ("#e66101", "GPS fix > 50 m from reference route", "dot"),
        ("#1a9850", "Reference route", "line"),
        ("#9a9a9a", "Road network (within 300 m)", "line"),
        ("#d7191c", "Broken route link", "dot"),
    ])
    return m


def _draw_trip(ax, geo: dict, xlim, ylim):
    net, c = geo["net"], geo["center_xy"]
    a = geo["net_arcs"]
    seg = np.stack([net.nodes_xy[net.src[a]], net.nodes_xy[net.dst[a]]], axis=1) - c
    ax.add_collection(LineCollection(seg, colors="#b0b0b0", linewidths=0.4, label="road network (within 300 m)"))
    r = geo["route"]
    rs = np.stack([net.nodes_xy[net.src[r]], net.nodes_xy[net.dst[r]]], axis=1) - c
    ax.add_collection(LineCollection(rs, colors="#1a9850", linewidths=2.2, alpha=0.8, label="reference route"))
    xy = geo["xy"] - c
    far = geo["far"]
    ax.scatter(xy[~far, 0], xy[~far, 1], s=1.5, c="#2b57d6", label="GPS fix", zorder=3)
    if far.any():
        ax.scatter(xy[far, 0], xy[far, 1], s=1.5, c="#e66101", label="GPS fix > 50 m from route", zorder=3)
    for i in geo["broken"]:
        p = net.nodes_xy[net.dst[r[i]]] - c
        ax.plot(p[0], p[1], "x", color="#d7191c", ms=12, mew=2.5, zorder=4)
    if len(geo["broken"]):
        ax.plot([], [], "x", color="#d7191c", ms=10, mew=2.5, label="broken route link")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, UTM, relative to trip centre)")
    ax.set_ylabel("y (m)")


def trip_static_png(trip_dir: Path, note: str, out: Path):
    """Static map in projected metres; trips with many far fixes get a zoom panel on them."""
    geo = trip_geometry(Path(trip_dir))
    xy = geo["xy"] - geo["center_xy"]
    far = geo["far"]
    pad = 200
    full = ((xy[:, 0].min() - pad, xy[:, 0].max() + pad), (xy[:, 1].min() - pad, xy[:, 1].max() + pad))
    zoom = far.sum() >= 1000
    fig, axes = plt.subplots(1, 2 if zoom else 1, figsize=(15 if zoom else 8, 8), squeeze=False)
    _draw_trip(axes[0, 0], geo, *full)
    axes[0, 0].set_title(f"Trip {geo['trip_id']} (train) - {note}")
    axes[0, 0].legend(loc="best", fontsize=8, markerscale=4)
    if zoom:
        f = xy[far]
        zx = (f[:, 0].min() - 100, f[:, 0].max() + 100)
        zy = (f[:, 1].min() - 100, f[:, 1].max() + 100)
        _draw_trip(axes[0, 1], geo, zx, zy)
        axes[0, 1].set_title(f"Zoom: {int(far.sum()):,} fixes > 50 m from the reference route")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI)
    plt.close(fig)


def world_map(stats: pd.DataFrame) -> folium.Map:
    m = folium.Map(location=[30, 10], zoom_start=2, tiles="OpenStreetMap")
    for part in PARTS:
        fg = folium.FeatureGroup(name=f"{part} ({(stats['partition'] == part).sum()} trips)")
        for _, s in stats[stats["partition"] == part].iterrows():
            folium.CircleMarker(
                [s["centroid_lat"], s["centroid_lon"]], radius=6, color=PART_COLORS[part], fill=True,
                fill_opacity=0.8, weight=1,
                popup=folium.Popup(f"trip {s['trip_id']} ({part})<br>{int(s['usable_points']):,} GPS points"
                                   f"<br>{s['length_km']:.1f} km, {s['duration_min']:.0f} min"
                                   f"<br>UTM EPSG:{int(s['utm_epsg'])}", max_width=220),
            ).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _add_legend(m, "PathSnap trips (trip centroid)", [
        (PART_COLORS["train"], "train", "dot"), (PART_COLORS["student_test"], "student_test", "dot")])
    return m


# ----------------------------------------------------------------------------- figures
def _hist_by_part(ax, arrays, key, bins, **kw):
    for part in PARTS:
        vals = np.concatenate([a[key] for a in arrays if a["partition"] == part])
        ax.hist(vals, bins=bins, alpha=0.6, label=part, color=PART_COLORS[part], **kw)


def make_figures(stats: pd.DataFrame, arrays: list[dict], broken: pd.DataFrame):
    # 1 inventory
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[0].hist(s["usable_points"], bins=np.logspace(2, 4.5, 25), alpha=0.6, color=PART_COLORS[part],
                   label=f"{part} ({len(s)} trips)")
    ax[0].set(xscale="log", xlabel="GPS observations per trip", ylabel="trips",
              title="GPS observations per trip")
    ax[0].legend()
    mb = pd.DataFrame({ext: stats[f"{ext}_bytes"] / 1e6 for ext in FILE_COLUMNS})
    ax[1].boxplot([mb[c] for c in mb.columns], tick_labels=[f".{c}" for c in mb.columns])
    ax[1].set(yscale="log", ylabel="MB", title="File size per trip (train + student_test)")
    rows = pd.DataFrame({ext: stats[f"{ext}_rows"] for ext in FILE_COLUMNS})
    ax[2].boxplot([rows[c] for c in rows.columns], tick_labels=[f".{c}" for c in rows.columns])
    ax[2].set(yscale="log", ylabel="rows", title="Rows per file (train + student_test)")
    fig.tight_layout()
    fig.savefig(OUT / "fig01_inventory.png", dpi=DPI)
    plt.close(fig)

    # 2 missing / invalid values
    issues = ["track_missing_values", "nodes_missing_values", "arcs_missing_values", "route_missing_values",
              "gps_nonfinite_rows", "gps_out_of_range_rows", "gps_null_island_rows", "t_duplicate_steps",
              "t_backward_steps", "dropped_points", "route_invalid_ids", "invalid_arc_refs"]
    tab = stats.groupby("partition")[issues].sum().T
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(issues))
    for k, part in enumerate(PARTS):
        vals = tab[part].to_numpy()
        ax.bar(xs + (k - 0.5) * 0.4, vals, width=0.4, color=PART_COLORS[part], label=part)
        for x, val in zip(xs + (k - 0.5) * 0.4, vals):
            ax.text(x, val, str(int(val)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, issues, rotation=35, ha="right")
    ax.set(ylabel="count (summed over trips)", title="Missing-value and invalid-value checks")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig02_missing_values.png", dpi=DPI)
    plt.close(fig)

    # 3 sampling
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    _hist_by_part(ax[0], arrays, "dt", bins=np.arange(0.5, 81.5, 1))
    ax[0].set(yscale="log", xlabel="time between consecutive fixes (s)", ylabel="steps",
              title="Sampling interval (usable fixes)")
    ax[0].legend()
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[1].hist(100 * s["share_1s_steps"], bins=np.linspace(80, 100, 21), alpha=0.6, color=PART_COLORS[part],
                   label=part)
        ax[2].scatter(s["usable_points"], s["max_gap_s"], color=PART_COLORS[part], label=part, s=18)
    ax[1].set(xlabel="% of steps exactly 1 s", ylabel="trips", title="Share of 1 s steps per trip")
    ax[2].set(xscale="log", yscale="log", xlabel="GPS observations", ylabel="largest gap (s)",
              title="Largest time gap per trip")
    ax[2].axhline(10, ls="--", c="gray", lw=0.8)
    ax[2].axhline(60, ls="--", c="gray", lw=0.8)
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig03_sampling_intervals.png", dpi=DPI)
    plt.close(fig)

    # 4 speed
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    _hist_by_part(ax[0], arrays, "speed", bins=np.linspace(0, 80, 81), density=True)
    ax[0].axvline(SPEED_LIMIT_MPS, c="red", ls="--", lw=1, label="70 m/s plausibility limit")
    ax[0].set(yscale="log", xlabel="speed between consecutive fixes (m/s)", ylabel="density",
              title="GPS step speed (usable fixes)")
    ax[0].legend()
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[1].scatter(s["length_km"], s["mean_speed_kmh"], color=PART_COLORS[part], label=part, s=18)
    ax[1].set(xscale="log", xlabel="trip length (km)", ylabel="mean speed (km/h)", title="Trip length vs mean speed")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig04_speed.png", dpi=DPI)
    plt.close(fig)

    # 5 geographic extent
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[0].scatter(s["centroid_lon"], s["centroid_lat"], color=PART_COLORS[part], label=part, s=22)
        ax[1].hist(s["length_km"], bins=np.logspace(-0.5, 2.5, 25), alpha=0.6, color=PART_COLORS[part], label=part)
        ax[2].scatter(s["extent_x_km"], s["extent_y_km"], color=PART_COLORS[part], label=part, s=18)
    ax[0].set(xlim=(-180, 180), ylim=(-60, 75), xlabel="longitude", ylabel="latitude",
              title="Trip centroids (lon/lat)")
    ax[0].axhline(0, c="gray", lw=0.6)
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[1].set(xscale="log", xlabel="GPS trajectory length (km)", ylabel="trips", title="Trajectory length")
    ax[1].legend()
    ax[2].set(xscale="log", yscale="log", xlabel="east-west extent (km)", ylabel="north-south extent (km)",
              title="Bounding-box extent per trip")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig05_geographic_extent.png", dpi=DPI)
    plt.close(fig)

    # 6 coordinate distributions
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    _hist_by_part(ax[0], arrays, "lon", bins=np.linspace(-180, 180, 73))
    _hist_by_part(ax[1], arrays, "lat", bins=np.linspace(-60, 75, 55))
    _hist_by_part(ax[2], arrays, "offset_km", bins=np.linspace(0, 30, 61), density=True)
    ax[0].set(xlabel="longitude", ylabel="GPS fixes", title="GPS longitude distribution")
    ax[1].set(xlabel="latitude", ylabel="GPS fixes", title="GPS latitude distribution")
    ax[2].set(xlabel="distance from trip centre (km)", ylabel="density",
              title="Spread of fixes around each trip centre")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig06_coordinate_distributions.png", dpi=DPI)
    plt.close(fig)

    # 7 trajectory patterns (12 train trips, projected metres)
    train_ids = stats.loc[stats["partition"] == "train", "trip_id"].tolist()
    pick = train_ids[:: max(1, len(train_ids) // 12)][:12]
    fig, axes = plt.subplots(3, 4, figsize=(15, 11))
    for ax_, tid in zip(axes.ravel(), pick):
        trip = load_trip(config.PARTITIONS["train"] / tid)
        tc = trip.track[clean_track(trip.track)]
        xy = Projector.for_points(tc[:, :2]).to_xy(tc[:, :2])
        xy = (xy - np.median(xy, axis=0)) / 1000
        sc = ax_.scatter(xy[:, 0], xy[:, 1], c=tc[:, 2] - tc[0, 2], s=0.6, cmap="viridis")
        ax_.set_aspect("equal")
        s = stats[stats["trip_id"] == tid].iloc[0]
        ax_.set_title(f"{tid}: {s['length_km']:.1f} km, {s['usable_points']:,} fixes", fontsize=9)
        ax_.tick_params(labelsize=7)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="seconds since first fix (last panel scale)")
    fig.suptitle("GPS trajectory patterns, 12 train trips (km, UTM relative to trip centre)")
    fig.savefig(OUT / "fig07_trajectory_patterns.png", dpi=DPI)
    plt.close(fig)

    # 8 road network
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[0].scatter(s["nodes"], s["arcs"], color=PART_COLORS[part], label=part, s=18)
        ax[1].hist(100 * s["largest_scc_share"], bins=np.linspace(0, 100, 21), alpha=0.6,
                   color=PART_COLORS[part], label=part)
        ax[2].hist(100 * s["twin_share"], bins=np.linspace(0, 100, 21), alpha=0.6,
                   color=PART_COLORS[part], label=part)
    _hist_by_part(ax[3], arrays, "arc_len_sample", bins=np.logspace(-1, 3.5, 46), density=True)
    ax[0].set(xscale="log", yscale="log", xlabel="nodes", ylabel="arcs", title="Road-network size per trip")
    ax[1].set(xlabel="% of nodes in largest strongly connected component", ylabel="trips",
              title="Network connectivity")
    ax[2].set(xlabel="% of arcs with a reverse-direction twin", ylabel="trips", title="Two-way roads")
    ax[3].set(xscale="log", xlabel="arc length (m)", ylabel="density", title="Arc length (sample per trip)")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig08_road_network.png", dpi=DPI)
    plt.close(fig)

    # 9 reference routes
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for part in PARTS:
        s = stats[stats["partition"] == part]
        ax[0].scatter(s["length_km"], s["route_km"], color=PART_COLORS[part], label=part, s=18)
        ax[1].hist(s["route_arcs"], bins=np.logspace(1.5, 3.5, 21), alpha=0.6, color=PART_COLORS[part], label=part)
    lim = [0.3, float(max(stats["length_km"].max(), stats["route_km"].max()) * 1.3)]
    ax[0].plot(lim, lim, "--", c="gray", lw=0.8, label="route km = GPS km")
    ax[0].set(xscale="log", yscale="log", xlabel="GPS trajectory length (km)", ylabel="reference route length (km)",
              title="GPS length vs reference-route length")
    ax[0].legend()
    ax[1].set(xscale="log", xlabel="arcs in reference route", ylabel="trips", title="Reference-route length (arcs)")
    ax[1].legend()
    bl = stats[stats["route_broken_links"] > 0].sort_values("route_broken_links", ascending=False)
    ax[2].bar(bl["trip_id"].str[-3:] + "\n" + bl["partition"].str[:5], bl["route_broken_links"],
              color=[PART_COLORS[p] for p in bl["partition"]])
    ax[2].set(xlabel="trip id (last 3 digits)", ylabel="broken links",
              title=f"Broken links in reference routes ({int(bl['route_broken_links'].sum())} in {len(bl)} trips)")
    ax[2].tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig09_reference_routes.png", dpi=DPI)
    plt.close(fig)

    # 10 GPS noise relative to the reference route
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    _hist_by_part(ax[0], arrays, "label_dist", bins=np.linspace(0, 300, 151))
    ax[0].axvline(FAR_M, c="red", ls="--", lw=1, label="50 m label threshold")
    ax[0].set(yscale="log", xlabel="distance from GPS fix to its aligned reference arc (m, axis cut at 300)",
              ylabel="GPS fixes", title="GPS noise relative to the reference route")
    ax[0].legend()
    s = stats.sort_values("share_points_gt_50m", ascending=False)
    s = s[s["share_points_gt_50m"] > 0]
    ax[1].bar(s["trip_id"].str[-3:], 100 * s["share_points_gt_50m"],
              color=[PART_COLORS[p] for p in s["partition"]])
    ax[1].set(xlabel="trip id (last 3 digits)", ylabel="% of fixes > 50 m from route",
              title="Trips with fixes far from their reference route")
    ax[1].tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_gps_vs_route_distance.png", dpi=DPI)
    plt.close(fig)


# ----------------------------------------------------------------------------- tables / text
def summary_by_partition(stats: pd.DataFrame) -> pd.DataFrame:
    cols = ["usable_points", "duration_min", "length_km", "share_1s_steps", "max_gap_s", "median_speed_mps",
            "nodes", "arcs", "largest_scc_share", "twin_share", "route_arcs", "route_km",
            "label_dist_median_m", "share_points_gt_50m"]
    out = []
    for part in PARTS:
        s = stats[stats["partition"] == part]
        for c in cols:
            out.append({"partition": part, "statistic": c, "trips": len(s), "sum": s[c].sum(), "mean": s[c].mean(),
                        "median": s[c].median(), "min": s[c].min(), "max": s[c].max()})
    return pd.DataFrame(out)


def pct(x, q):
    return float(np.percentile(x, q))


def key_numbers(stats: pd.DataFrame, arrays: list[dict], broken: pd.DataFrame) -> dict:
    kn: dict = {"partitions": {}}
    for part in PARTS:
        s = stats[stats["partition"] == part]
        dt = np.concatenate([a["dt"] for a in arrays if a["partition"] == part])
        d = np.concatenate([a["label_dist"] for a in arrays if a["partition"] == part])
        kn["partitions"][part] = {
            "trips": int(len(s)),
            "gps_rows": int(s["track_rows"].sum()),
            "usable_points": int(s["usable_points"].sum()),
            "dropped_points": int(s["dropped_points"].sum()),
            "points_per_trip_min_median_max": [int(s["usable_points"].min()), int(s["usable_points"].median()),
                                               int(s["usable_points"].max())],
            "total_hours": float(s["duration_min"].sum() / 60),
            "total_km": float(s["length_km"].sum()),
            "share_1s_steps": float(np.isclose(dt, 1.0).mean()),
            "steps_gt_10s": int((dt > 10).sum()), "trips_with_gap_gt_10s": int((s["gaps_gt_10s"] > 0).sum()),
            "steps_gt_60s": int((dt > 60).sum()), "trips_with_gap_gt_60s": int((s["gaps_gt_60s"] > 0).sum()),
            "max_gap_s": float(dt.max()),
            "steps_gt_70mps": int(s["steps_gt_70mps"].sum()),
            "missing_values_all_files": int(s[[f"{e}_missing_values" for e in FILE_COLUMNS]].sum().sum()),
            "nodes_median": int(s["nodes"].median()), "nodes_max": int(s["nodes"].max()),
            "arcs_median": int(s["arcs"].median()), "arcs_max": int(s["arcs"].max()),
            "twin_share_mean": float(s["twin_share"].mean()),
            "largest_scc_share_median": float(s["largest_scc_share"].median()),
            "largest_scc_share_min": float(s["largest_scc_share"].min()),
            "isolated_nodes": int(s["isolated_nodes"].sum()),
            "route_broken_links": int(s["route_broken_links"].sum()),
            "trips_with_broken_links": int((s["route_broken_links"] > 0).sum()),
            "route_uturns": int(s["route_uturns"].sum()),
            "route_repeated_arcs": int(s["route_repeated_arcs"].sum()),
            "label_dist_percentiles_m": {f"p{q}": pct(d, q) for q in (50, 75, 90, 95, 99)},
            "share_points_gt_50m": float((d > FAR_M).mean()),
            "trips_with_gt_20pct_far_points": s.loc[s["share_points_gt_50m"] > 0.2, "trip_id"].tolist(),
        }
    kn["utm_zones"] = int(stats["utm_epsg"].nunique())
    kn["southern_hemisphere_trips"] = stats.loc[stats["hemisphere"] == "S", "trip_id"].tolist()
    return kn


def write_data_quality(stats: pd.DataFrame, kn: dict, broken: pd.DataFrame, dtypes: pd.DataFrame, runtime: float):
    tr, te = kn["partitions"]["train"], kn["partitions"]["student_test"]
    allp = stats
    lines = [
        "# PathSnap data-quality log",
        "",
        "Generated by `eda/run_eda.py` from the raw files. Partitions: **train** (80 trips) and **student_test** "
        "(10 trips). The evaluator partition is not on disk and was not used. Numbers below are measured, not assumed.",
        "",
        "## Files and types (train + student_test)",
        f"- All four files (`.track`, `.nodes`, `.arcs`, `.route`) are present in "
        f"{int(allp[[f'has_{e}' for e in FILE_COLUMNS]].all(axis=1).sum())}/{len(allp)} trips.",
        f"- Column counts: .track {sorted(allp['track_cols'].unique().tolist())}, "
        f".nodes {sorted(allp['nodes_cols'].unique().tolist())}, .arcs {sorted(allp['arcs_cols'].unique().tolist())}, "
        f".route {sorted(allp['route_cols'].unique().tolist())} (matches the documented layout).",
        "- Inferred dtypes (pandas, no forced dtype): "
        + "; ".join(f"{r.file}.{r.column}: {r.dtype}" for r in dtypes.itertuples()) + ".",
        "- Track timestamps are relative seconds (first fix at t = "
        f"{allp['t_first_s'].min():g}-{allp['t_first_s'].max():g} s), not epoch times; "
        f"{int(allp['t_non_integer'].sum())} timestamps are non-integer.",
        "",
        "## Missing and invalid values",
        f"- Missing values over all files: train {tr['missing_values_all_files']}, "
        f"student_test {te['missing_values_all_files']}.",
        f"- GPS rows non-finite / out of range / (0,0): {int(allp['gps_nonfinite_rows'].sum())} / "
        f"{int(allp['gps_out_of_range_rows'].sum())} / {int(allp['gps_null_island_rows'].sum())}.",
        f"- Duplicate timestamps: {int(allp['t_duplicate_steps'].sum())}; backward timestamps: "
        f"{int(allp['t_backward_steps'].sum())}. Rows dropped by `clean_track`: train {tr['dropped_points']}, "
        f"student_test {te['dropped_points']}.",
        f"- Repeated consecutive fixes (same lon/lat, later timestamp - vehicle standing still): "
        f"{int(allp['dup_consecutive_fixes'].sum()):,} steps. **Decision:** kept (negligible; the heading feature "
        "is neutralised anyway when the vehicle barely moves).",
        f"- Route arc ids outside the network: {int(allp['route_invalid_ids'].sum())}; arcs referencing missing "
        f"nodes: {int(allp['invalid_arc_refs'].sum())}.",
        "",
        "## Sampling and gaps",
        f"- Share of exactly-1 s steps: train {100 * tr['share_1s_steps']:.2f} %, "
        f"student_test {100 * te['share_1s_steps']:.2f} %.",
        f"- Steps > 10 s: train {tr['steps_gt_10s']} (in {tr['trips_with_gap_gt_10s']} trips), "
        f"student_test {te['steps_gt_10s']} (in {te['trips_with_gap_gt_10s']} trips).",
        f"- Steps > 60 s: train {tr['steps_gt_60s']} (in {tr['trips_with_gap_gt_60s']} trips), "
        f"student_test {te['steps_gt_60s']} (in {te['trips_with_gap_gt_60s']} trips). Largest gap: train "
        f"{tr['max_gap_s']:.0f} s, student_test {te['max_gap_s']:.0f} s.",
        "- Trips with any gap > 60 s: "
        + (", ".join(f"{r.trip_id} ({r.partition}, max {r.max_gap_s:.0f} s)"
                     for r in allp[allp["gaps_gt_60s"] > 0].itertuples()) or "none") + ".",
        f"- Steps faster than 70 m/s: train {tr['steps_gt_70mps']}, student_test {te['steps_gt_70mps']}"
        + (" (trips: " + ", ".join(f"{r.trip_id} ({r.steps_gt_70mps})"
                                   for r in allp[allp["steps_gt_70mps"] > 0].itertuples()) + ")"
           if (allp["steps_gt_70mps"] > 0).any() else "") + ".",
        "",
        "## Geography and projection",
        f"- Trips fall in {kn['utm_zones']} different UTM zones; southern-hemisphere trips: "
        f"{', '.join(kn['southern_hemisphere_trips']) or 'none'}.",
        "- **Decision:** each trip is projected into the UTM zone of its median GPS fix (`pathsnap.geo.Projector`); "
        "distances/angles are computed in metres in that frame. Degrees are never used as distances.",
        "",
        "## Road network",
        f"- Size: nodes median {tr['nodes_median']:,} (max {max(tr['nodes_max'], te['nodes_max']):,}), arcs median "
        f"{tr['arcs_median']:,} (max {max(tr['arcs_max'], te['arcs_max']):,}) on train.",
        f"- Arcs with a reverse-direction twin (two-way roads): mean {100 * tr['twin_share_mean']:.1f} % (train), "
        f"{100 * te['twin_share_mean']:.1f} % (student_test). Twins share geometry, so direction must come from heading.",
        f"- Largest strongly connected component: median {100 * tr['largest_scc_share_median']:.1f} % of nodes "
        f"(min {100 * min(tr['largest_scc_share_min'], te['largest_scc_share_min']):.1f} %). "
        f"Isolated nodes: {int(allp['isolated_nodes'].sum())}. Zero-length arcs: {int(allp['zero_length_arcs'].sum())}; "
        f"self-loops: {int(allp['self_loops'].sum())}; duplicate arcs: {int(allp['duplicate_arcs'].sum())}.",
        "",
        "## Reference routes",
        f"- Broken links (arc i end != arc i+1 start): train {tr['route_broken_links']} in "
        f"{tr['trips_with_broken_links']} trips, student_test {te['route_broken_links']} in "
        f"{te['trips_with_broken_links']} trips (details in `broken_links.csv`).",
    ]
    for r in broken.itertuples():
        bridge = "not reachable within 5 km" if pd.isna(r.network_bridge_m) else f"network bridge {r.network_bridge_m:.0f} m"
        lines.append(f"  - {r.trip_id} ({r.partition}) position {r.route_pos}: straight gap "
                     f"{r.straight_gap_m:.0f} m, {bridge}")
    lines += [
        f"- U-turns (arc followed by its twin): {int(allp['route_uturns'].sum())}; repeated arcs within a route: "
        f"{int(allp['route_repeated_arcs'].sum())}.",
        f"- Route arcs inside the largest SCC: min over trips {100 * allp['route_arcs_in_largest_scc'].min():.1f} %.",
        "- **Assumption:** the route is reference information, not verified ground truth (as the brief says); "
        "broken links are left as they are (the per-point label alignment does not need connectivity).",
        "",
        "## GPS vs reference route",
        "- Distance from each usable GPS fix to its aligned reference arc (monotone alignment, `pathsnap.labels`), "
        "train percentiles: " + ", ".join(f"{k} {v:.1f} m" for k, v in tr["label_dist_percentiles_m"].items()) + ".",
        "- Student_test percentiles: " + ", ".join(f"{k} {v:.1f} m" for k, v in te["label_dist_percentiles_m"].items())
        + ".",
        f"- Fixes > 50 m from the route: train {100 * tr['share_points_gt_50m']:.1f} %, "
        f"student_test {100 * te['share_points_gt_50m']:.1f} %. The train distribution is bimodal: GPS noise "
        "(a few metres) plus whole stretches where track and route disagree.",
        "- Trips with > 20 % of fixes beyond 50 m: "
        + "; ".join(f"{r.trip_id} ({r.partition}): {100 * r.share_points_gt_50m:.0f} % of fixes, longest run "
                    f"{r.longest_far_run_points:,} fixes over {r.far_span_h:.1f} h; median net displacement over "
                    f"60 fixes only {r.far_net_disp_60s_m:.0f} m while the summed step length is "
                    f"{r.far_path_60s_m:.0f} m"
                    for r in allp[allp["share_points_gt_50m"] > 0.2].sort_values("share_points_gt_50m",
                                                                                ascending=False).itertuples())
        + ". These far fixes form clouds off the road network (see `trip_00000054_projected.png`, zoom panel): "
        "the vehicle is stationary (parked) and the GPS jitters by several metres per second. The reference "
        "route does not cover these periods, so the fixes have no trustworthy label.",
        f"- GPS jitter also inflates path-length statistics: in 60-fix windows with < 30 m net displacement the "
        f"summed step length has median {allp['stationary_path_m_per_60s'].median():.0f} m (median over trips); "
        "trajectory lengths and step speeds of trips with long stops are therefore overestimated.",
        "- Large time gaps are pauses, not jumps: e.g. "
        + ", ".join(f"{r.trip_id} gap {r.max_gap_s / 3600:.1f} h" for r in
                    allp.nlargest(2, "max_gap_s").itertuples())
        + " with the largest single GPS step in those trips only "
        + ", ".join(f"{r.max_step_m:.0f} m" for r in allp.nlargest(2, "max_gap_s").itertuples()) + ".",
        "- **Decision:** `LABEL_MAX_DIST_M = 50` - fixes farther than 50 m from their aligned arc get no label and "
        "are excluded from training and scoring (instead of being given a label we do not trust). Candidate radius "
        "and K were chosen separately on train (`results/dataset/candidate_grid.csv`, "
        "`results/dataset/candidate_miss_rate.png`).",
        "",
        f"_Runtime of `eda/run_eda.py`: {runtime:.0f} s._",
    ]
    (OUT / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)
    jobs = [(d, p) for p in PARTS for d in list_trips(config.PARTITIONS[p])]
    res = Parallel(n_jobs=N_JOBS)(delayed(analyse_trip)(d, p) for d, p in jobs)
    stats = pd.DataFrame([r[0] for r in res])
    arrays = [r[3] for r in res]
    broken = pd.DataFrame([b for r in res for b in r[2]],
                          columns=["partition", "trip_id", "route_pos", "arc_from", "arc_to", "straight_gap_m",
                                   "network_bridge_m", "arc_to_is_twin"])
    dt_all = pd.DataFrame([x for r in res for x in r[1]])
    dtypes = (dt_all.groupby(["file", "column", "dtype"]).size().rename("trips").reset_index())
    print(f"analysed {len(stats)} trips in {time.perf_counter() - t0:.1f} s", flush=True)

    stats.to_csv(OUT / "trip_stats.csv", index=False)
    inv_cols = ["partition", "trip_id"] + [f"has_{e}" for e in FILE_COLUMNS] + [f"{e}_bytes" for e in FILE_COLUMNS] \
        + [f"{e}_rows" for e in FILE_COLUMNS]
    stats[inv_cols].to_csv(OUT / "inventory.csv", index=False)
    inv_part = stats.groupby("partition").agg(
        trips=("trip_id", "size"), track_files=("has_track", "sum"), nodes_files=("has_nodes", "sum"),
        arcs_files=("has_arcs", "sum"), route_files=("has_route", "sum"),
        gps_observations=("track_rows", "sum"), nodes=("nodes_rows", "sum"), arcs=("arcs_rows", "sum"),
        route_arcs=("route_rows", "sum")).reset_index()
    inv_part["total_mb"] = [stats.loc[stats["partition"] == p, [f"{e}_bytes" for e in FILE_COLUMNS]].sum().sum() / 1e6
                            for p in inv_part["partition"]]
    inv_part.to_csv(OUT / "inventory_by_partition.csv", index=False)
    dtypes.to_csv(OUT / "dtypes.csv", index=False)
    miss_cols = [f"{e}_missing_values" for e in FILE_COLUMNS] + [
        "gps_nonfinite_rows", "gps_out_of_range_rows", "gps_null_island_rows", "t_duplicate_steps",
        "t_backward_steps", "t_non_integer", "dup_consecutive_fixes", "dropped_points"]
    stats.groupby("partition")[miss_cols].sum().reset_index().to_csv(OUT / "missing_values.csv", index=False)
    net_cols = ["partition", "trip_id", "nodes", "arcs", "invalid_arc_refs", "zero_length_arcs", "self_loops",
                "duplicate_arcs", "twin_share", "isolated_nodes", "weak_components", "largest_wcc_share",
                "strong_components", "largest_scc_share", "arc_length_median_m", "arc_length_p95_m",
                "network_km", "mean_out_degree"]
    stats[net_cols].to_csv(OUT / "network_stats.csv", index=False)
    route_cols = ["partition", "trip_id", "route_arcs", "route_invalid_ids", "route_km", "length_km",
                  "route_broken_links", "route_repeated_arcs", "route_uturns", "route_arcs_in_largest_scc",
                  "label_dist_median_m", "label_dist_p90_m", "share_points_gt_50m", "longest_far_run_points",
                  "far_span_h", "far_net_disp_60s_m", "far_path_60s_m", "stationary_share_60s"]
    stats[route_cols].to_csv(OUT / "route_stats.csv", index=False)
    broken.to_csv(OUT / "broken_links.csv", index=False)
    anomalies = stats[(stats["gaps_gt_60s"] > 0) | (stats["steps_gt_70mps"] > 0) | (stats["route_broken_links"] > 0)
                      | (stats["share_points_gt_50m"] > 0.2)][
        ["partition", "trip_id", "max_gap_s", "gaps_gt_60s", "steps_gt_70mps", "route_broken_links",
         "share_points_gt_50m"]]
    anomalies.to_csv(OUT / "anomalies.csv", index=False)
    summary_by_partition(stats).to_csv(OUT / "summary_by_partition.csv", index=False)

    make_figures(stats, arrays, broken)
    print(f"figures done at {time.perf_counter() - t0:.1f} s", flush=True)

    world_map(stats).save(MAPS / "eda_world_overview.html")
    for tid, (part, note) in REPRESENTATIVE.items():
        d = config.PARTITIONS[part] / tid
        trip_map(d, note).save(MAPS / f"eda_trip_{tid}.html")
        trip_static_png(d, note, OUT / f"trip_{tid}_projected.png")
    print(f"maps done at {time.perf_counter() - t0:.1f} s", flush=True)

    kn = key_numbers(stats, arrays, broken)
    runtime = time.perf_counter() - t0
    kn["runtime_s"] = runtime
    (OUT / "key_numbers.json").write_text(json.dumps(kn, indent=2))
    write_data_quality(stats, kn, broken, dtypes.drop_duplicates(["file", "column", "dtype"]), runtime)
    print(json.dumps(kn, indent=1))
    print(f"done in {runtime:.1f} s")


if __name__ == "__main__":
    main()
