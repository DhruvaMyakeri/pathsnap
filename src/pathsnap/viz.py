"""Folium map of one trip: GPS observations, nearby road network, reference and predicted routes."""
from __future__ import annotations

import folium
import numpy as np
from branca.element import Element

from .pipeline import TripContext, route_polylines

COLORS = {"gps": "#1f6fd1", "network": "#9a9a9a", "reference": "#2ca02c",
          "predicted": "#d62728", "classifier": "#ff9f1c"}


def _legend(items: list[tuple[str, str, str]]) -> Element:
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:18px;height:{h};'
        f'background:{c};margin-right:6px;vertical-align:middle;border-radius:2px"></span>{label}</div>'
        for label, c, h in items)
    return Element(
        '<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
        'padding:8px 10px;border:1px solid #bbb;border-radius:6px;font:13px sans-serif;'
        f'box-shadow:0 1px 4px rgba(0,0,0,.2)">{rows}</div>')


def nearby_arcs(ctx: TripContext, radius_m: float = 200.0) -> np.ndarray:
    pt, arc = ctx.net.query(ctx.xy, radius_m)
    return np.unique(arc)


def trip_map(ctx: TripContext, predicted_route=None, reference_route=None, classifier_route=None,
             title: str = "", network_radius_m: float = 120.0, max_gps_points: int = 4000) -> folium.Map:
    lat, lon = ctx.lonlat[:, 1], ctx.lonlat[:, 0]
    # OpenStreetMap only: CartoDB basemaps now need an API key and render blank without one.
    m = folium.Map(tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    net_layer = folium.FeatureGroup(name=f"Road network (within {network_radius_m:.0f} m)", show=True)
    for seg in ctx.net.arcs_lonlat(nearby_arcs(ctx, network_radius_m), ctx.nodes_lonlat):
        folium.PolyLine(seg, color=COLORS["network"], weight=1.5, opacity=0.6).add_to(net_layer)
    net_layer.add_to(m)

    def add_route(arcs, name, color, weight, dash=None):
        if arcs is None:
            return
        arcs = np.asarray(arcs, dtype=np.int64)
        arcs = arcs[(arcs >= 0) & (arcs < ctx.net.n_arcs)]
        if len(arcs) == 0:
            return
        layer = folium.FeatureGroup(name=name)
        for line in route_polylines(ctx, arcs):
            folium.PolyLine(line, color=color, weight=weight, opacity=0.85, dash_array=dash).add_to(layer)
        layer.add_to(m)

    # GPS first so the routes are drawn on top of the (dense) observation cloud.
    gps = folium.FeatureGroup(name="GPS observations")
    step = max(1, len(lat) // max_gps_points)
    for i in range(0, len(lat), step):
        folium.CircleMarker([lat[i], lon[i]], radius=1.5, color=COLORS["gps"], fill=True,
                            fill_opacity=0.45, weight=0).add_to(gps)
    gps.add_to(m)
    add_route(reference_route, "Reference route (ground truth)", COLORS["reference"], 7)
    add_route(classifier_route, "Classifier-only route", COLORS["classifier"], 3, dash="6 6")
    add_route(predicted_route, "Predicted route (classifier + HMM)", COLORS["predicted"], 3.5)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    items = [("GPS observations" + (f" (every {step}th)" if step > 1 else ""), COLORS["gps"], "8px"),
             ("Road network", COLORS["network"], "3px")]
    if reference_route is not None:
        items.append(("Reference route", COLORS["reference"], "6px"))
    if classifier_route is not None:
        items.append(("Classifier-only route", COLORS["classifier"], "3px"))
    if predicted_route is not None:
        items.append(("Predicted route (classifier + HMM)", COLORS["predicted"], "4px"))
    if title:
        items.insert(0, (f"<b>{title}</b>", "transparent", "0px"))
    m.get_root().html.add_child(_legend(items))
    return m
