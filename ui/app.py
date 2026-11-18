"""PathSnap evaluator UI (Laptop 2).

    streamlit run ui/app.py

The UI only talks to the PathSnap API over HTTP:
  * GET /health   connection status, loaded model, HMM on/off
  * GET /trips    the evaluation trips to choose from
  * POST /predict run the pipeline on the chosen trip

The response holds the GPS fixes, the predicted route and aggregate metrics
computed server-side. The UI never sees the reference route, so it cannot
draw one.

Set the API address in the sidebar, or with the PATHSNAP_API_URL environment variable.
"""
from __future__ import annotations

import math
import os

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium

DEFAULT_API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")
PREDICT_TIMEOUT_S = 180
GPS_COLOR = "#2b6cb0"
ROUTE_COLOR = "#e4572e"

st.set_page_config(page_title="PathSnap map-matching", page_icon=":world_map:", layout="wide")


# ---------------------------------------------------------------- API calls
@st.cache_data(ttl=10, show_spinner=False)
def get_health(api: str) -> dict:
    r = requests.get(f"{api}/health", timeout=5)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=30, show_spinner=False)
def get_trips(api: str) -> list[dict]:
    r = requests.get(f"{api}/trips", timeout=15)
    r.raise_for_status()
    return r.json()


def run_predict(api: str, trip_id: str) -> dict:
    r = requests.post(f"{api}/predict", json={"trip_id": trip_id}, timeout=PREDICT_TIMEOUT_S)
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise RuntimeError(f"API returned {r.status_code}: {detail}")
    return r.json()


# ---------------------------------------------------------------- formatting
def pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.2f} %"


def view_for(bounds, width_px: int = 900, height_px: int = 620) -> tuple[list[float], int]:
    """Centre and Web-Mercator zoom level that fit ``[[south, west], [north, east]]`` in the map frame.

    Passed to st_folium explicitly: Leaflet's fitBounds can run before the
    Streamlit iframe has its final size, which leaves the map zoomed far out.
    """
    (s, w), (n, e) = bounds
    merc = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))  # noqa: E731
    zx = math.log2(width_px * 360.0 / (256.0 * max(e - w, 1e-6)))
    zy = math.log2(height_px * 2 * math.pi / (256.0 * max(merc(n) - merc(s), 1e-9)))
    return [(s + n) / 2, (w + e) / 2], int(max(2, min(18, math.floor(min(zx, zy)))))


def build_map(res: dict) -> folium.Map:
    gps = res.get("gps") or []
    bounds = res.get("bounds")
    center = gps[len(gps) // 2] if gps else [0, 0]
    # OpenStreetMap only: CartoDB basemaps now need an API key and render blank without one.
    m = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap", control_scale=True)

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lat, lon in gps:
        folium.CircleMarker([lat, lon], radius=2, color=GPS_COLOR, weight=0, fill=True,
                            fill_color=GPS_COLOR, fill_opacity=0.7).add_to(gps_layer)
    gps_layer.add_to(m)

    route_layer = folium.FeatureGroup(name="Predicted route", show=True)
    for line in res.get("route") or []:
        if len(line) >= 2:
            folium.PolyLine(line, color=ROUTE_COLOR, weight=5, opacity=0.85,
                            tooltip="Predicted route").add_to(route_layer)
    route_layer.add_to(m)

    ends = folium.FeatureGroup(name="Start / end", show=True)
    if gps:
        folium.CircleMarker(gps[0], radius=8, color="#1a7f37", fill=True, fill_color="#2da44e",
                            fill_opacity=1, tooltip="Start").add_to(ends)
        folium.CircleMarker(gps[-1], radius=8, color="#a40e26", fill=True, fill_color="#cf222e",
                            fill_opacity=1, tooltip="End").add_to(ends)
    ends.add_to(m)

    legend = f"""
    <div style="position: fixed; bottom: 28px; left: 28px; z-index: 9999; background: white;
                padding: 10px 14px; border: 1px solid #bbb; border-radius: 8px; font-size: 13px;
                box-shadow: 0 1px 4px rgba(0,0,0,.2); line-height: 1.7;">
      <b>Legend</b><br>
      <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:{GPS_COLOR};"></span>
      &nbsp;GPS observation<br>
      <span style="display:inline-block;width:22px;height:4px;background:{ROUTE_COLOR};vertical-align:middle;"></span>
      &nbsp;Predicted route<br>
      <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#2da44e;"></span>
      &nbsp;Start&nbsp;&nbsp;
      <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#cf222e;"></span>
      &nbsp;End
    </div>"""
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    if bounds:
        m.fit_bounds(bounds, padding=(20, 20))
    return m


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Connection")
    api = st.text_input("API base URL", value=DEFAULT_API, help="Laptop 1 address, e.g. http://192.168.1.20:8000")
    api = api.strip().rstrip("/")
    if st.button("Check connection", width="stretch"):
        get_health.clear()
        get_trips.clear()
    health, health_err = None, None
    try:
        health = get_health(api)
    except Exception as exc:  # noqa: BLE001 - show any connection problem to the user
        health_err = str(exc)
    if health:
        st.success(f"Connected ({health.get('status')})")
        st.markdown(
            f"**Model:** {health.get('model_name')}  \n"
            f"**Mode:** {health.get('mode')}  \n"
            f"**HMM decoding:** {'on' if health.get('hmm_enabled') else 'off'}  \n"
            f"**Trips available:** {health.get('n_trips')}"
        )
    else:
        st.error("API not reachable")
        st.caption(health_err or "")
    st.divider()
    st.caption("Ground truth stays on the API server. This UI only receives the predicted route "
               "and aggregate metrics.")

# ---------------------------------------------------------------- main
st.title("PathSnap: GPS map-matching")
st.caption("Pick an evaluation trip. The API on Laptop 1 snaps its noisy GPS track onto the road network "
           "(candidate classifier + HMM/Viterbi) and scores the result server-side.")

if not health:
    st.info("Connect to the API to load the evaluation trips (check the URL in the sidebar).")
    st.stop()

try:
    trips = get_trips(api)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load trips: {exc}")
    st.stop()
if not trips:
    st.warning("The API has no trips in its evaluation folder.")
    st.stop()

labels = {t["trip_id"]: f"Trip {t['trip_id']}  ·  {t['n_points']:,} GPS points" for t in trips}
col_sel, col_btn = st.columns([4, 1], vertical_alignment="bottom")
with col_sel:
    trip_id = st.selectbox(f"Evaluation trips ({len(trips)})", options=list(labels), format_func=labels.get)
with col_btn:
    run = st.button("Run map-matching", type="primary", width="stretch")

if run:
    with st.spinner(f"Running the PathSnap pipeline on trip {trip_id} ..."):
        try:
            st.session_state["result"] = run_predict(api, trip_id)
            st.session_state.pop("error", None)
        except requests.exceptions.Timeout:
            st.session_state["error"] = f"The API did not answer within {PREDICT_TIMEOUT_S} s."
        except requests.exceptions.ConnectionError:
            st.session_state["error"] = "Lost connection to the API. Is it still running on Laptop 1?"
        except Exception as exc:  # noqa: BLE001
            st.session_state["error"] = str(exc)

if st.session_state.get("error"):
    st.error(st.session_state["error"])

res = st.session_state.get("result")
if res:
    mt, tm = res.get("metrics") or {}, res.get("timings") or {}
    st.subheader(f"Trip {res['trip_id']}")
    st.caption(f"Model: {res.get('model_name')}  ·  HMM decoding: {'on' if res.get('used_hmm') else 'off'}"
               f"  ·  HMM breaks: {res.get('n_breaks', 0)}  ·  GPS shown: {res['gps_downsample']['note']}")
    c = st.columns(6)
    c[0].metric("Accuracy (final route)", pct(mt.get("accuracy")),
                help="Share of labelled GPS points matched to the correct road segment.")
    c[1].metric("Classifier-only accuracy", pct(mt.get("accuracy_classifier_only")),
                help="Same metric before HMM/Viterbi decoding (best candidate per point).")
    c[2].metric("Route F1", pct(mt.get("route_f1")),
                help="Length-weighted overlap between predicted and reference route segments.")
    c[3].metric("Route continuity", pct(mt.get("continuity")),
                help="Share of consecutive predictions that are connected in the road network.")
    c[4].metric("GPS points", f"{res.get('n_points', 0):,}")
    c[5].metric("Processing time", f"{tm.get('request_total_s', 0):.2f} s")

    view = view_for(res["bounds"]) if res.get("bounds") else (None, None)
    st_folium(build_map(res), height=620, use_container_width=True, returned_objects=[],
              center=view[0], zoom=view[1], key=f"map_{res['trip_id']}")

    with st.expander("Details"):
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Metrics**")
            st.json(mt)
        with d2:
            st.markdown("**Timings (s)**")
            st.json({k: round(v, 3) for k, v in tm.items() if isinstance(v, (int, float))})
            st.markdown("**Route**")
            st.json(res.get("route_summary", {}))
