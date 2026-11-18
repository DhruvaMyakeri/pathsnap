"""Write eda/EDA.ipynb: a presentation notebook over the outputs of eda/run_eda.py.

The notebook only loads and displays results (tables, figures, one Folium map),
so it executes in seconds. Rebuild + execute:

    python eda/run_eda.py
    python eda/build_notebook.py
    python -m nbconvert --to notebook --execute --inplace eda/EDA.ipynb
"""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent

cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# PathSnap - dataset exploration (EDA)

Partitions analysed: **train** (80 trips) and **student_test** (10 trips). The evaluator partition is not on
disk and is not used. Every table, figure and map below is produced by `eda/run_eda.py` from the raw files;
this notebook only displays them. Each section states the partition it uses.
""")
code("""
import json, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = Path.cwd().parent if Path.cwd().name == "eda" else Path.cwd()
sys.path.insert(0, str(ROOT / "eda"))
EDA, DATASET, MAPS = ROOT / "results" / "eda", ROOT / "results" / "dataset", ROOT / "results" / "maps"
pd.set_option("display.width", 200, "display.max_columns", 30)
kn = json.loads((EDA / "key_numbers.json").read_text())
keys = ["trips", "gps_rows", "usable_points", "points_per_trip_min_median_max", "total_hours", "total_km",
        "share_1s_steps", "steps_gt_10s", "steps_gt_60s", "max_gap_s", "steps_gt_70mps", "missing_values_all_files",
        "nodes_median", "arcs_median", "twin_share_mean", "largest_scc_share_median", "route_broken_links",
        "trips_with_broken_links", "share_points_gt_50m"]
pd.DataFrame({p: {k: kn["partitions"][p][k] for k in keys} for p in kn["partitions"]})
""")

md("## 1. Inventory (train + student_test)\nFile presence, sizes and row counts; inferred column types.")
code("""
display(pd.read_csv(EDA / "inventory_by_partition.csv"))
display(pd.read_csv(EDA / "dtypes.csv"))
Image(EDA / "fig01_inventory.png")
""")

md("## 2. Missing and invalid values (train + student_test)")
code("""
display(pd.read_csv(EDA / "missing_values.csv"))
Image(EDA / "fig02_missing_values.png")
""")

md("## 3. Timestamp ordering and sampling gaps (train + student_test)")
code("""
Image(EDA / "fig03_sampling_intervals.png")
""")
code("""
s = pd.read_csv(EDA / "trip_stats.csv", dtype={"trip_id": str})
s.loc[s["gaps_gt_60s"] > 0, ["partition", "trip_id", "usable_points", "share_1s_steps", "max_gap_s",
                             "gaps_gt_10s", "gaps_gt_60s", "max_step_m"]].sort_values("max_gap_s", ascending=False)
""")

md("## 4. Speeds between consecutive fixes (train + student_test)")
code("""
display(s.loc[s["steps_gt_70mps"] > 0, ["partition", "trip_id", "median_speed_mps", "p95_speed_mps",
                                        "max_speed_mps", "steps_gt_70mps"]])
Image(EDA / "fig04_speed.png")
""")

md("## 5. Geographic extent and GPS coordinate distributions (train + student_test)")
code("""
print("UTM zones used:", kn["utm_zones"], "| southern-hemisphere trips:", kn["southern_hemisphere_trips"])
display(Image(EDA / "fig05_geographic_extent.png"))
Image(EDA / "fig06_coordinate_distributions.png")
""")

md("## 6. Trajectory patterns (12 train trips, projected metres)")
code("""
Image(EDA / "fig07_trajectory_patterns.png")
""")

md("## 7. Road-network size and connectivity (train + student_test)")
code("""
net = pd.read_csv(EDA / "network_stats.csv", dtype={"trip_id": str})
display(net.groupby("partition")[["nodes", "arcs", "twin_share", "isolated_nodes", "strong_components",
                                  "largest_scc_share", "zero_length_arcs", "self_loops", "duplicate_arcs",
                                  "arc_length_median_m"]].describe().T.round(3))
Image(EDA / "fig08_road_network.png")
""")

md("## 8. Reference routes (train + student_test)")
code("""
r = pd.read_csv(EDA / "route_stats.csv", dtype={"trip_id": str})
display(r.groupby("partition")[["route_arcs", "route_km", "route_broken_links", "route_repeated_arcs",
                                "route_uturns", "route_arcs_in_largest_scc"]].agg(["sum", "median", "min", "max"]).T)
display(pd.read_csv(EDA / "broken_links.csv", dtype={"trip_id": str}))
Image(EDA / "fig09_reference_routes.png")
""")

md("""
## 9. GPS noise relative to the reference route (train + student_test)
Distance from each fix to its aligned reference arc (monotone alignment, `pathsnap.labels`, no cut-off).
The candidate radius / K grid was computed on **train** only by `scripts/candidate_params.py`.
""")
code("""
display(pd.DataFrame({p: kn["partitions"][p]["label_dist_percentiles_m"] for p in kn["partitions"]}).round(1))
display(r.sort_values("share_points_gt_50m", ascending=False).head(8)[
    ["partition", "trip_id", "share_points_gt_50m", "longest_far_run_points", "label_dist_median_m"]])
display(Image(EDA / "fig10_gps_vs_route_distance.png"))
g = pd.read_csv(DATASET / "candidate_grid.csv")
display((100 * g.pivot(index="k", columns="radius_m", values="miss_rate")).round(2))
Image(DATASET / "candidate_miss_rate.png")
""")

md("## 10. Representative trips (train, projected metres)")
code("""
for tid in ["00000001", "00000019", "00000054"]:
    display(Image(EDA / f"trip_{tid}_projected.png"))
""")

md("""
## 11. Interactive Folium map - trip 00000001 (train)
GPS fixes, road network within 300 m and the reference route, with layer control. The other maps are in
`results/maps/` (`eda_world_overview.html`, `eda_trip_00000019.html`, `eda_trip_00000054.html`).
""")
code("""
from pathsnap import config
import run_eda
run_eda.trip_map(config.PARTITIONS["train"] / "00000001", "clean urban trip")
""")

md("## 12. Data-quality log (train + student_test)")
code("""
Markdown((EDA / "data_quality.md").read_text(encoding="utf-8"))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, HERE / "EDA.ipynb")
print("wrote", HERE / "EDA.ipynb")
