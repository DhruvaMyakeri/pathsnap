"""Choose the candidate search radius R and per-point cap K from TRAINING data only.

For every training trip, generate candidates with a generous radius (150 m, up
to 64 per point) and align the reference route to the points. Then, for a grid
of (R, K), measure:
  * candidate miss rate: share of labelled points whose reference arc is not
    among the K closest arcs within R metres (no classifier can recover these)
  * mean candidates per point (cost of training / decoding)

Outputs
  results/dataset/candidate_grid.csv
  results/dataset/label_distance_stats.json
  results/dataset/candidate_miss_rate.png   (miss rate vs K, and the label-distance histogram)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from pathsnap import config  # noqa: E402
from pathsnap.candidates import generate_candidates  # noqa: E402
from pathsnap.data import clean_track, list_trips, load_route, load_trip  # noqa: E402
from pathsnap.geo import Projector  # noqa: E402
from pathsnap.labels import point_labels  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

RADII = [10, 15, 20, 25, 30, 40, 50, 75, 100, 150]
KS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 64]


def trip_stats(trip_dir):
    trip = load_trip(trip_dir)
    track = trip.track[clean_track(trip.track)]
    proj = Projector.for_points(track[:, :2])
    xy = proj.to_xy(track[:, :2])
    net = RoadNetwork(proj.to_xy(trip.nodes), trip.arcs)
    lab = point_labels(net, xy, load_route(trip_dir), max_dist=np.inf)
    cands = generate_candidates(net, xy, radius=150.0, max_radius=150.0, max_k=64)
    n = len(xy)
    ref = lab["label_arc"]
    hit = cands[cands["arc"].to_numpy() == ref[cands["pt"].to_numpy()]]
    ref_rank = np.full(n, 10_000)
    ref_dist = np.full(n, np.inf)
    ref_rank[hit["pt"].to_numpy()] = hit["rank"].to_numpy()
    ref_dist[hit["pt"].to_numpy()] = hit["dist"].to_numpy()
    # number of arcs within each radius, per point
    within = np.stack([np.bincount(cands.loc[cands["dist"] <= r, "pt"], minlength=n) for r in RADII], axis=1)
    return pd.DataFrame({"trip_id": trip.trip_id, "label_dist": lab["label_dist"],
                         "ref_rank": ref_rank, "ref_dist": ref_dist}), within


def main():
    out_dir = config.RESULTS_DIR / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)
    trips = list_trips(config.PARTITIONS["train"])
    res = Parallel(n_jobs=12)(delayed(trip_stats)(d) for d in trips)
    pts = pd.concat([r[0] for r in res], ignore_index=True)
    within = np.concatenate([r[1] for r in res])

    d = pts["label_dist"].to_numpy()
    pct = {f"p{q}": float(np.percentile(d, q)) for q in (50, 75, 90, 95, 98, 99, 99.5, 99.9)}
    stats = {"partition": "train", "points": int(len(d)), "label_distance_m": pct,
             "share_within": {f"{m}m": float((d <= m).mean()) for m in (10, 20, 30, 50, 75, 100)}}
    (out_dir / "label_distance_stats.json").write_text(json.dumps(stats, indent=2))

    # Miss rate is measured over labelled points (aligned within LABEL_MAX_DIST_M of the
    # reference route); points beyond it are excluded from training and scoring anyway.
    labelled = d <= config.LABEL_MAX_DIST_M
    stats["labelled_share"] = float(labelled.mean())
    (out_dir / "label_distance_stats.json").write_text(json.dumps(stats, indent=2))
    rows = []
    for ri, r in enumerate(RADII):
        for k in KS:
            found = (pts["ref_dist"].to_numpy() <= r) & (pts["ref_rank"].to_numpy() < k)
            rows.append({"radius_m": r, "k": k, "miss_rate": 1 - found[labelled].mean(),
                         "miss_rate_all_points": 1 - found.mean(),
                         "mean_cands": np.minimum(within[:, ri], k).mean()})
    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "candidate_grid.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in [15, 20, 30, 50, 100]:
        g = grid[grid["radius_m"] == r]
        ax[0].plot(g["k"], 100 * g["miss_rate"], marker="o", label=f"R = {r} m")
    ax[0].set(xscale="log", xlabel="K (max candidates per point)", ylabel="candidate miss rate (%)",
              title="Candidate miss rate vs K (train, labelled points)")
    ax[0].set_xticks(KS, [str(k) for k in KS])
    ax[0].grid(alpha=.3)
    ax[0].legend()
    clipped = np.clip(d, 0, 100)
    ax[1].hist(clipped, bins=100, color="#378ADD")
    ax[1].set(xlabel="distance from GPS point to its aligned reference arc (m, clipped at 100)",
              ylabel="points", title="GPS noise relative to the reference route (train)")
    ax[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_dir / "candidate_miss_rate.png", dpi=130)
    print(json.dumps(stats, indent=1))
    print(grid.pivot(index="k", columns="radius_m", values="miss_rate").round(4).to_string())
    print(grid.pivot(index="k", columns="radius_m", values="mean_cands").round(1).to_string())


if __name__ == "__main__":
    main()
