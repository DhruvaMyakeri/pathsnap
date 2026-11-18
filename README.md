# PathSnap: map-matching noisy GPS to a road network

PathSnap reconstructs the route a vehicle actually drove from noisy ~1 Hz GPS
fixes. It works in three steps:

1. Find the road segments ("arcs") near every GPS point.
2. Score each (point, candidate arc) pair with a machine-learning classifier.
3. Decode the most likely connected route with a Hidden Markov Model and the Viterbi algorithm.

A FastAPI service on Laptop 1 runs the pipeline. A Streamlit UI on Laptop 2
lets the evaluator pick one of the reserved trips and shows the predicted
route and its accuracy.

> Course: UE24CS352A Machine Learning, PES University (PathSnap hackathon).

---

## Repository layout

```
pathsnap/
├── src/pathsnap/          core library (imported by scripts, API and notebook)
│   ├── config.py          paths (overridable by env vars) + pipeline constants
│   ├── data.py            reading .track/.nodes/.arcs (prediction) and .route (labels only)
│   ├── geo.py             per-trip UTM projection, bearings
│   ├── network.py         road graph: geometry, R-tree, twins, adjacency
│   ├── routing.py         bounded Dijkstra with caching, route distances
│   ├── candidates.py      candidate arcs per GPS point
│   ├── labels.py          monotone DP alignment of the reference route to GPS points
│   ├── features.py        per-(point, arc) features
│   ├── models.py          the classifiers compared (one per algorithm family)
│   ├── experiment.py      shared CV protocol helpers
│   ├── hmm.py             HMM + Viterbi decoder
│   ├── metrics.py         Top-1, miss rate, continuity, route P/R/F1, ...
│   ├── pipeline.py        end-to-end predict_trip() + server-side score_trip()
│   └── viz.py             Folium maps
├── scripts/               reproducible experiment steps (see "Reproduce")
├── eda/                   exploratory data analysis (script + executed notebook)
├── api/                   FastAPI backend (Laptop 1)
├── ui/                    Streamlit evaluator UI (Laptop 2)
├── tests/                 pytest suite
├── artifacts/             deployed model + HMM configuration
├── results/               every CSV / JSON / figure / map behind the report
└── docs/DEPLOYMENT.md     two-laptop runbook
```

## Data

The raw dataset is **not** committed. Put it next to the repository (default)
or point `PATHSNAP_DATA` at it:

```
ML-HACKATHON/
├── pathsnap/                      <- this repository
└── Student folder/Student folder/
    ├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          (80 trips)
    └── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   (10 trips)
```

The evaluator-testing trips are never used in development. On demo day they are
served by the API from the folder in `PATHSNAP_EVAL_DIR` (see `docs/DEPLOYMENT.md`).

## Setup (Windows, Python 3.12)

```bat
cd pathsnap
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

The UI laptop only needs `pip install -r ui/requirements.txt`.

## Reproduce

Run these steps in order from the repository root, with the venv active.

| Step | Command | Output |
|---|---|---|
| 1. EDA | `python eda/run_eda.py` | `results/eda/`, `results/maps/eda_*.html`, `eda/EDA.ipynb` |
| 2. Candidate parameters (train only) | `python scripts/candidate_params.py` | `results/dataset/candidate_grid.csv` |
| 3. Candidate / feature / label tables | `python scripts/build_dataset.py` | `data_cache/*.parquet`, `results/dataset/` |
| 4. Model comparison (5-fold group CV + student test) | `python scripts/train_models.py --models baseline_nearest_arc baseline_distance_heading logistic_regression gaussian_nb knn random_forest lightgbm mlp` then `--summarize` | `results/models/` |
| 5. Bayesian optimisation | `python scripts/tune_optuna.py --model lightgbm --trials 40` | `results/optuna/` |
| 6. Tuned model through the same protocol | `python scripts/train_models.py --models lightgbm --params results/optuna/best_params_lightgbm.json --name lightgbm_tuned` | `results/models/runs/lightgbm_tuned.json` |
| 7. HMM tuning on train OOF probabilities (both candidates; the selected one is run last because it writes `artifacts/hmm_config.json`) | `python scripts/tune_hmm.py --model lightgbm_tuned`, then `python scripts/tune_hmm.py --model lightgbm` | `results/hmm/`, `artifacts/hmm_config.json` |
| 8. Export the deployed model (selected on train-OOF HMM accuracy) | `python scripts/export_model.py --run lightgbm` | `artifacts/model.joblib` |
| 9. Classifier-only vs HMM on student test | `python scripts/eval_sequences.py` | `results/hmm/sequence_eval*.csv` |
| 10. Final end-to-end evaluation + Folium maps | `python scripts/evaluate_final.py` | `results/final/`, `results/maps/final_*.html` |
| 11. Report figures and tables | `python scripts/make_report_figures.py` | `results/figures/`, `results/tables/` |
| 12. Training (in-sample) vs CV vs test accuracy | `python scripts/train_vs_test.py` | `results/models/train_vs_test.csv` |
| Tests | `pytest` | |

Seeds are fixed (`config.SEED = 42`). Every run logs its configuration, timings
and metrics as JSON (`results/models/runs/*.json`, `results/optuna/*`).

## Results (student-test partition, 10 trips, 19,133 labelled GPS fixes)

| Stage | Pointwise accuracy | Route continuity | Route F1 |
|---|---|---|---|
| Nearest arc (no ML) | 0.535 | – | – |
| Distance + heading rule (no ML) | 0.854 | 0.952 | 0.890 |
| LightGBM, classifier only | 0.903 | 0.976 | 0.950 |
| **LightGBM + HMM/Viterbi (deployed)** | **0.934** | **0.9999** | **0.990** |

- Six model families were compared with 5-fold trip-grouped CV (`results/models/model_comparison.csv`).
- LightGBM had the best CV Top-1 (0.899 ± 0.024).
- Bayesian optimisation (Optuna TPE, 40 trials) found a flat optimum: no Top-1 gain, better log loss.
- The deployed configuration was chosen on training-data evidence (`report/REPORT_NOTES.md` §4).
- End-to-end inference takes 0.28 s per trip on average.

All numbers, figures and maps are in `results/`, and `report/REPORT_NOTES.md` maps them to the report sections.

## Run the system (two laptops)

Laptop 1 (API):

```bat
scripts\run_api.bat
```

Laptop 2 (UI):

```bat
set PATHSNAP_API_URL=http://<laptop-1-ip>:8000
scripts\run_ui.bat
```

See `docs/DEPLOYMENT.md` for the full runbook (network, firewall, demo-day steps).
