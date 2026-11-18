# Report fact sheet (for writing the technical report)

These are **facts and pointers, not report prose**. Every number below was
produced by code in this repository; the file named next to it is its source.
Write the report text in your own words and disclose AI assistance as the event
policy requires (Deliverables §8). Every team member should be able to re-run
and explain each step (see README → Reproduce).

Partitions: **train** = 80 trips (fitting, CV, tuning); **student_test** = 10
trips (development comparison only); the evaluator trips are never used.

---

## 2.2 Dataset understanding and EDA

Sources: `results/eda/*`, `eda/EDA.ipynb`, `results/eda/data_quality.md`, `results/dataset/*`.

**Files and records**
- Each trip has 4 headerless whitespace files: `.track` (lon, lat, t), `.nodes` (lon, lat), `.arcs` (from, to; directed) and `.route` (ordered arc ids).
- **Every trip carries its own road network.** Node and arc ids are local to the trip.
- train has 208,417 fixes (361 / 1,529 / 19,313 per trip, min / median / max), 126.2 h and 2,981 km. student_test has 19,160 fixes, 5.6 h and 297 km. (`key_numbers.json`)

**Missing and invalid values**
- 0 missing values in any file.
- 0 invalid, duplicate or out-of-order fixes, so `clean_track` drops nothing on either partition. (`missing_values.csv`, fig02)

**Sampling and gaps**
- 1 s steps make up 96.5 % of train steps and 98.6 % of student_test steps.
- train has 590 gaps > 10 s in 40 trips and 69 gaps > 60 s in 22 trips. The longest gap is a 42.5 h pause in trip 00000090, not a jump. (fig03)

**Geography**
- 25 UTM zones on 5 continents, with southern-hemisphere trips 71, 80 and 98. No single projected CRS fits all trips.
- Decision: project each trip to the UTM zone of its median fix (`geo.py`). (fig05, `results/maps/eda_world_overview.html`)

**Road network**
- 92.2 % of arcs (train) have a reverse twin with identical geometry: two-way roads stored once per direction.
- The largest strongly connected component holds a median 96.4 % of nodes (minimum 63.6 %, in trip 5).
- There are 106 zero-length arcs, 24 self-loops and 1,903 duplicate arcs. (`network_stats.csv`, fig08)

**Reference routes**
- 13 broken links in 10 train trips and 1 in student_test (trip 76): consecutive route arcs that do not connect. (`broken_links.csv`)
- 46 U-turns in train routes. (fig09)

**GPS noise relative to the reference route (train)**
- p50 = 5.7 m, p75 = 10.9 m, p90 = 89.8 m: **bimodal**. (fig10)
- 11.1 % of fixes lie more than 50 m away. They are concentrated in trips 54 (89 % of its fixes) and 33.
- They are **stationary jitter clouds**: the vehicle is parked off the network and the fix wanders 7–8 m per second. (`results/eda/trip_00000054_projected.png`)

**Preprocessing and decisions**
- Labels: a monotone DP aligns the route to the fixes (`labels.py`). Fixes more than 50 m from their aligned arc are unlabelled. 50 m sits in the valley of the bimodal distance distribution.
  - labelled: 185,254 / 208,417 train fixes (88.9 %) and 19,133 / 19,160 student_test fixes (99.9 %) (`results/dataset/dataset_summary.json`)
- Candidates: arcs within R = 50 m, capped at K = 32 per point.
  - chosen on train only: miss rate over labelled fixes is 0.47 % at K = 16 and 0.02 % at K = 32, with 7.3 vs 8.6 candidates per point (`results/dataset/candidate_grid.csv`, `candidate_miss_rate.png`)
  - final candidate miss rate: 0.018 % on train and 0 % on student_test
- Label fix found by a unit test: on out-and-back trips the twin arcs tie in the DP, so a direction penalty was added. It changes 0.99 % of train labels, 84 % of them direction corrections (`tests/test_core.py`, `labels.py` docstring).

**Features**: 23 per (fix, candidate arc), listed in `features.py` with the reason for each.
- Distance group: distance, rank, gap to the nearest candidate, ratio, number of candidates, number of close candidates.
- Direction group:
  - heading difference at ±2 s and ±5 s windows, plus its cosine
  - speed, displacement, path straightness at both windows
  - sampling gaps
- Arc and network group: arc length, projection fraction, overshoot beyond the segment end, two-way flag, node degrees.
- Context group: mean and max distance of the 4 neighbouring fixes to the same arc.
- Why direction matters:
  - Nearest arc alone gives only 0.564 CV Top-1, because twins have identical distance.
  - Adding heading lifts it to 0.850.
- Feature ablation: a strict "heading reliability" gate lowered train-CV Top-1 and was rejected (`results/experiments/heading_gate_ablation.md`).

---

## 3.1 Model comparison table

Ready-made table: `results/tables/model_comparison_report.md` / `.csv`.

| Model | CV Top-1 (mean ± sd) | CV macro F1 | CV log loss | Test Top-1 | Test macro F1 | Test log loss | Route continuity (clf only, test) | Fit s | Infer s* |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM (selected) | 0.8992 ± 0.0239 | 0.9396 ± 0.0141 | 0.0590 | 0.9033 | 0.9428 | 0.0472 | 0.9756 | 37.5 | 0.76 |
| Random forest | 0.8942 ± 0.0234 | 0.9390 ± 0.0138 | 0.0577 | 0.8972 | 0.9414 | 0.0491 | 0.9691 | 82.5 | 0.42 |
| MLP | 0.8940 ± 0.0227 | 0.9353 ± 0.0137 | 0.0641 | 0.8908 | 0.9357 | 0.0541 | 0.9712 | 19.0 | 0.06 |
| Logistic regression | 0.8904 ± 0.0232 | 0.9250 ± 0.0142 | 0.0832 | 0.8899 | 0.9232 | 0.0697 | 0.9731 | 5.3 | 0.02 |
| KNN | 0.8699 ± 0.0217 | 0.9140 ± 0.0151 | 0.0940 | 0.8698 | 0.9166 | 0.0759 | 0.9669 | 0.3 | 43.17 |
| Gaussian NB | 0.8658 ± 0.0152 | 0.8516 ± 0.0060 | 0.5486 | 0.8611 | 0.8370 | 0.5814 | 0.9601 | 1.1 | 0.07 |
| Rule: distance + heading | 0.8503 ± 0.0302 | 0.9161 | N/A (hard rule) | 0.8539 | 0.9192 | N/A | 0.9524 | 0 | 0.05 |
| Rule: nearest arc | 0.5637 ± 0.0539 | 0.7561 | N/A (hard rule) | 0.5352 | 0.7432 | N/A | – | 0 | 0.00 |

\*Inference over all 201,662 student-test candidate rows.

Sources: `results/models/model_comparison.csv`, `runs/*.json`, `results/hmm/sequence_eval.csv`.

**Primary selection metric**: observation-level Top-1 accuracy, mean over 5 trip-grouped CV folds on train.
- Why: the system outputs one arc per GPS fix.
- Macro F1 depends on the 0.5 threshold and ignores ranking within a fix's candidate set.
- Log loss is the secondary metric because the HMM consumes the probabilities.
- LightGBM leads on CV Top-1, CV macro F1 and test log loss. Random forest is close, with slightly better CV log loss, but takes 2.2× longer to train.

## 3.2 Per-model facts

Training vs test Top-1 (`results/models/train_vs_test.csv`, from `scripts/train_vs_test.py`). In-sample means the final model scored on the rows it was fitted on:

| Model | In-sample | CV | Student test |
|---|---|---|---|
| LightGBM | 0.960 | 0.899 | 0.903 |
| Random forest | 0.929 | 0.894 | 0.897 |
| MLP | 0.909 | 0.894 | 0.891 |
| Logistic regression | 0.894 | 0.890 | 0.890 |
| KNN | 1.000* | 0.870 | 0.870 |
| Gaussian NB | 0.872 | 0.866 | 0.861 |

\*KNN is 1.000 in-sample by construction: with distance weighting, every stored row is its own nearest neighbour.

The tree models memorise some of the training data (LightGBM has a 5.7-point in-sample gap) but generalise stably (CV ≈ test). The final technical report draft is in the Claude doc "PathSnap Technical Report".

Training data and protocol, shared by all models:
- Train only on labelled rows.
- Random forest uses every 2nd fix, KNN every 12th and the MLP every 4th. Consecutive 1 Hz fixes are near-duplicates.
- Every model is evaluated on all fixes.
- Seed 42.
- Hyperparameters: `src/pathsnap/models.py` (`DEFAULT_PARAMS`). Per-fold metrics: `results/models/cv_folds.csv`.

| Model | Objective / loss | Key hyperparameters (initial) | Over-/under-fitting evidence |
|---|---|---|---|
| Logistic regression | L2-regularised binary cross-entropy (lbfgs), standardised inputs | C = 1.0 | CV 0.890 ≈ test 0.890: no overfit. Linear boundary on non-linear interactions (heading × straightness) underfits, giving the worst log loss among the strong models. |
| Gaussian NB | Maximum-likelihood Gaussian class densities (no iterative loss) | var_smoothing 1e-9 | Independence assumption violated (dist, rank and ratio are correlated). Recall 0.94 but precision 0.58, and log loss 0.58: badly calibrated. |
| KNN | None (lazy), distance-weighted vote | k = 50, standardised, every 12th fix | CV ≈ test. Slowest inference (43 s), because the cost grows with training size. |
| Random forest | Gini impurity (bagging) | 200 trees, min_samples_leaf 20, max_features sqrt | CV 0.894 vs test 0.897. Leaf size limits overfitting. |
| LightGBM | Binary log loss (gradient boosting) | 600 trees, lr 0.05, 63 leaves, min_child 50, subsample / colsample 0.8, λ2 1 | Train and validation loss curves converge with no divergence (`results/figures/loss_curves.png`). CV 0.899 vs test 0.903. |
| MLP | Binary cross-entropy + L2 (Adam) | (64, 32), α 1e-4, batch 1024, lr 1e-3, early stopping patience 5 | Stopped after 15 epochs, best at epoch 13. Validation loss sits below training loss because the 10 % internal validation trips are easier. |

## 3.3 Figures (all generated from results)

All in `results/figures/`:
- `cv_top1.png`, `cv_macro_f1.png`
- `student_test_top1_macro_f1.png`, `log_loss.png`
- `cv_fold_stability.png`, `confusion_matrices.png`
- `loss_curves.png` (MLP and LightGBM only, since they have epoch or iteration histories)
- `inference_time.png`, `training_time.png`
- the HMM figures (§5)

## 4. Bayesian hyperparameter optimisation (LightGBM)

Sources: `scripts/tune_optuna.py`, `results/optuna/*`, `results/tables/lightgbm_tuned_vs_untuned.csv`.
- Sampler: Optuna TPE (multivariate, 8 random start-up trials) with a MedianPruner.
- Objective: mean Top-1 over 3 trip-grouped folds of train, fitting on every 2nd fix.
- Search space (log scale where marked):
  - n_estimators 150–1200 (log), learning_rate 0.01–0.2 (log)
  - num_leaves 15–255 (log), min_child_samples 10–400 (log)
  - subsample 0.5–1, colsample_bytree 0.4–1
  - reg_lambda 1e-3–30 (log), reg_alpha 1e-3–10 (log)
- Search: 40 trials (22 complete, 18 pruned) in 17.7 min.
- Best trial (#27):

| Parameter | Best value |
|---|---|
| n_estimators | 875 |
| learning_rate | 0.0236 |
| num_leaves | 60 |
| min_child_samples | 24 |
| subsample | 0.857 |
| colsample_bytree | 0.403 |
| reg_lambda | 0.478 |
| reg_alpha | 0.0132 |

- Search value 0.8969. The whole history spans only 0.892–0.897, so the surface is flat (`lightgbm_history.png`, `lightgbm_importance.png`).
- **Tuned vs untuned**, run through the same 5-fold protocol and the same partitions:

| Metric | Untuned | Tuned |
|---|---|---|
| CV Top-1 | 0.8992 | 0.8986 |
| CV macro F1 | 0.9396 | 0.9401 |
| CV log loss | 0.0590 | 0.0571 |
| Test Top-1 | 0.9033 | 0.9004 |
| Test macro F1 | 0.9428 | 0.9427 |
| Test log loss | 0.0472 | 0.0474 |
| Inference (s) | 0.76 | 1.34 |

- Tuning did **not** improve Top-1 (all differences are well within the fold SD of 0.024). It improved CV log loss.
- **Deployment choice, made on training data only**: with HMM decoding of out-of-fold train predictions, untuned + HMM scores 0.9320 against tuned + HMM 0.9303 (`results/hmm/hmm_config_*.json`). The untuned configuration is deployed. The student test agrees (0.9335 vs 0.9337, a tie).

## 5. HMM and Viterbi

Design rationale: `src/pathsnap/hmm.py` docstring.
- **Hidden states**: each fix's top-K candidate arcs by classifier probability, with K = 12.
- **Emission**: log(classifier probability), a scaled likelihood as in hybrid NN/HMM decoders.
- **Transition**: −|network route distance − GPS straight-line distance| / β (Newson & Krumm 2009), with β = 5 m.
  - Infeasible if no path exists within the search limit or if the implied speed is above 70 m/s.
  - Backward moves on the same arc are allowed up to 15 m, as jitter.
- **GPS gaps**: β and the Dijkstra limit scale with dt / 10 s.
- **Outliers**: up to 3 consecutive infeasible fixes are bridged and then filled from neighbours; otherwise the chain restarts (a break).
- **Fixes without candidates** take the neighbouring decoded arc.
- **Route assembly**: consecutive decoded arcs are joined with shortest paths.
- **Tuning**: only emission_weight × β matters, so the weight is fixed at 1. A grid over β and K on 30 train trips, using OOF probabilities, gave β = 5 and K = 12. Accuracy is flat for β between 3 and 20 (`results/hmm/hmm_tuning_lightgbm.csv`).
- **Results** (student_test, 19,133 labelled fixes, `results/hmm/sequence_eval.csv`, `results/figures/hmm_*.png`):

| | Accuracy | Direction-agnostic | Continuity | Route F1 |
|---|---|---|---|---|
| LightGBM, classifier only | 0.9033 | 0.9458 | 0.9756 | 0.9502 |
| LightGBM + HMM | **0.9335** | 0.9633 | **0.9999** | **0.9895** |

  - The HMM improves every model by 3–5 points. Even Gaussian NB rises from 0.861 to 0.909.
  - Decoding takes 0.11 s per trip on average; the largest train trip (19k fixes) takes about 2 s.
- **Where the HMM did not help**:
  - Trip 00000077: accuracy 0.760 → 0.793 only.
    - 484 of its fixes are on the right road but in the wrong direction, which is 85 % of all direction errors on student_test.
    - At those fixes the GPS heading is about 90° from both directions (median 100° vs the label, 80° vs the prediction), so direction carries no information.
    - Half of the involved arcs appear in **both** directions in its reference route (back-and-forth driving).
  - Remaining error mix, on all 19,133 labelled fixes:
    - direction errors 2.97 %
    - adjacent-arc boundary errors 2.0 % (labels are ambiguous at junctions)
    - other 1.7 %

## 6. Folium maps

- `results/maps/final_<trip>.html`: one per student_test trip, generated by `scripts/evaluate_final.py` with `src/pathsnap/viz.py`.
  - layers: GPS fixes, road network within 120 m, reference route (green), classifier-only route (orange, dashed), predicted route (red)
  - includes a legend and a layer control
- EDA maps: `results/maps/eda_*.html`.

## 7. API, UI and deployment

- Runbook: `docs/DEPLOYMENT.md`. Code: `api/main.py`, `ui/app.py`. Tests: `tests/test_api.py`.
- The API returns only predictions and 9 aggregate metrics (a whitelist). The reference route is read server-side, after prediction.
- Average end-to-end time on student_test: 0.28 s per trip, 0.55 s at most (`results/final/student_test_summary.json`).
- **Screenshots still to take yourselves**, on the real two-laptop setup:
  - the UI with a trip result
  - the CMD window on Laptop 1 showing the REQUEST / STAGES / RESULT / RESPONSE log lines
