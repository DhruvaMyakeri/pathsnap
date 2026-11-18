# Feature-engineering experiment: when should the heading feature be trusted?

**Question.** EDA found stationary GPS "jitter clouds": receivers parked off the
network that move 7–8 m per fix without going anywhere (trips 00000054 and
00000033). Their heading is random. Should `heading_diff` be set to the
neutral 90° unless the motion looks reliable?

**Variants.** The candidates and labels were identical in both runs; only the
heading gate in `features.py` differed.

| Variant | Rule for a usable heading (±2-fix window) |
|---|---|
| A, kept | net displacement ≥ 1 m. Path straightness is added as a feature so the model can learn how far to trust heading. |
| B, rejected | net displacement ≥ 3 m **and** straightness (net displacement / path length) ≥ 0.6 |

**Evidence (training partition, 5 trip-grouped CV folds; same folds for both variants)**

| Metric | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|
| Distance+heading rule, Top-1, variant A | 0.8404 | 0.8257 | 0.8235 | 0.8939 | 0.8681 | 0.8503 |
| Distance+heading rule, Top-1, variant B | 0.8114 | 0.7640 | 0.7642 | 0.7788 | 0.8438 | 0.7924 |
| LightGBM, Top-1, variant B (run stopped after 2 folds) | 0.8515 | 0.8000 | – | – | – | – |

The LightGBM numbers for variant A are in `results/models/runs/lightgbm_folds.csv`.

**Why B lost.** Diagnosed on the student_test labelled fixes, the strict gate
neutralised the heading of 22.4 % of them. Their
median net displacement over the 4 s window was 10.3 m (interquartile range
6.4–16.1 m): this is slow driving, not standing still. At 1 Hz, normal GPS
noise makes the travelled path look wiggly, so straightness drops below 0.6
even though the direction of the net displacement is correct.

**Decision.** Variant A for the classifier features. The strict test is kept
only for the *label alignment* direction penalty (`labels.align_to_route`).
There, leaving a tie unresolved is safer than creating a wrong label.
