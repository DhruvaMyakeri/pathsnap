"""Central configuration: paths, partitions and pipeline constants.

Paths can be overridden with environment variables so the same code runs on
both laptops:

* ``PATHSNAP_DATA``     folder that contains the ``train/`` and ``student_test/`` partitions
* ``PATHSNAP_EVAL_DIR`` folder of trips the API serves (used by ``api/``)
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PATHSNAP_DATA", REPO_ROOT.parent / "Student folder" / "Student folder"))
PARTITIONS = {
    "train": DATA_ROOT / "train",
    "student_test": DATA_ROOT / "student_test",
}

RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CACHE_DIR = REPO_ROOT / "data_cache"

SEED = 42

# --- Candidate generation -------------------------------------------------
# Arcs whose segment lies within CAND_RADIUS_M of a GPS point become candidates.
# If a point has none, the radius doubles up to CAND_MAX_RADIUS_M.
# Only the CAND_MAX_K closest arcs are kept per point.
# Chosen on the TRAINING partition only (scripts/candidate_params.py ->
# results/dataset/candidate_grid.csv). With R = 50 m (= the label threshold
# below), the candidate miss rate over labelled points is 0.47 % for K = 16 and
# 0.02 % for K = 32, while the mean number of candidates only grows from 7.3 to
# 8.6 per point (most points have few arcs nearby), so K = 32 is kept.
CAND_RADIUS_M = 50.0
CAND_MAX_RADIUS_M = 200.0
CAND_MAX_K = 32

# --- Label alignment ------------------------------------------------------
# Per-point cost is capped so GPS glitches cannot drag the alignment around.
LABEL_COST_CAP_M = 100.0
# Points farther than this from their aligned reference arc get no label
# (excluded from training and scoring instead of receiving a wrong label).
# The train distribution is bimodal: 75 % of points lie within 11 m of the
# reference route, ~10 % lie 90-500 m away (stretches where the GPS track and
# the reference route disagree, mostly parked off-network as in trip 00000054),
# and only 2.2 % fall between 30 and 75 m, so 50 m sits in the valley between
# the two modes.
LABEL_MAX_DIST_M = 50.0
# Direction penalty in the alignment cost (see labels.align_to_route): aligning a
# moving fix to an arc pointing the opposite way costs this many metres extra.
LABEL_HEADING_PENALTY_M = 20.0
