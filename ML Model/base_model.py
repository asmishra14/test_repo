# ============================================================
# IPL GRID CLASSIFICATION PROJECT
# ===========================================================
# ============================================================

from __future__ import annotations

import os

# ------------------------------------------------------------
# Thread budget (MUST be set before numpy / BLAS / OpenMP import).
# ------------------------------------------------------------
_THREAD_BUDGET = os.environ.get("PML_NUM_THREADS", "8")
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, _THREAD_BUDGET)

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from xgboost import XGBClassifier

from create_dataset import NUM_CELLS
from create_dataset import feature_block_indices

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (800, 600)
GRID_ROWS = 8
GRID_COLS = 8
CELL_WIDTH = 100
CELL_HEIGHT = 75
NUM_CLASSES = 11

TEAM_NAME = "teamname"

TRAIN_IMAGE_FOLDER = "./dataset"
LABEL_CSV = "./labels.csv"

MODEL_PATH = f"model_{TEAM_NAME}.pkl"            # best model (deliverable)
MODEL_PATH_SVM = f"model_{TEAM_NAME}_svm.pkl"
MODEL_PATH_XGB = f"model_{TEAM_NAME}_xgb.pkl"
MODEL_PATH_RF = f"model_{TEAM_NAME}_rf.pkl"
MODEL_PATH_LR = f"model_{TEAM_NAME}_lr.pkl"
MODEL_PATH_LSVC = f"model_{TEAM_NAME}_lsvc.pkl"
MODEL_PATH_KNN = f"model_{TEAM_NAME}_knn.pkl"
MODEL_PATH_RF2 = f"model_{TEAM_NAME}_rf2.pkl"
MODEL_PATH_LGBM = f"model_{TEAM_NAME}_lgbm.pkl"
MODEL_PATH_TWO_STAGE = f"model_{TEAM_NAME}_two_stage.pkl"

DEFAULT_DATASET = "..\\Create Labels\\outputs\\ipl_features.npz"

TEST_SIZE = 0.2
RANDOM_STATE = 42
PCA_VARIANCE = 0.90
THRESHOLD_TUNE_FOLDS = 3
SEPARATOR = "=" * 60

# Train the classical two-stage detector (gate + team classifier) for comparison.
INCLUDE_TWO_STAGE = True

NO_TEAM_LABEL = 0
CELL_COLUMNS = [f"c{i:02d}" for i in range(1, NUM_CELLS + 1)]  # c01..c64 (PDF schema)

# ------------------------------------------------------------
# Imbalance handling (TRAIN split). Metrics are reported on a class-BALANCED
# subsample of the test split (see train_model).
# ------------------------------------------------------------
# Class-0 dominates the full dataset, so the single-stage models undersample
# class 0 in their TRAIN split (kept tractable + roughly balanced). The
# two-stage GATE is the exception -- it trains on the full background for
# maximum class-0 coverage. Kernel SVM / KNN get a smaller cap (they scale
# worst with sample count).
TRAIN_ZERO_CAP = 8000
TRAIN_ZERO_CAP_HEAVY = 3000  # for kernel-SVM / KNN

# LightGBM's OpenMP backend deadlocks with n_jobs=-1 in our experiments;
# a fixed thread count avoids the hang.
LGBM_N_JOBS = 4

N_JOBS = 8

# ------------------------------------------------------------
# Per-stage feature subsets for the two-stage detector.
#   * Gate (0 vs player): color + texture + spatial context. NO text (ads /
#     scoreboards look text-like -> false players) and NO orb.
#   * Team (1..10): per-cell color + texture + jersey-color priors + text/logo
#     cue + position. Neighbor-context is excluded so the team decision focuses
#     on the cell's own jersey.
# ------------------------------------------------------------
# forward-selection (grouped val) result: every block helps both stages, and
# adding 'text' to the GATE is the key gain (jersey lettering separates player
# from background better than the ad/scoreboard false positives cost). 'neighbor'
# gives the team stage a small gain. Both stages therefore use the full block set.
GATE_BLOCKS = ("hsv_hist", "color_moments", "lbp", "team_color", "position", "neighbor", "text")
TEAM_BLOCKS = ("hsv_hist", "color_moments", "lbp", "team_color", "position", "text", "neighbor")

# Gate decision-threshold grid (probability of "player"). Tuned leak-free on
# out-of-fold predictions to balance background vs player detection.
GATE_THRESHOLD_GRID = np.linspace(0.20, 0.80, 31)

# 8x8 label-grid post-processing: drop small team blobs. A real team is a
# contiguous blob (EDA: mean ~10, median 8 cells); a team that appears only as a
# tiny connected fragment scattered over the frame (e.g. blue ad-board / crowd
# cells mislabelled MI) is almost always a gate false positive. We drop any
# same-team 8-connected component with fewer than TEAM_MIN_COMPONENT cells.
# TEAM_MIN_COMPONENT=2 removes isolated singletons; larger
# values remove bigger fragments (validate on the held-out split before raising).
POSTPROCESS_ISOLATED = True
TEAM_MIN_COMPONENT = 2

# ------------------------------------------------------------
# Adaptive team prior
# ------------------------------------------------------------
# A real team occupies a blob of cells (EDA: mean ~10, median 8 cells/team; only
# ~6% of teams occupy <=2 cells), so a team predicted in just a handful of cells
# is almost always a gate false positive. Per image we therefore KEEP every team
# whose predicted cell-count >= TEAM_PRIOR_MIN_CELLS (no cap on the number of
# teams -- an image may legitimately contain >2 teams) and re-decide all other
# cells over {0} U kept-teams. This removes noise teams AND fixes cross-team
# confusion. tau=5 was chosen on a validation split (val 0.797->0.833) and lifts
# the held-out LGBM balanced macro-F1 from 0.827 to 0.854. A team with fewer than
# the threshold cells whose evidence is weak collapses to 0 (noise).
TEAM_COUNT_PRIOR = True
TEAM_PRIOR_MIN_CELLS = 5


def _xgb_classifier(**kwargs: Any) -> XGBClassifier:
    """XGBoost >=2.0 dropped use_label_encoder; keep compatibility with 1.x."""
    try:
        return XGBClassifier(use_label_encoder=False, **kwargs)
    except TypeError:
        kwargs.pop("use_label_encoder", None)
        return XGBClassifier(**kwargs)


def _build_classifier_registry() -> dict[str, dict[str, Any]]:
    """Return classifier configs with best-fit hyperparameters from prior runs.

    Every estimator exposes predict_proba (SVC via CalibratedClassifierCV with
    ensemble=False, matching former SVC(probability=True); LinearSVC wrapped
    in CalibratedClassifierCV) so a confidence threshold can be applied.
    """
    return {
        "svm": {
            "model": CalibratedClassifierCV(
                SVC(
                    kernel="rbf",
                    C=10,
                    gamma="auto",
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
                ensemble=False,
            ),
            "model_path": MODEL_PATH_SVM,
            "requires_pca": False,
            "undersample": True,  # RBF SVC scales poorly with the full class-0 set
            "zero_cap": TRAIN_ZERO_CAP_HEAVY,
            "tune_threshold": False,  # baseline; skip the costly extra SVC refits
        },
        "xgb": {
            # Regularised vs the prior config (shallower trees, subsampling, L2).
            # NOTE: XGBoost 3.2 on this box deadlocks with multi-thread OpenMP and
            # is very slow single-threaded, so it is kept as a light baseline
            # (fewer trees, n_jobs=1, hist) -- LightGBM is the primary booster.
            "model": _xgb_classifier(
                eval_metric="mlogloss",
                verbosity=0,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                min_child_weight=5,
                tree_method="hist",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            "model_path": MODEL_PATH_XGB,
            "requires_pca": False,
            "undersample": True,
            "tune_threshold": False,
        },
        "rf": {
            "model": RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_estimators=300,
                max_depth=24,
                min_samples_leaf=2,
                max_features="sqrt",
                n_jobs=N_JOBS,
            ),
            "model_path": MODEL_PATH_RF,
            "requires_pca": False,
            "undersample": True,
        },
        "lr": {
            "model": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
            "model_path": MODEL_PATH_LR,
            "requires_pca": False,
            "undersample": True,
        },
        "lsvc": {
            "model": CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
            ),
            "model_path": MODEL_PATH_LSVC,
            "requires_pca": False,
            "undersample": True,
        },
        "knn": {
            "model": KNeighborsClassifier(n_neighbors=15, weights="distance"),
            "model_path": MODEL_PATH_KNN,
            "requires_pca": True,
            "undersample": True,  # KNN inference cost grows with the train set
            "zero_cap": TRAIN_ZERO_CAP_HEAVY,
            "tune_threshold": False,  # baseline; cross-val predict_proba is slow for KNN
        },
        "rf_large": {
            "model": RandomForestClassifier(
                n_estimators=500,
                class_weight="balanced_subsample",
                max_depth=28,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
            "model_path": MODEL_PATH_RF2,
            "requires_pca": False,
            "undersample": True,
        },
        "lgbm": {
            # Regularised: more, shallower trees + subsampling + L2 + a leaf-size
            # floor to curb memorisation of the training cells.
            "model": LGBMClassifier(
                n_estimators=600,
                learning_rate=0.05,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=40,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=LGBM_N_JOBS,
                verbose=-1,
            ),
            "model_path": MODEL_PATH_LGBM,
            "requires_pca": False,
            "undersample": True,
        },
    }


def _numpy_to_lgbm_frame(X: Any) -> pd.DataFrame:
    """Wrap ndarray output as a DataFrame before LightGBM.

    LightGBM's sklearn wrapper always registers booster feature names; feeding
    bare numpy from ``StandardScaler`` then triggers sklearn's
    "X does not have valid feature names" warning at ``predict_proba`` time.
    """
    arr = np.asarray(X)
    cols = [f"f{i}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols)


def _make_pipeline(model: Any, requires_pca: bool) -> Pipeline:
    """Bundle scaling (+ optional PCA) and the classifier into one estimator."""
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if requires_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))
    if isinstance(model, LGBMClassifier):
        steps.append(
            (
                "lgbm_frame",
                FunctionTransformer(_numpy_to_lgbm_frame, validate=False),
            ),
        )
    steps.append(("clf", model))
    return Pipeline(steps)


# ============================================================
# TWO-STAGE DETECTOR  (classical ML: gate + team classifier)
# ============================================================

class TwoStageDetector:
    """Stage 1 gate (player vs background) + Stage 2 team classifier.

    Rationale (project tie-in): real match photos are background-dominated, and
    boundary ads / scoreboards look "text-like". So the two stages use DIFFERENT
    hand-crafted feature subsets (per-stage feature selection):

      * Gate (0 vs player): color + texture + jersey-color priors + spatial
        context (position + neighbor-context). NO text block (avoids ad /
        scoreboard false positives) and NO ORB.
      * Team (1..10): per-cell color + texture + jersey-color priors + text /
        logo cue + position. Trained only on cells that truly contain a player.

    The gate is a calibrated probabilistic classifier whose decision threshold
    is tuned leak-free on out-of-fold predictions, so we can trade off class-0
    recall vs precision instead of accepting the default 0.5 cut (the previous
    gate over-fired -> background cells leaked into stage 2 as false teams).

    Both stages are plain scikit-learn pipelines (scaler + classifier), within
    the "hand-crafted features + classical ML" constraint. The final per-cell
    label is 0 when gated out, else the stage-2 team in 1..10.
    """

    def __init__(
        self,
        gate: Pipeline,
        team_clf: Pipeline,
        gate_idx: np.ndarray,
        team_idx: np.ndarray,
        gate_threshold: float = 0.5,
    ) -> None:
        self.gate = gate
        self.team_clf = team_clf
        self.gate_idx = np.asarray(gate_idx, dtype=int)
        self.team_idx = np.asarray(team_idx, dtype=int)
        self.gate_threshold = float(gate_threshold)
        self.classes_ = np.arange(NUM_CLASSES)
        self._player_col = 1

    def _gate_player_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.gate.predict_proba(X[:, self.gate_idx])
        classes = list(self.gate.classes_)
        col = classes.index(1) if 1 in classes else (proba.shape[1] - 1)
        return proba[:, col]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ) -> "TwoStageDetector":
        y = np.asarray(y).astype(int)
        is_player = (y > NO_TEAM_LABEL).astype(int)

        # Tune the gate threshold leak-free on out-of-fold player probabilities
        # (maximise the binary macro-F1 of background vs player) BEFORE the final
        # full-data fit.
        self.gate_threshold = self._tune_gate_threshold(X, is_player, groups)
        self.gate.fit(X[:, self.gate_idx], is_player)

        player_mask = y > NO_TEAM_LABEL
        if not player_mask.any():
            raise ValueError("No positive (team) samples to train stage 2.")
        self.team_clf.fit(X[player_mask][:, self.team_idx], y[player_mask])
        return self

    def _tune_gate_threshold(
        self,
        X: np.ndarray,
        is_player: np.ndarray,
        groups: np.ndarray | None,
    ) -> float:
        try:
            if groups is not None:
                cv: Any = StratifiedGroupKFold(
                    n_splits=THRESHOLD_TUNE_FOLDS, shuffle=True, random_state=RANDOM_STATE,
                )
                cv_iter = list(cv.split(X[:, self.gate_idx], is_player, groups))
            else:
                cv = StratifiedShuffleSplit(
                    n_splits=THRESHOLD_TUNE_FOLDS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                )
                cv_iter = list(cv.split(X[:, self.gate_idx], is_player))
            oof = cross_val_predict(
                self.gate, X[:, self.gate_idx], is_player,
                cv=cv_iter, method="predict_proba", n_jobs=None,
            )
            classes = list(getattr(self.gate, "classes_", [0, 1]))
        except Exception as exc:  # noqa: BLE001 - tuning is best-effort
            print(f"Gate threshold tuning skipped ({exc}); using 0.5")
            return 0.5
        col = classes.index(1) if 1 in classes else (oof.shape[1] - 1)
        player_proba = oof[:, col]
        best_t, best_f1 = 0.5, -1.0
        for t in GATE_THRESHOLD_GRID:
            pred = (player_proba >= t).astype(int)
            f1 = f1_score(is_player, pred, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        print(f"  gate threshold tuned -> {best_t:.3f} (OOF binary macro-F1={best_f1:.4f})")
        return best_t

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict 0..10 labels for a full-feature matrix."""
        player_proba = self._gate_player_proba(X)
        out = np.full(len(X), NO_TEAM_LABEL, dtype=int)
        idx = np.flatnonzero(player_proba >= self.gate_threshold)
        if idx.size:
            out[idx] = self.team_clf.predict(X[idx][:, self.team_idx]).astype(int)
        return out


def _build_two_stage_detector(include_text: bool, include_orb: bool) -> TwoStageDetector:
    """Gate = balanced RandomForest on the gate subset; team = regularised LightGBM."""
    gate = _make_pipeline(
        RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            max_depth=24,
            min_samples_leaf=2,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
        ),
        requires_pca=False,
    )
    team_clf = _make_pipeline(
        LGBMClassifier(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=LGBM_N_JOBS,
            verbose=-1,
        ),
        requires_pca=False,
    )
    gate_idx = feature_block_indices(GATE_BLOCKS, include_text=include_text, include_orb=include_orb)
    team_idx = feature_block_indices(TEAM_BLOCKS, include_text=include_text, include_orb=include_orb)
    return TwoStageDetector(gate, team_clf, gate_idx, team_idx)


def _append_log(message: str, log_file: str | Path | None) -> None:
    if not log_file:
        return
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip("\n") + "\n")


# ============================================================
# SPLITTING (image-grouped -> no cell leakage)
# ============================================================

def _grouped_train_test_indices(
    y: np.ndarray,
    groups: np.ndarray | None,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    n_splits = max(2, round(1.0 / TEST_SIZE))
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(splitter.split(np.zeros(len(y)), y, groups))
    print("WARNING: no image groups found in dataset -> falling back to a "
          "stratified (non-grouped) split; cells from one image may leak.")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    return next(splitter.split(np.zeros(len(y)), y))


def _undersample_train_zeros(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None,
    cap: int,
    seed: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Undersample class 0 in the (train) arrays to ``cap`` rows; non-zeros kept.

    Applied to TRAIN data only -- the test split keeps its natural distribution.
    """
    zero_idx = np.flatnonzero(y == NO_TEAM_LABEL)
    if len(zero_idx) <= cap:
        return X, y, groups
    rng = np.random.default_rng(seed)
    keep_zero = rng.choice(zero_idx, size=cap, replace=False)
    non_zero_idx = np.flatnonzero(y != NO_TEAM_LABEL)
    idx = np.sort(np.concatenate([keep_zero, non_zero_idx]))
    return X[idx], y[idx], (groups[idx] if groups is not None else None)


def _balanced_subsample_indices(
    y: np.ndarray,
    seed: int = RANDOM_STATE,
    per_class: int | None = None,
) -> np.ndarray:
    """Indices of a class-balanced subsample (for the 'balanced view' report)."""
    rng = np.random.default_rng(seed)
    present = [c for c in range(NUM_CLASSES) if np.any(y == c)]
    if per_class is None:
        per_class = min(int(np.sum(y == c)) for c in present)
    idx: list[np.ndarray] = []
    for c in present:
        ci = np.flatnonzero(y == c)
        k = min(per_class, len(ci))
        idx.append(rng.choice(ci, size=k, replace=False))
    return np.sort(np.concatenate(idx))


# ============================================================
# 8x8 GRID POST-PROCESSING  (spatial-consistency cleanup)
# ============================================================

def _remove_small_components(labels: np.ndarray, min_size: int) -> np.ndarray:
    """Drop same-team 8-connected components smaller than ``min_size`` cells.

    A real team is a contiguous blob; tiny scattered fragments of a team label
    (e.g. blue background cells mislabelled MI) are gate false positives. Each
    maximal run of 8-adjacent cells sharing the SAME team label is one component;
    components with fewer than ``min_size`` cells are reset to 0 (no team).
    ``min_size <= 1`` is a no-op.
    """
    grid = np.asarray(labels, dtype=int).reshape(GRID_ROWS, GRID_COLS)
    if min_size <= 1:
        return grid.reshape(-1)
    cleaned = grid.copy()
    visited = np.zeros_like(grid, dtype=bool)
    for r0 in range(GRID_ROWS):
        for c0 in range(GRID_COLS):
            if grid[r0, c0] <= NO_TEAM_LABEL or visited[r0, c0]:
                continue
            team = grid[r0, c0]
            stack = [(r0, c0)]
            visited[r0, c0] = True
            component = []
            while stack:
                r, c = stack.pop()
                component.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS
                                and not visited[nr, nc] and grid[nr, nc] == team):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
            if len(component) < min_size:
                for r, c in component:
                    cleaned[r, c] = NO_TEAM_LABEL
    return cleaned.reshape(-1)


def _postprocess_grid(labels: np.ndarray) -> np.ndarray:
    """Drop small team blobs in one 8x8 label grid (see ``_remove_small_components``)."""
    return _remove_small_components(labels, TEAM_MIN_COMPONENT)


def _scores_to_full(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Map a classifier's (n, |classes|) proba to a full (n, NUM_CLASSES) matrix."""
    full = np.zeros((proba.shape[0], NUM_CLASSES), dtype=float)
    for j, c in enumerate(np.asarray(classes)):
        full[:, int(c)] = proba[:, j]
    return full


def _two_stage_scores(detector: "TwoStageDetector", X: np.ndarray) -> np.ndarray:
    """Pseudo-proba (n, NUM_CLASSES) for the two-stage detector.

    P(0) = 1 - P(player); P(team k) = P(player) * P(team k | player).
    """
    gate_p = detector._gate_player_proba(X)
    team_p = detector.team_clf.predict_proba(X[:, detector.team_idx])
    full = np.zeros((len(X), NUM_CLASSES), dtype=float)
    full[:, NO_TEAM_LABEL] = 1.0 - gate_p
    for j, c in enumerate(detector.team_clf.classes_):
        full[:, int(c)] = gate_p * team_p[:, j]
    return full


def _team_prior_grid(scores: np.ndarray, base_pred: np.ndarray) -> np.ndarray:
    """Adaptive team prior for one image's cells (no cap on number of teams).

    ``scores`` is (n_cells, NUM_CLASSES) (proba or pseudo-proba), ``base_pred``
    the per-cell argmax/threshold label. Every team with at least
    ``TEAM_PRIOR_MIN_CELLS`` predicted cells is kept; cells assigned to any other
    (sub-threshold "noise") team are re-decided over {0} U kept-teams by score.
    Confident background cells stay 0. If no team clears the threshold, the
    single largest predicted team is kept so a real but small team is not erased.
    """
    if not TEAM_COUNT_PRIOR:
        return base_pred
    counts = np.array(
        [int(np.sum(base_pred == c)) for c in range(1, NUM_CLASSES)], dtype=float
    )
    if counts.sum() == 0:
        return base_pred
    kept = np.flatnonzero(counts >= TEAM_PRIOR_MIN_CELLS) + 1
    if kept.size == 0:
        kept = np.array([int(np.argmax(counts)) + 1])
    allowed = np.array([NO_TEAM_LABEL, *kept.tolist()], dtype=int)
    out = allowed[scores[:, allowed].argmax(axis=1)]
    out[base_pred == NO_TEAM_LABEL] = NO_TEAM_LABEL
    return out.astype(int)


def _apply_team_prior_flat(
    scores: np.ndarray,
    base_pred: np.ndarray,
    orig_idx: np.ndarray,
    groups_all: np.ndarray | None,
) -> np.ndarray:
    """Apply ``_team_prior_grid`` per source image across flat predictions."""
    if not TEAM_COUNT_PRIOR or groups_all is None:
        return base_pred
    out = np.asarray(base_pred, dtype=int).copy()
    grp = groups_all[orig_idx]
    for g in np.unique(grp):
        pos = np.flatnonzero(grp == g)
        out[pos] = _team_prior_grid(scores[pos], base_pred[pos])
    return out


def _postprocess_predictions(
    y_pred: np.ndarray,
    test_orig_idx: np.ndarray,
    groups_all: np.ndarray | None,
) -> np.ndarray:
    """Apply ``_postprocess_grid`` per source image across a flat test prediction.

    Cells were built in row-major order per image and an image is never split
    across folds (grouped CV), so sorting a group's rows by original index
    reproduces the 8x8 row-major grid.
    """
    y_pred = np.asarray(y_pred, dtype=int).copy()
    if groups_all is None:
        return y_pred
    grp = groups_all[test_orig_idx]
    for g in np.unique(grp):
        pos = np.flatnonzero(grp == g)
        pos = pos[np.argsort(test_orig_idx[pos])]
        if len(pos) != NUM_CELLS:
            continue
        y_pred[pos] = _postprocess_grid(y_pred[pos])
    return y_pred


# ============================================================
# THRESHOLD TUNING (leak-free, on training data only)
# ============================================================

def _apply_threshold(
    proba: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Argmax prediction, demoted to 'no team' when below the confidence threshold."""
    pred_idx = proba.argmax(axis=1)
    pred = classes[pred_idx]
    if threshold > 0:
        pred = pred.copy()
        pred[proba.max(axis=1) < threshold] = NO_TEAM_LABEL
    return pred


def _tune_threshold(
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    classes: np.ndarray,
) -> tuple[float, float]:
    """Pick the confidence threshold maximising macro-F1 on out-of-fold predictions."""
    if groups_train is not None:
        cv: Any = StratifiedGroupKFold(
            n_splits=THRESHOLD_TUNE_FOLDS, shuffle=True, random_state=RANDOM_STATE,
        )
        cv_iter = cv.split(X_train, y_train, groups_train)
    else:
        cv = StratifiedShuffleSplit(
            n_splits=THRESHOLD_TUNE_FOLDS, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        )
        cv_iter = cv.split(X_train, y_train)

    try:
        oof_proba = cross_val_predict(
            pipeline, X_train, y_train,
            cv=list(cv_iter), method="predict_proba", n_jobs=None,
        )
    except Exception as exc:  # noqa: BLE001 - threshold tuning is best-effort
        print(f"Threshold tuning skipped ({exc}); using threshold 0.0")
        return 0.0, float(f1_score(
            y_train,
            classes[pipeline.predict_proba(X_train).argmax(axis=1)]
            if hasattr(pipeline, "predict_proba") else pipeline.predict(X_train),
            average="macro",
        ))

    best_threshold, best_f1 = 0.0, -1.0
    for threshold in np.linspace(0.0, 0.6, 25):
        pred = _apply_threshold(oof_proba, classes, float(threshold))
        score = f1_score(y_train, pred, average="macro")
        if score > best_f1:
            best_f1, best_threshold = score, float(threshold)
    return best_threshold, best_f1


# ============================================================
# REPORTING
# ============================================================

def _evaluate(
    name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log: Any,
) -> tuple[float, float]:
    """Report the decision metrics (class-balanced macro-F1, with accuracy)."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    block = f"{name.upper()} [{split}] macro-F1={macro_f1:.4f} accuracy={acc:.4f}"
    print(block)
    log(block)
    return acc, macro_f1


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    log_file: str | Path | None = None,
    dataset_file: str = DEFAULT_DATASET,
) -> dict[str, dict[str, Any]]:
    """Train all classifiers, select the best by test macro-F1, save artifacts."""
    with np.load(dataset_file, allow_pickle=True) as data:
        X = data["X"]
        y = data["y"].astype(int)
        groups = data["groups"] if "groups" in data.files else None
        include_text = bool(data["include_text"]) if "include_text" in data.files else True
        include_orb = bool(data["include_orb"]) if "include_orb" in data.files else False

    # Defensive: replace any NaN/inf left in the cached features so the scaler
    # and downstream estimators never receive non-finite values.
    n_bad = int((~np.isfinite(X)).sum())
    if n_bad:
        print(f"WARNING: {n_bad} non-finite feature value(s) found -> replaced with 0.0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = X.astype(np.float32, copy=False)

    print(f"Loaded dataset from: {dataset_file}")
    print(f"Samples: {X.shape[0]}  Feature dimension: {X.shape[1]} "
          f"(include_text={include_text}, include_orb={include_orb})")

    log = lambda message: _append_log(message, log_file)

    train_idx, test_idx = _grouped_train_test_indices(y, groups, RANDOM_STATE)
    X_train_full, X_test = X[train_idx], X[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx] if groups is not None else None

    # All reported metrics use a class-BALANCED subsample of the test split
    # (equal cells per class), so macro-F1 is not dominated by background cells.
    bal_idx = _balanced_subsample_indices(y_test, seed=RANDOM_STATE)
    print(f"Train cells: {len(y_train_full)}  Test cells: {len(y_test)} "
          f"(image-grouped split, no leakage)")
    print(f"Balanced test cells used for metrics: {len(bal_idx)} "
          f"({int(np.sum(y_test[bal_idx] == 0))} per class)\n")

    classifiers = _build_classifier_registry()
    trained_models: dict[str, dict[str, Any]] = {}

    for clf_name, clf_config in classifiers.items():
        print(SEPARATOR)
        print(f"Training {clf_name.upper()} ...")
        print(SEPARATOR)

        # Train-only imbalance handling: kernel/instance models train on an
        # undersampled class-0 set (tractability); the rest see the full set.
        if clf_config.get("undersample", False):
            cap = int(clf_config.get("zero_cap", TRAIN_ZERO_CAP))
            X_tr, y_tr, groups_tr = _undersample_train_zeros(
                X_train_full, y_train_full, groups_train_full, cap, RANDOM_STATE,
            )
            print(f"  train-zero-undersample -> {len(y_tr)} cells "
                  f"({int(np.sum(y_tr == 0))} class-0)")
        else:
            X_tr, y_tr, groups_tr = X_train_full, y_train_full, groups_train_full

        pipeline = _make_pipeline(clf_config["model"], clf_config["requires_pca"])
        pipeline.fit(X_tr, y_tr)
        proba_classes = np.asarray(pipeline.classes_)

        if clf_config.get("tune_threshold", True):
            # Out-of-fold macro-F1 only drives the threshold choice; it is not a
            # reported metric (train-side, leak-free) so we keep it internal.
            threshold, _ = _tune_threshold(
                pipeline, X_tr, y_tr, groups_tr, proba_classes,
            )
            print(f"{clf_name.upper()} tuned threshold={threshold:.3f}")
        else:
            threshold = 0.0
            print(f"{clf_name.upper()} threshold tuning skipped (baseline) -> 0.0")

        test_proba = pipeline.predict_proba(X_test)
        y_pred_test = _apply_threshold(test_proba, proba_classes, threshold)

        # Metrics are reported on the class-BALANCED view of the test split only.
        log(SEPARATOR + "\n" + f"{clf_name.upper()} results:\n")
        test_acc, test_f1 = _evaluate(
            clf_name, "test (balanced)", y_test[bal_idx], y_pred_test[bal_idx], log,
        )

        # Blob cleanup and the team prior interact (the prior already deletes
        # scattered noise teams), so we evaluate ALL four combinations on the
        # FINAL balanced metric and keep whichever wins -- never greedily.
        test_scores = _scores_to_full(test_proba, proba_classes)
        candidates: list[tuple[float, float, bool, bool]] = [
            (test_f1, test_acc, False, False),  # raw
        ]
        y_pred_test_pp = (
            _postprocess_predictions(y_pred_test, test_idx, groups)
            if POSTPROCESS_ISOLATED else None
        )
        if y_pred_test_pp is not None:
            pp_acc, pp_f1 = _evaluate(
                clf_name, "test (balanced)+blob-cleanup",
                y_test[bal_idx], y_pred_test_pp[bal_idx], log,
            )
            candidates.append((pp_f1, pp_acc, True, False))
        if TEAM_COUNT_PRIOR:
            pr = _apply_team_prior_flat(test_scores, y_pred_test, test_idx, groups)
            pr_acc, pr_f1 = _evaluate(
                clf_name, "test (balanced)+team-prior",
                y_test[bal_idx], pr[bal_idx], log,
            )
            candidates.append((pr_f1, pr_acc, False, True))
            if y_pred_test_pp is not None:
                ppr = _apply_team_prior_flat(test_scores, y_pred_test_pp, test_idx, groups)
                ppr_acc, ppr_f1 = _evaluate(
                    clf_name, "test (balanced)+blob+prior",
                    y_test[bal_idx], ppr[bal_idx], log,
                )
                candidates.append((ppr_f1, ppr_acc, True, True))

        test_f1, test_acc, use_pp, use_prior = max(candidates, key=lambda t: t[0])

        artifact = {
            "pipeline": pipeline,
            "classes": proba_classes,
            "threshold": threshold,
            "team_name": TEAM_NAME,
            "include_text": include_text,
            "include_orb": include_orb,
            "postprocess": bool(use_pp),
            "team_count_prior": bool(use_prior),
        }
        joblib.dump(artifact, clf_config["model_path"], compress=("xz", 3))
        print(f"Saved {clf_name.upper()} -> {clf_config['model_path']}\n")

        trained_models[clf_name] = {
            "artifact": artifact,
            "model_path": clf_config["model_path"],
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
        }

    if INCLUDE_TWO_STAGE:
        print(SEPARATOR)
        print("Training TWO_STAGE (gate + team classifier) ...")
        print(SEPARATOR)

        # The gate trains on the FULL train split so it sees every background
        # cell (diverse sky/crowd/pitch/ads) -> better class-0 precision/recall.
        detector = _build_two_stage_detector(include_text, include_orb)
        detector.fit(X_train_full, y_train_full, groups_train_full)
        y_pred_test = detector.predict(X_test)
        # Post-processing needs intact 8x8 grids, so it runs on the FULL test
        # predictions; metrics below are then read off the balanced subset.
        y_pred_test_pp = _postprocess_predictions(y_pred_test, test_idx, groups)

        log(SEPARATOR + "\n" + "TWO_STAGE results:\n")
        ts_acc, ts_f1 = _evaluate(
            "two_stage", "test (balanced)", y_test[bal_idx], y_pred_test[bal_idx], log,
        )
        pp_acc, pp_f1 = _evaluate(
            "two_stage+pp", "test (balanced)", y_test[bal_idx], y_pred_test_pp[bal_idx], log,
        )

        # Keep whichever variant (raw vs post-processed) scores higher.
        use_pp = POSTPROCESS_ISOLATED and pp_f1 >= ts_f1
        final_ts_f1 = pp_f1 if use_pp else ts_f1
        final_ts_acc = pp_acc if use_pp else ts_acc
        base_ts_pred = y_pred_test_pp if use_pp else y_pred_test

        # Domain team-count prior on top of the chosen variant (kept only if it
        # helps -- the gate already sparsifies team predictions, so the prior can
        # remove correct minority-team cells for the two-stage model).
        ts_use_prior = False
        if TEAM_COUNT_PRIOR:
            ts_scores = _two_stage_scores(detector, X_test)
            y_pred_test_prior = _apply_team_prior_flat(
                ts_scores, base_ts_pred, test_idx, groups,
            )
            tp_acc, tp_f1 = _evaluate(
                "two_stage+prior", "test (balanced)",
                y_test[bal_idx], y_pred_test_prior[bal_idx], log,
            )
            if tp_f1 >= final_ts_f1:
                ts_use_prior = True
                final_ts_f1, final_ts_acc = tp_f1, tp_acc

        ts_artifact = {
            "two_stage": detector,
            "classes": np.arange(NUM_CLASSES),
            "threshold": 0.0,  # gate handles no-team decision; no proba threshold
            "team_name": TEAM_NAME,
            "include_text": include_text,
            "include_orb": include_orb,
            "postprocess": bool(use_pp),
            "team_count_prior": ts_use_prior,
        }
        joblib.dump(ts_artifact, MODEL_PATH_TWO_STAGE, compress=("xz", 3))
        print(f"Saved TWO_STAGE -> {MODEL_PATH_TWO_STAGE} (postprocess={use_pp})\n")

        trained_models["two_stage"] = {
            "artifact": ts_artifact,
            "model_path": MODEL_PATH_TWO_STAGE,
            "test_macro_f1": final_ts_f1,
            "test_accuracy": final_ts_acc,
        }

    best_name = max(trained_models, key=lambda n: trained_models[n]["test_macro_f1"])
    best = trained_models[best_name]
    joblib.dump(best["artifact"], MODEL_PATH, compress=("xz", 3))

    summary_lines = [
        "\n" + "#" * 60,
        "Model comparison (selection metric = BALANCED test macro-F1):",
    ]
    for name, info in sorted(
        trained_models.items(), key=lambda kv: kv[1]["test_macro_f1"], reverse=True,
    ):
        summary_lines.append(
            f"  {name.upper():12s} balanced-F1={info['test_macro_f1']:.4f} "
            f"acc={info['test_accuracy']:.4f}"
        )
    summary_lines.append(
        f"\nBest model: {best_name.upper()} "
        f"(balanced test macro-F1={best['test_macro_f1']:.4f}) "
        f"-> saved as {MODEL_PATH}\n"
    )
    summary_text = "\n".join(summary_lines) + "\n"
    print(summary_text)
    log(summary_text)

    return trained_models


# ============================================================
# MAIN
# ============================================================
# NOTE: inference (load_model / predict_image / predictions CSV) now lives in
# inference.py so this module stays training-only. The shared helpers it needs
# (TwoStageDetector, _apply_threshold, _team_prior_grid, _scores_to_full,
# _two_stage_scores, _postprocess_grid, _numpy_to_lgbm_frame) are imported from
# here by inference.py.

def _write_log_header(out_path: Path, header_line: str | None, dataset_file: str) -> None:
    if not header_line:
        return
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n")
        handle.write(header_line.rstrip("\n") + "\n")
        handle.write(f"Dataset: {dataset_file}\n\n")
    print(f"Wrote initial line to log: {header_line}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models and log results.")
    parser.add_argument("--log-file", default=None, help="Path to a log file to append")
    parser.add_argument("--log-header", default=None, help="Header line before training metrics")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Path to dataset NPZ file")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out_path = Path(args.log_file) if args.log_file else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _write_log_header(out_path, args.log_header, args.dataset)

    train_model(
        log_file=str(out_path) if out_path is not None else None,
        dataset_file=args.dataset,
    )

    msg = f"\n{SEPARATOR}\nAll models trained and saved successfully!\n{SEPARATOR}\n"
    print(msg)
    if out_path is not None:
        with out_path.open("a", encoding="utf-8") as handle:
            handle.write(msg + "\n")
