"""Build the cell-level feature dataset from labelled IPL grid images.

Each 800x600 image is split into an 8x8 grid (64 cells) and every cell is
described by classical, hand-crafted computer-vision features (colour
histograms, colour moments, LBP texture, jersey-colour priors, positional and
neighbour-context cues, and rule-based text-likeness statistics).
The resulting (cells, features) matrix, the per-cell team
labels, and the source-image group ids are saved to a compressed .npz for
training and evaluation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import hog
from skimage.feature import local_binary_pattern

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = (800, 600)
GRID_ROWS = 8
GRID_COLS = 8
CELL_WIDTH = 100
CELL_HEIGHT = 75
NUM_CELLS = GRID_ROWS * GRID_COLS
NUM_CLASSES = 11

TRAIN_IMAGE_FOLDER = "..\\Create Labels\\static\\dataset"
LABEL_CSV = "..\\Create Labels\\outputs\\labels.csv"
OUTPUT_NPZ = "..\\Create Labels\\outputs\\ipl_features.npz"
DISTRIBUTION_PLOT = "class_distribution.png"

# Reproducibility
RANDOM_SEED = 42

# The label CSV uses non-padded headers c1..c64; the deliverable predictions CSV
# uses zero-padded c01..c64 (handled in base_model.py).
CELL_LABEL_COLUMNS = tuple(f"c{i}" for i in range(1, NUM_CELLS + 1))

_EPS = 1e-6

# HSV histogram bin counts
_H_BINS, _S_BINS, _V_BINS = 32, 16, 16
_H_RANGE = (0, 180)
_SV_RANGE = (0, 256)

# LBP parameters
_LBP_RADIUS = 1
_LBP_N_POINTS = 8 * _LBP_RADIUS

# ORB parameters
_ORB_N_FEATURES = 32

# ------------------------------------------------------------
# ORB keypoint block
# ------------------------------------------------------------
# Ablation finding: the mean ORB descriptor (averaging binary ORB descriptors
# across keypoints) is only weakly discriminative for jersey/team identity and
# adds 32 noisy dims that mostly increase the trees' capacity to memorise the
# training set. It is therefore OFF by default. Flip to True to A/B test it.
INCLUDE_ORB = False
ORB_FEATURE_DIM = 32

# ------------------------------------------------------------
# Team-color block (jersey color priors, helps the weakest team: RCB)
# ------------------------------------------------------------
# Hand-crafted dominant-color cues. RCB (class 9) is the hardest team for every
# model; explicit red/dark fractions give the classifier a direct red+black cue
# instead of having to infer it from the marginal HSV histogram. The 7th dim
# (navy fraction) separates GT's dark navy from MI's bright royal blue, which
# otherwise collapse into the single "blue" fraction and cause GT<->MI confusion.
TEAM_COLOR_DIM = 7

# ------------------------------------------------------------
# Positional + neighbor-context block
# ------------------------------------------------------------
# Cell (row, col) within the 8x8 grid is a free, hand-crafted feature: sky /
# crowd cluster in the top rows, pitch in the middle, etc. Neighbor-context
# aggregates (grass fraction / saturation / value of the 4-neighborhood) add
# spatial smoothness using only simple, hand-crafted statistics.
POSITION_DIM = 2          # normalized (row, col)
NEIGHBOR_DIM = 4          # self grass-frac + neighbor mean grass-frac / sat / val

# ------------------------------------------------------------
# Text-feature config
# ------------------------------------------------------------
# Toggle the 9-dim text block on/off. This is the knob for the ablation
# experiment (train/eval macro-F1 with vs. without the text block); the chosen
# value is recorded in the saved .npz so inference stays consistent.
INCLUDE_TEXT_FEATURES = True
TEXT_FEATURE_DIM = 9

# Canny thresholds. Fixed thresholds are brittle under varied stadium lighting,
# so by default we derive them from Otsu per cell; set False to use the fixed
# fallback below.
TEXT_ADAPTIVE_CANNY = True
_CANNY_LOW_FIXED = 100
_CANNY_HIGH_FIXED = 200

# Cheap "is this worth scoring as text?" gate. Real match photos contain
# boundary ads and scoreboards that look very text-like and cause false
# positives. A full player-vs-background gate (two-stage) is the robust fix; as
# a cheap stand-in we suppress text features on grass-dominant (pitch) cells.
# NOTE: this does NOT remove ad/scoreboard false positives
# Set False to disable.
TEXT_USE_GRASS_GATE = True
_GRASS_HUE_LO, _GRASS_HUE_HI = 35, 90      # OpenCV hue range [0, 180]
_GRASS_MIN_SAT = 40
_GRASS_DOMINANT_FRACTION = 0.5


# ============================================================
# FEATURE EXTRACTION  (classical, hand-crafted descriptors)
# ============================================================

def _normalize_histogram(hist: np.ndarray) -> np.ndarray:
    """Return L1-normalized histogram with numerical stability."""
    hist = hist.astype(np.float64, copy=False)
    return hist / (hist.sum() + _EPS)


def _to_grayscale(cell_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)


def extract_hsv_histogram(cell_img: np.ndarray) -> np.ndarray:
    """Marginal HSV histograms (jersey color is the dominant team cue)."""
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [_H_BINS], list(_H_RANGE))
    s_hist = cv2.calcHist([hsv], [1], None, [_S_BINS], list(_SV_RANGE))
    v_hist = cv2.calcHist([hsv], [2], None, [_V_BINS], list(_SV_RANGE))
    return _normalize_histogram(np.concatenate([h.flatten() for h in (h_hist, s_hist, v_hist)]))


def _channel_moments(channel: np.ndarray) -> tuple[float, float, float]:
    """Mean, std and (clipped) skewness of a single channel (color moments).

    The channel is normalised to [0, 1] so all three moments stay in a small,
    bounded range; skewness is clipped to avoid overflow on near-constant cells
    (a tiny std otherwise produces enormous, possibly non-finite, values).
    """
    x = channel.astype(np.float64).ravel() / 255.0
    mean = float(x.mean())
    std = float(x.std())
    if std < 1e-4:
        skew = 0.0
    else:
        skew = float(np.mean(((x - mean) / std) ** 3))
        skew = float(np.clip(skew, -10.0, 10.0))
    return mean, std, skew


def extract_color_moments(cell_img: np.ndarray) -> np.ndarray:
    """Color moments in HSV and LAB spaces (compact, lighting-robust color cue)."""
    feats: list[float] = []
    for conversion in (cv2.COLOR_BGR2HSV, cv2.COLOR_BGR2LAB):
        converted = cv2.cvtColor(cell_img, conversion)
        for ch in range(3):
            feats.extend(_channel_moments(converted[..., ch]))
    return np.asarray(feats, dtype=np.float64)  # 2 spaces * 3 ch * 3 moments = 18


def extract_team_color_features(cell_img: np.ndarray) -> np.ndarray:
    """Dominant jersey-color priors (hand-crafted, no learning).

    Layout (7 dims), all fractions in [0, 1] except the last:
      [0] red fraction     (CSK/RCB/PBKS/SRH lettering, RCB base)  -- hue wraps
      [1] yellow fraction  (CSK)
      [2] blue fraction    (DC/MI/RR/GT/KKR family)
      [3] dark fraction    (RCB black, low V)
      [4] bright fraction  (white kit elements / glare, high V)
      [5] dominant hue (normalised peak of the saturated-pixel hue histogram)
      [6] navy fraction    (GT dark navy: blue hue but low V) -- separates GT
                           from MI's bright royal blue, the main GT<->MI mixup

    RCB (class 9) is the weakest team across all models; the explicit red+dark
    cues give the classifier a direct signal rather than relying on the marginal
    HSV histogram alone. GT (navy) and MI (royal blue) share the blue hue band,
    so a dedicated navy cue (blue hue at low value) lets the classifier tell them
    apart instead of defaulting navy players to MI.
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    n = float(hue.size) + _EPS
    saturated = sat > 60

    blue_hue = (hue >= 90) & (hue <= 130)
    red = (((hue <= 10) | (hue >= 170)) & saturated)
    yellow = ((hue >= 20) & (hue <= 35) & saturated)
    blue = (blue_hue & saturated)
    dark = val < 50
    bright = val > 205
    # Navy = blue hue, saturated, but dim (royal blue is brighter). Bounds chosen
    # so GT navy (V~40-110) registers while MI royal blue (V>~120) does not.
    navy = (blue_hue & (sat > 50) & (val >= 40) & (val < 115))

    red_frac = float(red.sum()) / n
    yellow_frac = float(yellow.sum()) / n
    blue_frac = float(blue.sum()) / n
    dark_frac = float(dark.sum()) / n
    bright_frac = float(bright.sum()) / n
    navy_frac = float(navy.sum()) / n

    if saturated.any():
        dom_hue = float(np.bincount(hue[saturated].ravel(), minlength=180).argmax()) / 180.0
    else:
        dom_hue = 0.0

    return np.asarray(
        [red_frac, yellow_frac, blue_frac, dark_frac, bright_frac, dom_hue, navy_frac],
        dtype=np.float64,
    )


def _cell_color_summary(cell_img: np.ndarray) -> tuple[float, float, float]:
    """Compact per-cell color summary used to build neighbor-context features.

    Returns (grass_fraction, mean_saturation_norm, mean_value_norm).
    """
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hue >= _GRASS_HUE_LO) & (hue <= _GRASS_HUE_HI) & (sat > _GRASS_MIN_SAT)
    return (
        float(grass.mean()),
        float(sat.mean()) / 255.0,
        float(val.mean()) / 255.0,
    )


def extract_hog_features(cell_img: np.ndarray) -> np.ndarray:
    """Histogram of Oriented Gradients (shape/edge-orientation descriptor).

    Retained as a documented ablation: HOG added many shape dimensions without a
    measurable held-out gain (it mainly increased model capacity), so it is left
    DISABLED in ``extract_cell_base_features``. Kept here so the ablation can be
    reproduced by re-enabling the block.
    """
    return hog(
        _to_grayscale(cell_img),
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )


def extract_lbp_features(cell_img: np.ndarray) -> np.ndarray:
    """Uniform LBP histogram (jersey texture / patterns)."""
    lbp = local_binary_pattern(
        _to_grayscale(cell_img),
        _LBP_N_POINTS,
        _LBP_RADIUS,
        method="uniform",
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, _LBP_N_POINTS + 3),
        range=(0, _LBP_N_POINTS + 2),
    )
    return _normalize_histogram(hist)


def _canny_edges(gray: np.ndarray) -> np.ndarray:
    """Canny edges with Otsu-adaptive thresholds (robust to lighting)."""
    if TEXT_ADAPTIVE_CANNY:
        otsu_high, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high = max(1.0, float(otsu_high))
        low = 0.5 * high
        return cv2.Canny(gray, low, high)
    return cv2.Canny(gray, _CANNY_LOW_FIXED, _CANNY_HIGH_FIXED)


def _is_grass_dominant(cell_img: np.ndarray) -> bool:
    """Cheap gate: True when the cell is mostly green pitch (likely no jersey)."""
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0], hsv[..., 1]
    grass = (hue >= _GRASS_HUE_LO) & (hue <= _GRASS_HUE_HI) & (sat > _GRASS_MIN_SAT)
    return float(grass.mean()) > _GRASS_DOMINANT_FRACTION


def _mser_text_features(gray: np.ndarray) -> tuple[float, float, float]:
    """MSER blob stats: (region count, mean aspect ratio, std of region areas).

    Letters appear as many small, similarly-sized maximally-stable regions.
    Returns zeros when no regions are found.
    """
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    if not regions:
        return 0.0, 0.0, 0.0

    areas, aspects = [], []
    for region in regions:
        _, _, w, h = cv2.boundingRect(region)
        if h <= 0:
            continue
        areas.append(float(w * h))
        aspects.append(float(w) / float(h))
    if not areas:
        return float(len(regions)), 0.0, 0.0
    return float(len(regions)), float(np.mean(aspects)), float(np.std(areas))


def _stroke_width_features(edges: np.ndarray) -> tuple[float, float]:
    """Stroke-width stats from a distance transform of the edge map.

    Text has roughly constant stroke thickness -> low std. Returns zeros when
    there are no edges.
    """
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    stroke = dist[dist > 0]
    if stroke.size == 0:
        return 0.0, 0.0
    return float(stroke.mean()), float(stroke.std())


def _stroke_color_features(cell_img: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean HSV of edge (stroke) pixels -- the lettering color is a team cue.

    Returns zeros when there are no edge pixels.
    """
    mask = edges > 0
    if not mask.any():
        return np.zeros(3, dtype=np.float64)
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
    return hsv[mask].mean(axis=0).astype(np.float64)


def extract_text_features(cell_img: np.ndarray) -> np.ndarray:
    """Rule-based "how text-like is this cell?" descriptor.

    Jersey lettering / logos leave predictable fingerprints (many small,
    equal-thickness, high-contrast strokes). These are summarised with simple
    image statistics (MSER region shape, stroke width, edge density and stroke
    colour) -- this measures the *presence* of text-like structure and performs
    no character recognition. Layout of the 9-dim vector:
      [0]   MSER region count
      [1]   MSER mean aspect ratio
      [2]   MSER std of region areas
      [3]   stroke-width mean (distance transform of edges)
      [4]   stroke-width std
      [5]   edge density (fraction of edge pixels)
      [6:9] mean H, S, V of edge pixels (stroke color)

    A grass-dominant gate suppresses these features on pitch cells to limit
    false positives. Returns all-zeros when gated out or when no signal exists.
    """
    if TEXT_USE_GRASS_GATE and _is_grass_dominant(cell_img):
        return np.zeros(TEXT_FEATURE_DIM, dtype=np.float64)

    gray = _to_grayscale(cell_img)
    edges = _canny_edges(gray)

    count, mean_aspect, std_area = _mser_text_features(gray)
    stroke_mean, stroke_std = _stroke_width_features(edges)
    edge_density = float(edges.mean()) / 255.0
    stroke_color = _stroke_color_features(cell_img, edges)

    return np.asarray(
        [count, mean_aspect, std_area, stroke_mean, stroke_std, edge_density,
         *stroke_color],
        dtype=np.float64,
    )


def extract_orb_features(cell_img: np.ndarray) -> np.ndarray:
    """Mean ORB descriptor (hand-crafted keypoint cue)."""
    orb = cv2.ORB_create(nfeatures=_ORB_N_FEATURES)
    _, descriptors = orb.detectAndCompute(_to_grayscale(cell_img), None)
    if descriptors is None:
        return np.zeros(_ORB_N_FEATURES, dtype=np.float64)
    return descriptors.mean(axis=0)


def extract_cell_base_features(
    cell_img: np.ndarray,
    include_orb: bool = INCLUDE_ORB,
) -> np.ndarray:
    """Per-cell descriptors that depend ONLY on the cell itself.

    Excludes position, neighbor-context and the text block -- those are added by
    ``extract_grid_features`` (position/neighbor need grid context, text is kept
    last so the gate can cheaply ignore it). Order:
      hsv_hist(64), color_moments(18), lbp(10), team_color(7), [orb(32)]
    """
    blocks = [
        extract_hsv_histogram(cell_img),        # 64  color
        extract_color_moments(cell_img),        # 18  color (HSV + LAB)
        extract_lbp_features(cell_img),         # 10  texture
        extract_team_color_features(cell_img),  # 7   jersey-color priors
        # extract_hog_features(cell_img),       #     shape (disabled ablation)
    ]
    if include_orb:
        blocks.append(extract_orb_features(cell_img))  # 32  keypoints (ablated off)
    vector = np.concatenate(blocks)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# FEATURE SCHEMA  (shared with base_model.py for per-stage selection)
# ============================================================

def feature_layout(
    include_text: bool = INCLUDE_TEXT_FEATURES,
    include_orb: bool = INCLUDE_ORB,
) -> list[tuple[str, int]]:
    """Ordered (block_name, dim) list describing the full per-cell vector.

    The order here is the canonical feature order used everywhere (dataset build
    and inference). Position/neighbor are inserted before the text block.
    """
    layout: list[tuple[str, int]] = [
        ("hsv_hist", _H_BINS + _S_BINS + _V_BINS),  # 64
        ("color_moments", 18),
        ("lbp", _LBP_N_POINTS + 2),                  # 10
        ("team_color", TEAM_COLOR_DIM),              # 7
    ]
    if include_orb:
        layout.append(("orb", ORB_FEATURE_DIM))
    layout.append(("position", POSITION_DIM))
    layout.append(("neighbor", NEIGHBOR_DIM))
    if include_text:
        layout.append(("text", TEXT_FEATURE_DIM))
    return layout


def feature_names(
    include_text: bool = INCLUDE_TEXT_FEATURES,
    include_orb: bool = INCLUDE_ORB,
) -> list[str]:
    """Flat per-dimension feature names (e.g. ``hsv_hist_000``)."""
    names: list[str] = []
    for block, dim in feature_layout(include_text, include_orb):
        names.extend(f"{block}_{i:03d}" for i in range(dim))
    return names


def feature_block_indices(
    blocks: tuple[str, ...],
    include_text: bool = INCLUDE_TEXT_FEATURES,
    include_orb: bool = INCLUDE_ORB,
) -> np.ndarray:
    """Column indices spanned by the named blocks (for per-stage selection)."""
    wanted = set(blocks)
    idx: list[int] = []
    offset = 0
    for block, dim in feature_layout(include_text, include_orb):
        if block in wanted:
            idx.extend(range(offset, offset + dim))
        offset += dim
    return np.asarray(sorted(idx), dtype=int)


def extract_grid_features(
    image: np.ndarray,
    include_text: bool = INCLUDE_TEXT_FEATURES,
    include_orb: bool = INCLUDE_ORB,
) -> np.ndarray:
    """Full (NUM_CELLS, D) feature matrix for one already-resized image.

    Adds positional (normalized row/col) and neighbor-context features that a
    single isolated cell cannot provide. Used by both dataset build and
    inference so train/serve features are identical.
    """
    cells = split_into_grid(image)
    base = np.stack([extract_cell_base_features(c, include_orb=include_orb) for c in cells])

    summaries = np.array([_cell_color_summary(c) for c in cells], dtype=np.float64)
    grid_summ = summaries.reshape(GRID_ROWS, GRID_COLS, 3)

    rows = np.repeat(np.arange(GRID_ROWS), GRID_COLS)
    cols = np.tile(np.arange(GRID_COLS), GRID_ROWS)
    position = np.stack(
        [rows / (GRID_ROWS - 1), cols / (GRID_COLS - 1)], axis=1
    ).astype(np.float64)

    neighbor = np.zeros((NUM_CELLS, NEIGHBOR_DIM), dtype=np.float64)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell_idx = r * GRID_COLS + c
            neigh = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    neigh.append(grid_summ[nr, nc])
            neigh_mean = np.mean(neigh, axis=0) if neigh else np.zeros(3)
            neighbor[cell_idx] = [
                grid_summ[r, c, 0],   # self grass-fraction
                neigh_mean[0],        # neighbor mean grass-fraction
                neigh_mean[1],        # neighbor mean saturation
                neigh_mean[2],        # neighbor mean value
            ]

    parts = [base, position, neighbor]
    if include_text:
        text = np.stack([extract_text_features(c) for c in cells])
        parts.append(text)
    grid = np.concatenate(parts, axis=1)
    return np.nan_to_num(grid, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================
# GRID SPLITTING
# ============================================================

def split_into_grid(image: np.ndarray) -> list[np.ndarray]:
    """Split a full image into GRID_ROWS x GRID_COLS cell crops (row-major order)."""
    return [
        image[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT, col * CELL_WIDTH:(col + 1) * CELL_WIDTH]
        for row in range(GRID_ROWS)
        for col in range(GRID_COLS)
    ]


# ============================================================
# DATASET CREATION
# ============================================================

def build_dataset(
    label_csv: str = LABEL_CSV,
    image_folder: str = TRAIN_IMAGE_FOLDER,
    include_text: bool = INCLUDE_TEXT_FEATURES,
    include_orb: bool = INCLUDE_ORB,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, groups) where groups is the source image name per cell.

    The image name is tracked so downstream splitting/CV can be image-grouped
    (cells from one image must never be split across train/test -> no leakage).
    Features are grid-aware (include positional + neighbor-context); include_text
    and include_orb toggle the corresponding blocks for ablation studies.
    """
    labels_df = pd.read_csv(label_csv)
    features: list[np.ndarray] = []
    labels: list[object] = []
    groups: list[str] = []

    for row in labels_df.itertuples(index=False):
        image_path = Path(image_folder) / row.image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not read {image_path}")
            continue

        grid = extract_grid_features(
            cv2.resize(image, IMAGE_SIZE),
            include_text=include_text,
            include_orb=include_orb,
        )
        for cell_idx, label_col in enumerate(CELL_LABEL_COLUMNS):
            features.append(grid[cell_idx].astype(np.float32))
            labels.append(getattr(row, label_col))
            groups.append(row.image)

        print(f"Processed {row.image}")

    X = np.asarray(features)
    y = np.asarray(labels)
    groups_arr = np.asarray(groups)
    print(Counter(y))
    plot_class_distribution(y, "Original Class Distribution")
    return X, y, groups_arr


# ============================================================
# VISUALIZATION
# ============================================================

def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution") -> None:
    """Plot and save bar chart of class distribution."""
    counter = Counter(y)
    classes = sorted(counter)
    counts = [counter[c] for c in classes]

    _, ax = plt.subplots(figsize=(12, 5))
    ax.bar(classes, counts, color="steelblue", edgecolor="black")
    ax.set(xlabel="Class Label", ylabel="Count", title=title)
    ax.set_xticks(classes)
    ax.grid(axis="y", alpha=0.3)

    for cls, count in zip(classes, counts):
        ax.text(cls, count, str(count), ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(DISTRIBUTION_PLOT, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the cell-level feature dataset.")
    parser.add_argument(
        "--no-text",
        action="store_true",
        help="Drop the 9-dim text block (ablation). Writes ipl_features_no_text.npz.",
    )
    parser.add_argument("--output", default=None, help="Override output .npz path.")
    parser.add_argument(
        "--with-orb",
        action="store_true",
        help="Include the 32-dim ORB block (off by default; ablation A/B).",
    )
    args = parser.parse_args()

    include_text = INCLUDE_TEXT_FEATURES and not args.no_text
    include_orb = INCLUDE_ORB or args.with_orb
    output_npz = args.output or (OUTPUT_NPZ if include_text else "ipl_features_no_text.npz")

    np.random.seed(RANDOM_SEED)
    print(f"Building dataset (include_text={include_text}, include_orb={include_orb})...")
    # NOTE: we deliberately save the FULL dataset (no class-0 undersampling here).
    # Class-0 dominates a real 8x8 grid, so the held-out evaluation must reflect
    # that. Imbalance is handled at TRAIN time only (base_model.py), keeping the
    # test split's natural class distribution intact for a realistic metric.
    X, y, groups = build_dataset(include_text=include_text, include_orb=include_orb)

    print(Counter(y))
    print("Dataset shape:", X.shape)

    names = np.array(feature_names(include_text, include_orb))
    block_names = np.array([b for b, _ in feature_layout(include_text, include_orb)])
    block_dims = np.array([d for _, d in feature_layout(include_text, include_orb)])

    np.savez_compressed(
        output_npz,
        X=X.astype(np.float32),
        y=y,
        groups=groups,
        include_text=np.array(include_text),
        include_orb=np.array(include_orb),
        feature_names=names,
        block_names=block_names,
        block_dims=block_dims,
    )
    print(f"Saved FULL features ({X.shape}) to {output_npz} "
          f"(include_text={include_text}, include_orb={include_orb})")
