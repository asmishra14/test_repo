"""Inference / serving pipeline for the IPL grid classifier.

Reads a saved ``model_<teamname>.pkl`` produced by ``base_model.py``, accepts a
single image or folders of images, and emits per-cell predictions (c01..c64,
row-major) or the deliverable predictions CSV
("Image File Name, Train Or Test, c01..c64").

Training/evaluation live in ``base_model.py``; this module is the lightweight
read-the-pickle-and-predict path required by the problem statement. The feature
extraction, the TwoStageDetector class and the shared post-processing helpers are
imported from ``base_model`` so train/serve stay byte-for-byte consistent.

Usage:
  # single image -> printed 8x8 grid
  python inference.py --model model_teamname.pkl --image path/to/img.jpg

  # both splits -> predictions CSV (PDF schema)
  python inference.py --model model_teamname.pkl \
      --train-images data/train --test-images data/test \
      --predictions-csv predictions.csv
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import base_model
from base_model import (
    CELL_COLUMNS,
    GRID_COLS,
    GRID_ROWS,
    IMAGE_SIZE,
    MODEL_PATH,
    NUM_CLASSES,
    TwoStageDetector,
    _apply_threshold,
    _postprocess_grid,
    _scores_to_full,
    _team_prior_grid,
    _two_stage_scores,
)
from create_dataset import extract_grid_features

# ------------------------------------------------------------
# Pickle-compatibility shim.
# ------------------------------------------------------------
# The artifacts were saved while ``base_model.py`` ran as ``__main__`` (i.e.
# ``python base_model.py``), so the pickled LightGBM FunctionTransformer and the
# TwoStageDetector reference ``__main__._numpy_to_lgbm_frame`` /
# ``__main__.TwoStageDetector``. Expose those names on whatever module is acting
# as ``__main__`` here so ``joblib.load`` can resolve them, no matter how this
# file is launched.
_main = sys.modules.get("__main__")
if _main is not None:
    for _name in ("TwoStageDetector", "_numpy_to_lgbm_frame"):
        if not hasattr(_main, _name):
            setattr(_main, _name, getattr(base_model, _name))

IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)
JPG_EXTENSIONS = frozenset({".jpg", ".jpeg"})

# Team id -> short label and an overlay color (BGR, for OpenCV drawing).
TEAM_NAMES = {
    0: "-", 1: "CSK", 2: "DC", 3: "GT", 4: "KKR", 5: "LSG",
    6: "MI", 7: "PBKS", 8: "RR", 9: "RCB", 10: "SRH",
}
TEAM_COLORS_BGR = {
    1: (0, 215, 255),    # CSK yellow
    2: (255, 90, 0),     # DC blue
    3: (140, 90, 30),    # GT navy/teal
    4: (130, 0, 90),     # KKR purple
    5: (200, 160, 40),   # LSG cyan
    6: (200, 80, 0),     # MI royal blue
    7: (40, 40, 220),    # PBKS red
    8: (200, 80, 200),   # RR pink
    9: (40, 40, 150),    # RCB dark red
    10: (0, 140, 255),   # SRH orange
}


# ============================================================
# IMAGE PREPROCESSING  (format conversion + resize to 800x600)
# ============================================================

def _read_any_image(path: Path) -> np.ndarray:
    """Read an image as BGR. Tries OpenCV, then Pillow for formats OpenCV's
    build can't decode (e.g. some webp/tiff/heic variants)."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        from PIL import Image  # lazy: only needed for the fallback path
    except ImportError as exc:  # pragma: no cover
        raise FileNotFoundError(
            f"Could not read {path} with OpenCV and Pillow is unavailable."
        ) from exc
    try:
        with Image.open(path) as pil_img:
            rgb = np.asarray(pil_img.convert("RGB"))
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        raise FileNotFoundError(
            f"Could not decode image {path} (unsupported format?). "
            f"For HEIC/AVIF install pillow-heif / pillow-avif-plugin."
        ) from exc
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def prepare_image(
    image_path: str | Path,
    convert_dir: str | Path | None = None,
) -> tuple[np.ndarray, Path]:
    """Return an (800x600 BGR image, path-actually-used) ready for inference.

    Steps required by the problem statement, applied up-front:
      1. If the file is not already a JPG (any other format), decode it and write
         a sibling ``<stem>.jpg`` so a JPG copy exists, then use it.
      2. If the image is not 800x600, resize it to the required 800x600.
    """
    src = Path(image_path)
    image = _read_any_image(src)

    used_path = src
    if src.suffix.lower() not in JPG_EXTENSIONS:
        out_dir = Path(convert_dir) if convert_dir else src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        used_path = out_dir / (src.stem + ".jpg")
        cv2.imwrite(str(used_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"  converted {src.name} -> {used_path.name} (JPG)")

    h, w = image.shape[:2]
    if (w, h) != IMAGE_SIZE:
        print(f"  resized {w}x{h} -> {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
        image = cv2.resize(image, IMAGE_SIZE)
    return image, used_path


# ============================================================
# MODEL LOADING + SINGLE-IMAGE PREDICTION
# ============================================================

def load_model(model: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a saved artifact (path) or pass through an already-loaded dict."""
    if isinstance(model, dict):
        return model
    return joblib.load(str(model))


def predict_image_array(
    model: str | Path | dict[str, Any],
    image_bgr: np.ndarray,
    threshold: float | None = None,
) -> list[int]:
    """Predict NUM_CELLS labels (c01..c64, row-major) for an already-prepared
    800x600 BGR image array."""
    artifact = load_model(model)
    include_text = artifact.get("include_text", True)
    include_orb = artifact.get("include_orb", False)

    # Grid-aware features (positional + neighbor-context); train/serve use the
    # exact same extractor so there is no train/serve skew.
    features = extract_grid_features(
        image_bgr,
        include_text=include_text,
        include_orb=include_orb,
    ).astype(np.float64)

    # Two-stage path: detector internally selects its gate/team feature subsets.
    if "two_stage" in artifact:
        detector: TwoStageDetector = artifact["two_stage"]
        preds = detector.predict(features)
        if artifact.get("postprocess", False):
            preds = _postprocess_grid(preds)
        if artifact.get("team_count_prior", False):
            scores = _two_stage_scores(detector, features)
            preds = _team_prior_grid(scores, np.asarray(preds, dtype=int))
        return [int(p) for p in preds]

    # Single-stage path.
    pipeline = artifact["pipeline"]
    classes = np.asarray(artifact.get("classes", np.arange(NUM_CLASSES)))
    used_threshold = artifact.get("threshold", 0.0) if threshold is None else threshold
    clf = pipeline.named_steps.get("clf")
    if isinstance(clf, LGBMClassifier) and "lgbm_frame" not in pipeline.named_steps:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
                category=UserWarning,
            )
            proba = pipeline.predict_proba(features)
    else:
        proba = pipeline.predict_proba(features)
    preds = _apply_threshold(proba, classes, float(used_threshold))
    if artifact.get("postprocess", False):
        preds = _postprocess_grid(preds)
    if artifact.get("team_count_prior", False):
        scores = _scores_to_full(proba, classes)
        preds = _team_prior_grid(scores, np.asarray(preds, dtype=int))
    return [int(p) for p in preds]


def predict_image(
    model: str | Path | dict[str, Any],
    image_path: str | Path,
    threshold: float | None = None,
) -> list[int]:
    """Predict NUM_CELLS cell labels (c01..c64, row-major) for one image file.

    The image is first converted to JPG (if needed) and resized to 800x600.
    """
    image, _ = prepare_image(image_path)
    return predict_image_array(model, image, threshold)


def predict_grid(
    model: str | Path | dict[str, Any],
    image_path: str | Path,
    threshold: float | None = None,
) -> np.ndarray:
    """Like ``predict_image`` but reshaped to the 8x8 grid for display."""
    return np.asarray(predict_image(model, image_path, threshold), dtype=int).reshape(
        GRID_ROWS, GRID_COLS
    )


# ============================================================
# VISUALIZATION  (overlay predicted team per grid cell)
# ============================================================

def _put_text_outlined(img, text, org, scale=0.5, color=(255, 255, 255)) -> None:
    """White text with a black outline so it reads on any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def visualize_prediction(
    image_bgr: np.ndarray,
    preds: list[int] | np.ndarray,
    out_path: str | Path,
    alpha: float = 0.35,
    count: bool = True,
) -> tuple[Path, dict[int, int]]:
    """Render the 800x600 image with the 8x8 prediction overlaid and save it.

    Team cells are tinted with the team color (so the predicted regions are
    visible). When ``count`` is on, individual players are detected (pixel-level,
    classical CV), each is boxed and labelled ``TEAM-i`` near its top edge, and a
    per-team player tally is drawn in the top-left panel. Returns (path, {team:
    n_players}).
    """
    from create_dataset import CELL_HEIGHT, CELL_WIDTH

    base = image_bgr.copy()
    overlay = base.copy()
    grid = np.asarray(preds, dtype=int).reshape(GRID_ROWS, GRID_COLS)

    # translucent team-color tint per predicted cell
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            team = int(grid[r, c])
            if team > 0:
                x0, y0 = c * CELL_WIDTH, r * CELL_HEIGHT
                cv2.rectangle(overlay, (x0, y0), (x0 + CELL_WIDTH, y0 + CELL_HEIGHT),
                              TEAM_COLORS_BGR.get(team, (128, 128, 128)), -1)
    out = cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    # grid lines
    for r in range(GRID_ROWS + 1):
        cv2.line(out, (0, r * CELL_HEIGHT), (GRID_COLS * CELL_WIDTH, r * CELL_HEIGHT),
                 (60, 60, 60), 1)
    for c in range(GRID_COLS + 1):
        cv2.line(out, (c * CELL_WIDTH, 0), (c * CELL_WIDTH, GRID_ROWS * CELL_HEIGHT),
                 (60, 60, 60), 1)

    # per-cell team label (kept alongside the per-player indexing below)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            team = int(grid[r, c])
            if team > 0:
                x0, y0 = c * CELL_WIDTH, r * CELL_HEIGHT
                label = TEAM_NAMES.get(team, str(team))
                cv2.putText(out, label, (x0 + 3, y0 + CELL_HEIGHT - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, label, (x0 + 3, y0 + CELL_HEIGHT - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    # detect + localize players, box each instance and label TEAM-i
    counts: dict[int, int] = {}
    if count:
        from player_count import count_players

        detections = count_players(image_bgr, grid)
        for team, instances in detections.items():
            counts[team] = len(instances)
            color = TEAM_COLORS_BGR.get(team, (128, 128, 128))
            for inst in instances:
                x, y, w, h = inst.bbox
                cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
                tag = f"{TEAM_NAMES.get(team, str(team))}-{inst.index}"
                tx = max(0, min(inst.top_center[0] - 12, IMAGE_SIZE[0] - 60))
                ty = y - 6 if y - 6 > 12 else y + 16  # above the box, else inside
                _put_text_outlined(out, tag, (tx, ty), 0.55, color=(255, 255, 255))
    else:
        counts = {int(t): 0 for t in np.unique(grid) if t > 0}

    # top-left panel: a clear per-team player count, e.g. "RCB: 2" per row
    teams_present = sorted(counts)
    if teams_present:
        pad, row_h, sw = 6, 24, 16
        title_h = 22
        panel_w = 180
        panel_h = pad * 2 + title_h + row_h * len(teams_present)
        cv2.rectangle(out, (4, 4), (4 + panel_w, 4 + panel_h), (255, 255, 255), -1)
        cv2.rectangle(out, (4, 4), (4 + panel_w, 4 + panel_h), (0, 0, 0), 1)
        cv2.putText(out, "Players per team", (12, 4 + pad + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        for i, team in enumerate(teams_present):
            yrow = 4 + pad + title_h + row_h * i
            color = TEAM_COLORS_BGR.get(team, (128, 128, 128))
            cv2.rectangle(out, (12, yrow), (12 + sw, yrow + sw), color, -1)
            cv2.rectangle(out, (12, yrow), (12 + sw, yrow + sw), (0, 0, 0), 1)
            cv2.putText(out, f"{TEAM_NAMES.get(team, str(team))}: {counts[team]}",
                        (12 + sw + 8, yrow + sw - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2, cv2.LINE_AA)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)
    print(f"Saved visualization -> {out_path}")
    return out_path, counts


# ============================================================
# PREDICTIONS CSV  (PDF schema, train + test)
# ============================================================

def _list_images(folder: str | Path) -> list[str]:
    directory = Path(folder)
    if not directory.is_dir():
        return []
    return sorted(
        p.name for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def generate_predictions_csv(
    model: str | Path | dict[str, Any],
    splits: dict[str, str | Path],
    output_csv: str | Path = "predictions.csv",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Write predictions for both splits.

    ``splits`` maps the "Train Or Test" value to its image folder, e.g.
    {"Train": "data/train", "Test": "data/test"}.
    Output columns: Image File Name, Train Or Test, c01, ..., c64.
    """
    artifact = load_model(model)
    rows = []
    for split_name, folder in splits.items():
        for image_name in _list_images(folder):
            preds = predict_image(artifact, Path(folder) / image_name, threshold)
            row = {"Image File Name": image_name, "Train Or Test": split_name}
            row.update(dict(zip(CELL_COLUMNS, preds)))
            rows.append(row)
            print(f"Predicted [{split_name}] {image_name}")

    columns = ["Image File Name", "Train Or Test", *CELL_COLUMNS]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions ({len(df)} rows) to {output_csv}")
    return df


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a saved model.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model_<teamname>.pkl")
    parser.add_argument("--image", default=None, help="Single image -> print 8x8 grid")
    parser.add_argument("--viz", nargs="?", const="__auto__", default=None,
                        help="Save an overlay visualization (optional output path; "
                             "defaults to <image>_pred.png)")
    parser.add_argument("--train-images", default=None, help="Train image folder for the CSV")
    parser.add_argument("--test-images", default=None, help="Test image folder for the CSV")
    parser.add_argument("--predictions-csv", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.image:
        # Preprocess once (convert non-JPG -> JPG, resize to 800x600), then
        # predict and optionally visualize on the SAME prepared image.
        image, _ = prepare_image(args.image)
        preds = predict_image_array(args.model, image, args.threshold)
        grid = np.asarray(preds, dtype=int).reshape(GRID_ROWS, GRID_COLS)
        print(f"Predictions for {args.image} (model={args.model}):")
        print(grid)
        print("non-zero cells:", int((grid > 0).sum()),
              "teams present:", [TEAM_NAMES[t] for t in sorted(set(grid[grid > 0].tolist()))])
        if args.viz is not None:
            out = (Path(args.image).with_name(Path(args.image).stem + "_pred.png")
                   if args.viz == "__auto__" else Path(args.viz))
            _, counts = visualize_prediction(image, preds, out)
            print("player counts:",
                  {TEAM_NAMES[t]: n for t, n in sorted(counts.items())})
        return

    if args.train_images or args.test_images:
        splits: dict[str, str | Path] = {}
        if args.train_images:
            splits["Train"] = args.train_images
        if args.test_images:
            splits["Test"] = args.test_images
        generate_predictions_csv(args.model, splits, args.predictions_csv, args.threshold)
        return

    raise SystemExit(
        "Nothing to do: pass --image <path> or --train-images/--test-images for a CSV."
    )


if __name__ == "__main__":
    main()
