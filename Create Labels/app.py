from flask import Flask, render_template, request, jsonify
import os
import json
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

app = Flask(__name__)

# ============================================================
# TEAM LABELS & COLOR-BASED AUTO-RECOMMEND
# ============================================================

TEAM_MAP = {
    'CSK': 1, 'DC': 2, 'GT': 3, 'KKR': 4, 'LSG': 5,
    'MI': 6, 'PBKS': 7, 'RR': 8, 'RCB': 9, 'SRH': 10
}

# Cache trained per-matchup models so we don't retrain every click
matchup_models = {}


def parse_teams_from_filename(filename):
    """Extract the two team IDs from a filename like 'DCvsRR_image_0.jpg'."""
    name = filename.split('_image_')[0] if '_image_' in filename else filename.rsplit('_', 1)[0]
    name = name.upper()
    for sep in ['VS', '_VS_']:
        if sep in name:
            parts = name.split(sep)
            if len(parts) == 2:
                t1 = parts[0].strip('_')
                t2 = parts[1].strip('_')
                return TEAM_MAP.get(t1), TEAM_MAP.get(t2)
    return None, None


def get_matchup_key(team1, team2):
    """Canonical key for a matchup regardless of order."""
    return f"{min(team1,team2)}vs{max(team1,team2)}"


def extract_cell_features(hsv_cell):
    """
    Extract robust HSV histogram + color-ratio features from a single cell.
    Returns a 1D numpy array of features.
    """
    h = hsv_cell[:, :, 0].ravel()
    s = hsv_cell[:, :, 1].ravel()
    v = hsv_cell[:, :, 2].ravel()
    total = len(h)

    # H histogram (18 bins, each covers 10° of hue)
    h_hist, _ = np.histogram(h, bins=18, range=(0, 180))
    h_hist = h_hist / (h_hist.sum() + 1e-8)

    # S histogram (8 bins)
    s_hist, _ = np.histogram(s, bins=8, range=(0, 256))
    s_hist = s_hist / (s_hist.sum() + 1e-8)

    # V histogram (8 bins)
    v_hist, _ = np.histogram(v, bins=8, range=(0, 256))
    v_hist = v_hist / (v_hist.sum() + 1e-8)

    # Semantic color-region percentages (IPL jersey palette)
    green_pct  = ((h >= 35) & (h <= 85) & (s > 30) & (v > 20)).sum() / total
    blue_pct   = ((h >= 90) & (h <= 140) & (s > 40) & (v > 30)).sum() / total
    red_pct    = (((h <= 10) | (h >= 170)) & (s > 40) & (v > 30)).sum() / total
    pink_pct   = ((h >= 145) & (h <= 175) & (s > 30) & (v > 60)).sum() / total
    orange_pct = ((h >= 10) & (h <= 25) & (s > 60) & (v > 60)).sum() / total
    yellow_pct = ((h >= 20) & (h <= 35) & (s > 60) & (v > 60)).sum() / total
    teal_pct   = ((h >= 80) & (h <= 100) & (s > 40) & (v > 30)).sum() / total
    purple_pct = ((h >= 130) & (h <= 160) & (s > 30) & (v > 30)).sum() / total
    dark_pct   = (v < 50).sum() / total
    bright_low_sat = ((s < 40) & (v > 150)).sum() / total

    stats = [
        np.mean(s) / 255.0, np.std(s) / 255.0,
        np.mean(v) / 255.0, np.std(v) / 255.0,
        np.mean(h) / 180.0, np.std(h) / 180.0,
        green_pct, blue_pct, red_pct, pink_pct,
        orange_pct, yellow_pct, teal_pct, purple_pct,
        dark_pct, bright_low_sat,
    ]

    return np.concatenate([h_hist, s_hist, v_hist, stats])


def train_matchup_model(matchup_key, team1, team2):
    """
    Train a 2-stage model for a specific matchup:
      Stage 1 (detector): binary RF — is this cell a jersey or background?
      Stage 2 (team):     RF — which team's jersey is it?

    Training data:
      - POSITIVE = cells explicitly labeled with a team number (1-10)
      - NEGATIVE = cells NOT labeled in annotated images (subsampled
        to 2× positives so the detector stays balanced)
    """
    X_pos, X_neg = [], []
    y_team = []

    for img_name, cells in annotations.items():
        t1, t2 = parse_teams_from_filename(img_name)
        if t1 is None or t2 is None:
            continue
        key = get_matchup_key(t1, t2)
        if key != matchup_key:
            continue
        if len(cells) == 0:
            continue

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        labeled_cells = set(cells.keys())

        for idx in range(64):
            row = idx // 8
            col = idx % 8
            cell = hsv[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT,
                        col * CELL_WIDTH:(col + 1) * CELL_WIDTH]
            feat = extract_cell_features(cell)
            cell_id = f"c{idx + 1}"

            if cell_id in labeled_cells:
                label = cells[cell_id]
                if label != 0:
                    X_pos.append(feat)
                    y_team.append(label)
                else:
                    X_neg.append(feat)
            else:
                X_neg.append(feat)

    if len(X_pos) == 0:
        return None

    # Subsample negatives to at most 2× positives for balanced detection
    rng = np.random.RandomState(42)
    max_neg = min(len(X_neg), len(X_pos) * 2)
    neg_idx = rng.choice(len(X_neg), max_neg, replace=False)
    X_neg_sub = [X_neg[i] for i in neg_idx]

    # Stage 1: binary detector
    X_bin = np.array(X_pos + X_neg_sub)
    y_bin = np.array([1] * len(X_pos) + [0] * len(X_neg_sub))
    clf_detect = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    clf_detect.fit(X_bin, y_bin)

    # Stage 2: team classifier (only on jersey cells)
    clf_team = RandomForestClassifier(
        n_estimators=150, random_state=42, n_jobs=-1
    )
    clf_team.fit(np.array(X_pos), np.array(y_team))

    return {"detector": clf_detect, "team": clf_team}


def predict_image_cells(model_dict, hsv_img):
    """
    Run 2-stage prediction on all 64 cells of an HSV image.
    Returns dict of {cell_id: team_label} for detected jersey cells only.
    """
    clf_detect = model_dict["detector"]
    clf_team = model_dict["team"]

    X_all = []
    for idx in range(64):
        row = idx // 8
        col = idx % 8
        cell = hsv_img[row * CELL_HEIGHT:(row + 1) * CELL_HEIGHT,
                        col * CELL_WIDTH:(col + 1) * CELL_WIDTH]
        X_all.append(extract_cell_features(cell))

    X_all = np.array(X_all)

    # Stage 1: detect jerseys
    detected = clf_detect.predict(X_all)

    # Stage 2: classify team for detected cells
    result = {}
    jersey_indices = [i for i, d in enumerate(detected) if d == 1]
    if jersey_indices:
        X_jersey = X_all[jersey_indices]
        teams = clf_team.predict(X_jersey)
        for i, team in zip(jersey_indices, teams):
            result[f"c{i + 1}"] = int(team)

    return result


IMAGE_FOLDER = "static/dataset/"

ANNOTATION_FILE = "annotations.json"

GRID_ROWS = 8
GRID_COLS = 8

CELL_WIDTH = 100
CELL_HEIGHT = 75

# ============================================================
# LOAD IMAGES
# ============================================================

image_files = sorted(os.listdir(IMAGE_FOLDER))

current_index = 0

# ============================================================
# LOAD EXISTING ANNOTATIONS
# ============================================================

if os.path.exists(ANNOTATION_FILE):

    with open(ANNOTATION_FILE, "r") as f:
        annotations = json.load(f)

else:
    annotations = {}

# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    global current_index, image_files

    # Load/reload image files dynamically so new conversions appear without server restart
    image_files = sorted(os.listdir(IMAGE_FOLDER))

    if not image_files:
        return "<h1>No images found in static/dataset!</h1><p>Please run the image conversion script first to convert images from <code>static/raw</code> to <code>static/dataset</code>:<br><code>python image_convert.py</code></p>", 404

    # Bound check the index
    if current_index >= len(image_files):
        current_index = 0
    elif current_index < 0:
        current_index = len(image_files) - 1

    image_name = image_files[current_index]

    if image_name not in annotations:

        annotations[image_name] = {}

    total_images = len(image_files)
    annotated_count = sum(1 for img in image_files if len(annotations.get(img, {})) > 0)
    current_image_num = current_index + 1

    # Count the distribution of labels across all images in the dataset
    label_counts = {}
    total_labels_count = 0
    for img in image_files:
        img_ann = annotations.get(img, {})
        for label in img_ann.values():
            label_counts[label] = label_counts.get(label, 0) + 1
            total_labels_count += 1

    # Sort label counts naturally by label
    sorted_label_counts = sorted(label_counts.items(), key=lambda x: str(x[0]))

    return render_template(
        "index.html",
        image_name=image_name,
        annotations=annotations[image_name],
        total_images=total_images,
        annotated_count=annotated_count,
        current_image_num=current_image_num,
        label_counts=sorted_label_counts,
        total_labels_count=total_labels_count
    )

# ============================================================
# SAVE ANNOTATION
# ============================================================

@app.route("/annotate", methods=["POST"])
def annotate():

    data = request.json

    image_name = data["image_name"]

    cell_id = data["cell_id"]

    label = data["label"]

    if image_name not in annotations:

        annotations[image_name] = {}

    annotations[image_name][cell_id] = label

    with open(ANNOTATION_FILE, "w") as f:

        json.dump(annotations, f, indent=4)

    return jsonify({"status": "success"})

# ============================================================
# AUTO RECOMMEND
# ============================================================

@app.route("/auto_recommend", methods=["POST"])
def auto_recommend():
    data = request.json
    image_name = data.get("image_name")

    team1, team2 = parse_teams_from_filename(image_name)
    if team1 is None or team2 is None:
        return jsonify({
            "status": "error",
            "message": "Could not detect teams from filename. "
                       "Expected format: TEAMAvsTEAMB_image_N.jpg"
        }), 400

    matchup_key = get_matchup_key(team1, team2)

    if matchup_key not in matchup_models:
        print(f"[Auto-Recommend] Training 2-stage model for {matchup_key} ...")
        model = train_matchup_model(matchup_key, team1, team2)
        if model is None:
            return jsonify({
                "status": "error",
                "message": f"No annotated data for matchup {matchup_key}. "
                           "Manually annotate a few images first!"
            }), 400
        matchup_models[matchup_key] = model
        print(f"[Auto-Recommend] Model ready.")

    img_path = os.path.join(IMAGE_FOLDER, image_name)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"status": "error", "message": "Image not found."}), 404
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    predictions = predict_image_cells(matchup_models[matchup_key], hsv)

    if image_name not in annotations:
        annotations[image_name] = {}

    for cell_id, team in predictions.items():
        annotations[image_name][cell_id] = team

    with open(ANNOTATION_FILE, "w") as f:
        json.dump(annotations, f, indent=4)

    return jsonify({
        "status": "success",
        "recommended": len(predictions),
        "message": f"Labeled {len(predictions)} jersey cells."
    })

# ============================================================
# NEXT IMAGE
# ============================================================

@app.route("/next")
def next_image():

    global current_index

    current_index += 1

    if current_index >= len(image_files):

        current_index = 0

    return jsonify({"success": True})

# ============================================================
# PREVIOUS IMAGE
# ============================================================

@app.route("/prev")
def prev_image():

    global current_index

    current_index -= 1

    if current_index < 0:

        current_index = len(image_files) - 1

    return jsonify({"success": True})

# ============================================================
# GOTO IMAGE
# ============================================================

@app.route("/goto", methods=["POST"])
def goto_image():
    global current_index
    data = request.json
    idx = data.get("index", 0)
    current_index = max(0, min(idx, len(image_files) - 1))
    return jsonify({"success": True})

# ============================================================
# AUTO RECOMMEND ALL UNANNOTATED
# ============================================================

@app.route("/auto_recommend_all", methods=["POST"])
def auto_recommend_all():
    """Batch auto-recommend all unannotated images in the dataset."""
    processed = 0
    skipped = 0
    errors = []

    for img_name in image_files:
        if img_name in annotations and len(annotations[img_name]) > 0:
            skipped += 1
            continue

        team1, team2 = parse_teams_from_filename(img_name)
        if team1 is None or team2 is None:
            errors.append(f"{img_name}: could not parse teams")
            continue

        matchup_key = get_matchup_key(team1, team2)

        if matchup_key not in matchup_models:
            model = train_matchup_model(matchup_key, team1, team2)
            if model is None:
                errors.append(f"{img_name}: no training data for {matchup_key}")
                continue
            matchup_models[matchup_key] = model

        img_path = os.path.join(IMAGE_FOLDER, img_name)
        img = cv2.imread(img_path)
        if img is None:
            errors.append(f"{img_name}: could not read image")
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        predictions = predict_image_cells(matchup_models[matchup_key], hsv)

        if img_name not in annotations:
            annotations[img_name] = {}

        for cell_id, team in predictions.items():
            annotations[img_name][cell_id] = team

        processed += 1

    with open(ANNOTATION_FILE, "w") as f:
        json.dump(annotations, f, indent=4)

    return jsonify({
        "status": "success",
        "processed": processed,
        "skipped": skipped,
        "errors": len(errors),
        "error_details": errors[:10],
        "message": f"Done! Processed {processed} images, skipped {skipped} already-annotated."
    })

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(debug=True)