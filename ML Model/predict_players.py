import numpy as np
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
import os
import cv2
import warnings  # Added to manage clean stream outputs

from create_dataset import split_into_grid, extract_features

# ============================================================
# CONFIG
# ============================================================
IMAGE_SIZE = (800, 600)
GRID_ROWS = 8
GRID_COLS = 8
CELL_WIDTH = 100
CELL_HEIGHT = 75
NUM_CLASSES = 11

TRAIN_IMAGE_FOLDER = "../Create Labels/static/dataset"
LABEL_CSV = "../Create Labels/outputs/labels.csv"

MODEL_PATH_SVM = "model_svm.pkl"
MODEL_PATH_LR = "model_lr.pkl"    
MODEL_PATH_XGB = "model_xgb.pkl"
MODEL_PATH_RF = "model_rf.pkl"

TEAM_MAP = {
    0: "BG", 1: "CSK", 2: "DC", 3: "GT", 4: "KKR",
    5: "LSG", 6: "MI", 7: "PBKS", 8: "RR", 9: "RCB", 10: "SRH"
}

# ============================================================
# INFERENCE
# ============================================================

# Modifying function signature to accept the loaded model object instead of a string path
def predict_image(loaded_model_dict, image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [0] * 64
        
    image = cv2.resize(image, IMAGE_SIZE)
    cells = split_into_grid(image)

    predictions = []
    model = loaded_model_dict['model']
    label_encoder = loaded_model_dict.get('label_encoder', None)

    for cell in cells:
        features = extract_features(cell)
        features = features.reshape(1, -1)
        
        if label_encoder is not None:  # XGBoost path using pre-loaded encoder
            xgb_raw_preds = model.predict(features)
            pred = label_encoder.inverse_transform(xgb_raw_preds.astype(int))[0]
        else:                          # SVM, RF, LR paths
            pred = model.predict(features)[0]

        predictions.append(pred)

    return predictions

# ============================================================
# GENERATE CSV OUTPUT
# ============================================================

def generate_predictions_csv(model_path, image_folder, output_csv):
    print(f"Generating predictions for images in {image_folder} using model {model_path}...")
    
    # --------------------------------------------------------
    # CRITICAL FIX: Load the pickle artifact ONCE right here!
    # --------------------------------------------------------
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}. Skipping.")
        return

    with open(model_path, "rb") as f:
        raw_artifact = pickle.load(f)
        
    # Standardize structure into an operational dictionary
    if model_path == MODEL_PATH_XGB:
        loaded_model_dict = {
            'model': raw_artifact['pipeline'],
            'label_encoder': raw_artifact['label_encoder']
        }
    else:
        loaded_model_dict = {
            'model': raw_artifact,
            'label_encoder': None
        }

    rows = []
    image_files = os.listdir(image_folder)

    for image_name in image_files:
        image_path = os.path.join(image_folder, image_name)
        
        # Pass the pre-loaded dictionary instead of the filename string
        preds = predict_image(loaded_model_dict, image_path)

        row = {
            "Image File Name": image_name,
            "Train Or Test": "test"
        }

        for i in range(64):
            row[f"c{i+1}"] = int(preds[i])

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{model_path}_{output_csv}".replace(".pkl",""), index=False)
    print(f" Saved predictions completely to {model_path}_{output_csv}")


def count_players_in_huddle(grid_predictions, avg_player_cells=4):
    G = nx.Graph()
    rows, cols = grid_predictions.shape
    
    for r in range(rows):
        for c in range(cols):
            if grid_predictions[r, c] > 0:
                G.add_node((r, c), team=grid_predictions[r, c])
                
    for (r, c) in G.nodes():
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                
                if (nr, nc) in G.nodes():
                    if G.nodes[(r, c)]['team'] == G.nodes[(nr, nc)]['team']:
                        G.add_edge((r, c), (nr, nc))
                        
    total_players = 0
    subgraphs = list(nx.connected_components(G))
    cluster_info = {}
    
    for cluster in subgraphs:
        cluster_players = 0
        cluster_size = len(cluster)
        
        r_coords = [node[0] for node in cluster]
        c_coords = [node[1] for node in cluster]
        cluster_id = int(grid_predictions[r_coords[0], c_coords[0]])
        width = max(c_coords) - min(c_coords) + 1
        height = max(r_coords) - min(r_coords) + 1
        
        if cluster_size > avg_player_cells and (width >= height or width >= avg_player_cells):
            estimated_players = int(np.round(cluster_size / avg_player_cells))
            total_players += max(1, estimated_players)
            cluster_players += max(1, estimated_players)
        elif cluster_size == 1 or height == 1:
            total_players += 0
            cluster_players += 0
        else:
            total_players += 1
            cluster_players += 1
            
        if cluster_players > 0:
            if cluster_id in cluster_info:
                cluster_info[cluster_id] += cluster_players
            else:
                cluster_info[cluster_id] = cluster_players
                
    return total_players, cluster_info


if __name__ == "__main__":
    # Clean up standard parallel warnings streams safely if any lingering systems remain
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
    
    models = [MODEL_PATH_SVM, MODEL_PATH_XGB, MODEL_PATH_RF, MODEL_PATH_LR]
    
    for model_path in models:
        # Check if model exists before running pipeline steps
        if not os.path.exists(model_path):
            continue
            
        generate_predictions_csv(model_path, "../Create Labels/test_image", "predictions.csv")
        print("=" * 40)
        output_csv = "predictions.csv"
        predictions_path = Path(__file__).with_name(f"{model_path}_{output_csv}".replace(".pkl",""))
        if not predictions_path.exists():
            continue
            
        pred_df = pd.read_csv(predictions_path)

        for _, row in pred_df.iterrows():
            grid_values = row.filter(regex=r"^c\d+$").astype(int).to_numpy().reshape(8, 8)
            grid_predictions = np.asarray(grid_values, dtype=int).reshape(8, 8)
            player_count, cluster_info = count_players_in_huddle(grid_predictions)

            cluster_info = {TEAM_MAP[key]: value for key, value in cluster_info.items()}
            
            print(f"{row['Image File Name']} ({model_path}): counted players = {player_count}")
            print(f"Cluster information: {cluster_info}")
            print("-" * 40)