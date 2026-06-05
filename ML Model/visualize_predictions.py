import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

TEAM_MAP = {
    0: "BG",
    1: "CSK",
    2: "DC",
    3: "GT",
    4: "KKR",
    5: "LSG",
    6: "MI",
    7: "PBKS",
    8: "RR",
    9: "RCB",
    10: "SRH"
}

CELL_W = 100
CELL_H = 75

def visualize_prediction(image_path, prediction_row):

    image = cv2.imread(image_path)

    image = cv2.resize(image, (800, 600))

    # Draw grid
    for i in range(9):

        cv2.line(
            image,
            (i * CELL_W, 0),
            (i * CELL_W, 600),
            (0, 255, 0),
            1
        )

        cv2.line(
            image,
            (0, i * CELL_H),
            (800, i * CELL_H),
            (0, 255, 0),
            1
        )

    # Draw predictions
    for idx in range(64):

        label = int(prediction_row[f'c{idx+1}'])

        row = idx // 8
        col = idx % 8

        x = col * CELL_W + 25
        y = row * CELL_H + 40

        team_name = TEAM_MAP[label]

        cv2.putText(
            image,
            team_name,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

pred_df = pd.read_csv("predictions.csv")
# Visualize the all prediction
for _, row in pred_df.iterrows():
    image_path = os.path.join("../Create Labels/test_image", row["Image File Name"])
    visualize_prediction(
        image_path,
    row     
)