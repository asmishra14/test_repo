import json
import pandas as pd

ANNOTATION_FILE = "annotations.json"

OUTPUT_CSV = "outputs/labels.csv"

with open(ANNOTATION_FILE, "r") as f:

    annotations = json.load(f)

rows = []

for image_name, labels in annotations.items():

    row = {
        "image": image_name
    }

    for i in range(1, 65):

        cell_id = f"c{i}"

        row[cell_id] = labels.get(cell_id, 0)

    rows.append(row)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_CSV, index=False)

print("CSV exported successfully")