from PIL import Image
import pillow_avif
import os

INPUT_FOLDER = "static/raw"
OUTPUT_FOLDER = "static/dataset"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):

    path = os.path.join(INPUT_FOLDER, file)

    try:

        img = Image.open(path).convert("RGB")

        img = img.resize((800, 600))

        output_name = os.path.splitext(file)[0] + ".jpg"

        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_name
        )

        img.save(output_path, quality=95)

        print("Converted:", file)

    except Exception as e:

        print("Failed:", file, e)