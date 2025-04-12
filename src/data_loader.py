import pandas as pd
import numpy as np
from PIL import Image
import os

def load_and_preprocess_image(path, img_size=(256, 256)):
    image = Image.open(path).convert('RGB')
    image = image.resize(img_size)
    return np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]

def load_data(csv_path='data/fabfashion.csv', img_size=(256, 256)):
    df = pd.read_csv(csv_path)

    fabrics = []
    outfits = []

    for _, row in df.iterrows():
        # Paths relative to the project root, since data/ is the base folder
        fabric_path = os.path.join('data', row['FabricPath'])
        outfit_path = os.path.join('data', row['OutfitPath'])

        if os.path.exists(fabric_path) and os.path.exists(outfit_path):
            fabrics.append(load_and_preprocess_image(fabric_path, img_size))
            outfits.append(load_and_preprocess_image(outfit_path, img_size))
        else:
            print(f"❌ Missing file: {fabric_path} or {outfit_path}")

    return np.array(fabrics), np.array(outfits)
