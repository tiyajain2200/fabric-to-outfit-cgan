import pandas as pd
import numpy as np
from PIL import Image
import os
import tensorflow as tf

def load_and_preprocess_image(path, img_size=(256, 256), augment=True):
    image = Image.open(path).convert('RGB').resize(img_size)
    image = (np.array(image, dtype=np.float32) / 127.5) - 1  # Normalize to [-1, 1]

    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image += tf.random.normal(shape=image.shape, mean=0.0, stddev=0.02)

    return image

def load_data(csv_path='data/fabfashion.csv', img_size=(256, 256)):
    df = pd.read_csv(csv_path)

    fabrics = []
    outfits = []

    for _, row in df.iterrows():
        fabric_path = os.path.join('data', row['FabricPath'])
        outfit_path = os.path.join('data', row['OutfitPath'])

        if os.path.exists(fabric_path) and os.path.exists(outfit_path):
            fabrics.append(load_and_preprocess_image(fabric_path, img_size))
            outfits.append(load_and_preprocess_image(outfit_path, img_size))
        else:
            print(f"❌ Missing file: {fabric_path} or {outfit_path}")

    return np.array(fabrics), np.array(outfits)

