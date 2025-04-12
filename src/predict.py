import tensorflow as tf
import numpy as np
from PIL import Image
import os

def predict(fabric_image, model_path='checkpoints/generator.h5', output_path='images/generated/output.png'):
    # Load the trained generator model
    model = tf.keras.models.load_model(model_path, compile=False)

    # Expand dims to match model input shape: (1, H, W, C)
    input_tensor = np.expand_dims(fabric_image, axis=0)

    # Generate prediction from model
    prediction = model(input_tensor, training=False).numpy()[0]

    # Rescale from [-1, 1] → [0, 1]
    prediction = (prediction + 1.0) / 2.0

    # Convert to uint8 image format
    prediction = np.clip(prediction * 255.0, 0, 255).astype(np.uint8)

    # Create output directory if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the output image
    Image.fromarray(prediction).save(output_path)
    print(f"✅ Generated outfit saved at: {output_path}")
