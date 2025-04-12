from src.data_loader import load_and_preprocess_image
from src.predict import predict

# Use a real test fabric (already used in training is fine for now)
fabric_img = load_and_preprocess_image("test.jpg")

predict(fabric_img)

