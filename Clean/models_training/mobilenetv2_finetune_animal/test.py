import keras
import numpy as np
from PIL import Image

MODEL_PATH = "cat_best_model.h5"
IMG_PATH = "IMG_6127.JPG"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["cat"]

model = keras.models.load_model(MODEL_PATH)

def load_and_preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.expand_dims(image, axis=0)
    return image

input_data = load_and_preprocess_image(IMG_PATH)

predictions = model.predict(input_data)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])

print(f"Prediction: {CLASS_NAMES[predicted_class]} (confidence: {confidence:.2f})")
