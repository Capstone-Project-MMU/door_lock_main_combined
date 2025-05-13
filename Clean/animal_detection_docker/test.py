import keras
import numpy as np
from PIL import Image
# import logging
# import os

MODEL_PATH = "mobilenetv2_animal.h5"
IMG_PATH = "image2.png"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["monkey", "cat", "squirrel", "snake"]

# logging testing, that's all. ignore the comments of this file

# def logger_creation():
#     log_dir = "test_log"
    
#     docker_logger = logging.getLogger()
#     docker_logger.setLevel(logging.DEBUG)
#     os.makedirs(log_dir, exist_ok=True)
#     log_dir = os.path.join(log_dir, f"animal_detection.log")
#     file_handler = logging.FileHandler(log_dir)
#     file_handler.setLevel(logging.DEBUG)
#     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     file_handler.setFormatter(formatter)

#     if not docker_logger.handlers: 
#         docker_logger.addHandler(file_handler)
    
#     return docker_logger

# logger = logger_creation()

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

# logger.debug(f"[router] Prediction: {CLASS_NAMES[predicted_class]}, Confidence: {confidence:.2f}")
print(f"Prediction: {CLASS_NAMES[predicted_class]} (confidence: {confidence:.2f})")
