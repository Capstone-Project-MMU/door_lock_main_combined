import keras
import numpy as np
from fastapi import FastAPI
import logging
import os
from test import load_and_preprocess_image
"""
uvicorn api:app --host 0.0.0.0 --port 8087
docker build -t router .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/router_docker:/current \
    -p 8087:8087 \
    router
"""

MODEL_PATH = "current/mobilenetv2_human_animal_none_balanced.h5"
IMG_PATH = "current/image2.png"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["animal", "human", "none"]

def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"router.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger


model = keras.models.load_model(MODEL_PATH)
app = FastAPI()
logger = logger_creation()


@app.post("/router")
def router():
    image_input = load_and_preprocess_image(IMG_PATH)
    predictions = model.predict(image_input)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    logger.debug(f"[router] Prediction: {CLASS_NAMES[predicted_class]}, Confidence: {confidence:.2f}")
    return f"Prediction: {CLASS_NAMES[predicted_class]}, Confidence: {confidence:.2f})"