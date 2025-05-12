from fastapi import FastAPI
import keras
import numpy as np
import logging
import os

from test import load_and_preprocess_image
#uvicorn api:app --host 0.0.0.0 --port 8088
#docker build -t animal_detect .
#docker run -p 8088:8088 animal_detect
# Note i couldn't run this file bc of the Tensorflow error 
# but i think this is right

MODEL_PATH = "mobilenetv2_animal.h5"
IMG_PATH = "image2.png"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["monkey", "cat", "squirrel", "snake"]

def logger_creation(name : str):
    log_dir = "door_lock_main_combined/Clean/logs"
    
    docker_logger = logging.getLogger(name)
    docker_logger.setLevel(logging.DEBUG)

    log_dir = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger


app = FastAPI()
model = keras.models.load_model(MODEL_PATH)
logger = logger_creation("animal")

@app.post("/animal_detect")
def animal_detect():
    input_image = load_and_preprocess_image(IMG_PATH)
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    logger.debug(f"[animal] Prediction: {CLASS_NAMES[predicted_class]}, Confidence: {confidence:.2f}")
    return f"Prediction: {CLASS_NAMES[predicted_class]}, Confidence: {confidence:.2f})"