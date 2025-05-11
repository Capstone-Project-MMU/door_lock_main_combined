from fastapi import FastAPI
import keras
import numpy as np

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

app = FastAPI()
model = keras.models.load_model(MODEL_PATH)

@app.post("/animal_detect")
def animal_detect():
    input_image = load_and_preprocess_image(IMG_PATH)
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    print(f"""Prediction: {CLASS_NAMES[predicted_class]} 
          (confidence: {confidence:.2f})""")