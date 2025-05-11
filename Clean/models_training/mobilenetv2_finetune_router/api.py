import keras
import numpy as np
from fastapi import FastAPI

from test import load_and_preprocess_image
#uvicorn api:app --host 0.0.0.0 --port 8087
#docker build -t router .
#docker run -p 8087:8087 router
# Note i couldn't run this file bc of the Tensorflow error 
# but i think this is right

MODEL_PATH = "mobilenetv2_human_animal_none_balanced.h5"
IMG_PATH = "image2.png"
IMG_SIZE = (160, 160)
CLASS_NAMES = ["animal", "human", "none"]

model = keras.models.load_model(MODEL_PATH)
app = FastAPI()

@app.post("/router")
def router():
    image_input = load_and_preprocess_image(IMG_PATH)
    predictions = model.predict(image_input)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    print(f"""Prediction: {CLASS_NAMES[predicted_class]} 
          (confidence: {confidence:.2f})""")