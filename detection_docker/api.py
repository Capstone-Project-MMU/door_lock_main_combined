from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp

import numpy as np
from face_detection import detect_faces
#uvicorn api:app --host 0.0.0.0 --port 8081
#docker build -t face-detection-api .
#docker run -p 8081:8081 face-detection-api

app = FastAPI()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(frame)
    return {"faces_detected": len(faces)}



