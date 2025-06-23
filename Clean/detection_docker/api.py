from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp
import logging
import os

import numpy as np
from face_detection import detect_faces

"""
uvicorn api:app --host 0.0.0.0 --port 8082
docker build -t face_detect .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -p 8082:8082 \
    face_detect
"""


app = FastAPI()
#logger = logger_creation()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(frame)
    #logger.debug(f"[detection] Faces Detected: {len(faces)}")
    return {"faces_detected": len(faces)}



