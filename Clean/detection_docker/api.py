from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp
import logging

import numpy as np
from face_detection import detect_faces
#uvicorn api:app --host 0.0.0.0 --port 8082
#docker build -t face_detect .
#docker run -p 8082:8082 face_detect

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
logger = logger_creation("detection")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(frame)
    logger.debug(f"[detection] Faces Detected: {len(faces)}")
    return {"faces_detected": len(faces)}



