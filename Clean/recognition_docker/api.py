from fastapi import FastAPI, UploadFile, File
import cv2
import logging
import mediapipe as mp
import face_recognition
import numpy as np
from face_utils import recognize_face  
import requests
import os
"""
uvicorn api:app --host 0.0.0.0 --port 8083
docker build -t face_recognize .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -p 8083:8083 \
    face_recognize
"""
YOUR_PI_IP = "192.168.1.107"

def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"recognition.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger


app = FastAPI()
#logger = logger_creation()

# Load reference image (Moh)
reference_image = face_recognition.load_image_file("moh.png")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Endpoint to recognize if the detected face matches the reference image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    """Removal of this section and input of direct frame works"""
    # faces = detect_faces(frame) 

    # if not faces:
    #     return {"message": "No faces detected"}

    # x, y, width, height, face = faces[0]
    is_moh = bool(recognize_face(frame, reference_encoding))  # Convert numpy.bool_ to Python bool
    #logger.debug(f"[recognition] match : {is_moh} ")
    requests.post(f"http://{YOUR_PI_IP}:8000/send-word", json={"word": "Recognition: Complete -> Mohtady"})
    return {"match": is_moh}


