from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from face_utils import  recognize_face  
from face_detection import detect_faces

app = FastAPI()

# Load reference image (Moh)
reference_image = face_recognition.load_image_file("moh.png")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """Endpoint to detect faces in an uploaded image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(frame)
    return {"faces_detected": len(faces)}



