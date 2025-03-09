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


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    """Endpoint to recognize if the detected face matches the reference image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    faces = detect_faces(frame) ### i think this line here will be fine since it can still run detect_faces() from face_detection.py

    if not faces:
        return {"message": "No faces detected"}

    x, y, width, height, face = faces[0]
    is_moh = bool(recognize_face(face, reference_encoding))  # Convert numpy.bool_ to Python bool

    return {"match": is_moh}

