from fastapi import FastAPI, UploadFile, File
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
from face_utils import recognize_face  
#uvicorn api:app --host 0.0.0.0 --port 8082

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
    """Removal of this section and input of direct frame works"""
    # faces = detect_faces(frame) 

    # if not faces:
    #     return {"message": "No faces detected"}

    # x, y, width, height, face = faces[0]
    is_moh = bool(recognize_face(frame, reference_encoding))  # Convert numpy.bool_ to Python bool

    return {"match": is_moh}


