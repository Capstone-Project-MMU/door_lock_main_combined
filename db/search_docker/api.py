from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from search_faces import search_face
#uvicorn api:app --host 0.0.0.0 --port 8084


app = FastAPI()

UPLOAD_DIR = "../uploads"
FILTERED_DIR = "../images_with_filters"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file: UploadFile):
    """Saves uploaded image file temporarily."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return file_path


@app.post("/search")
async def search_face_endpoint(image: UploadFile = File(...), k: int = 5):
    """Finds the k nearest faces to an uploaded image."""
    image_path = save_uploaded_file(image)
    result = search_face(image_path, k)
    return {"message": "Search completed", "result": result}