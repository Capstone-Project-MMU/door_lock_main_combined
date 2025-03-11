from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import requests
from store_faces import store_face
#uvicorn api:app --host 0.0.0.0 --port 8083


app = FastAPI()

UPLOAD_DIR = "../uploads"
FILTERED_DIR = "../images_with_filters"
ADD_FILTER_URL = "http://127.0.0.1:8083/add-filters" ### Endpoint for applying filters
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

def save_uploaded_file(uploaded_file: UploadFile):
    """Saves uploaded image file temporarily."""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return file_path

def apply_filters_request(image_path, FILTERED_DIR, person_name):
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
    return requests.post(f"{ADD_FILTER_URL}", files=files, params=None)


@app.post("/store")
async def store_face_endpoint(
    image: UploadFile = File(...),
    person_name: str = Form(...),
    apply_filter: str = Form("false"),  # Store as a string and compare later
):
    """
    Stores a face image either directly or after applying filters.
    - If `apply_filter="true"`, it applies filters before storing.
    - If `apply_filter="false"`, it stores the original image.
    """
    image_path = save_uploaded_file(image)

    if apply_filter.lower() == "true":  
        apply_filters_request(image_path, FILTERED_DIR, person_name) ### I think this works will have to test
        filtered_images = [
            f for f in os.listdir(FILTERED_DIR) if f.startswith(person_name)
        ]
        if not filtered_images:
            return {"error": "No filtered images generated"}

        for filtered_img in filtered_images:
            store_face(os.path.join(FILTERED_DIR, filtered_img))
        return {"message": f"Stored filtered images for {person_name}"}

    else:
        store_face(image_path)
        return {"message": f"Stored original image for {person_name}"}
