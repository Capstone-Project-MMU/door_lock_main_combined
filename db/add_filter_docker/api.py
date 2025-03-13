from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from add_filters import apply_filters
#uvicorn api:app --host 0.0.0.0 --port 8083
#docker build -t face-rec-api .
#docker run -p 8083:8083 face-rec-api


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

@app.post("/add-filters")
async def add_filters_endpoint(image: UploadFile = File(...), person_name: str = Form(...)):
    """Applies filters and saves images with person's name."""
    image_path = save_uploaded_file(image)
    apply_filters(image_path, FILTERED_DIR, person_name)
    return {"message": f"Filters applied for {person_name}", "output_directory": FILTERED_DIR}
