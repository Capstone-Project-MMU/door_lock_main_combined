from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from add_filters import apply_filters
import logging, sys
#uvicorn api:app --host 0.0.0.0 --port 8086
#docker build -t add_filters .
#docker run -p 8086:8086 add_filters


app = FastAPI()

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)


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
    logging.debug(f"[add-filters] Received image: {image.filename} for person: {person_name}")
    image_path = save_uploaded_file(image)
    apply_filters(image_path, FILTERED_DIR, person_name)
    logging.debug(f"[add-filters] Filters applied and saved to {FILTERED_DIR} for {person_name}")
    return {"message": f"Filters applied for {person_name}", "output_directory": FILTERED_DIR}
