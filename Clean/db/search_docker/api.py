from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
from search_faces import search_face
import logging
"""
uvicorn api:app --host 0.0.0.0 --port 8085
docker build -t search_face .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/uploads:/uploads \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/images_with_filters:/images_with_filters \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/store_docker/fais_db:/fais_db \
    -p 8085:8085 \
    search_face
"""


def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"search.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger

app = FastAPI()
logger = logger_creation()

UPLOAD_DIR = "uploads"
FILTERED_DIR = "images_with_filters"
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
    logger.debug(f"[Search] Search completed. Result: \n{result}")
    return {"message": "Search completed", "result": result}