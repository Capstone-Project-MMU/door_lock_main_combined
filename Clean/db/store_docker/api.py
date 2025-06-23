from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os
import requests
import subprocess
from store_faces import store_face
import uvicorn
from fastapi import APIRouter
from time import sleep
import logging

"""
uvicorn api:app --host 0.0.0.0 --port 8084
docker build -t store_face .
docker run \
  -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
  -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/uploads:/uploads \
  -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/images_with_filters:/images_with_filters \
  -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/store_docker/fais_db:/fais_db \
  -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/db/add_filter_docker:/add_filter_docker \
  -p 8084:8084 \
  store_face
  """



def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"store.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger

app = FastAPI()
# logger = logger_creation()

UPLOAD_DIR = "../uploads" # needed to be fixed for mounting in dockers
FILTERED_DIR = "../images_with_filters"
ADD_FILTER_URL = "http://127.0.0.1:8083/add-filters"
SHUTDOWN_URL = "http://127.0.0.1:8083/shutdown"
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
        data = {"person_name": person_name}
        response = requests.post(f"{ADD_FILTER_URL}", files=files, data=data)
    return response

@app.post("/store")
async def store_face_endpoint(
    image: UploadFile = File(...),
    person_name: str = Form(...),
    apply_filter: str = Form("false"),
):
    """
    Stores a face image either directly or after applying filters.
    - If `apply_filter="true"`, it applies filters before storing.
    - If `apply_filter="false"`, it stores the original image.
    """  
    image_path = save_uploaded_file(image)
    logger.debug(f"[store] apply_filter: {apply_filter}")
    print(f"apply_filter: {apply_filter}")  # Debugging statement

    if apply_filter.lower() == "true":
        # Start the apply_filter endpoint server
        subprocess.Popen(
            ["python3", "start_apply_filter_server.py"],
            cwd="add_filter_docker"
        )

        try:
            sleep(5)  # Wait for the server to start
            response = apply_filters_request(image_path, FILTERED_DIR, person_name)
            if response.status_code != 200:
                logger.debug(f"[store] Error: Failed to apply filters: {response.text}")
                return {"error": f"Failed to apply filters: {response.text}"}
        except Exception as e:
            logger.debug(f"[store] Error in apply_filters_request: {e}")
            print(f"Error in apply_filters_request: {e}")  # Debugging statement
            return {"error": str(e)}

        filtered_images = [
            f for f in os.listdir(FILTERED_DIR) if f.startswith(person_name)
        ]
        if not filtered_images:
            return {"error": "No filtered images generated"}

        for filtered_img in filtered_images:
            store_face(os.path.join(FILTERED_DIR, filtered_img))

        # Shutdown the apply_filter endpoint server
        requests.post(SHUTDOWN_URL)
        logger.debug(f"[store] Stored filtered images for {person_name}")
        return {"message": f"Stored filtered images for {person_name}"}

    else:
        store_face(image_path)
        logger.debug(f"[store] Stored original images for {person_name}")
        return {"message": f"Stored original image for {person_name}"}

@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    shutdown_event = app.router.shutdown_event
    if shutdown_event:
        await shutdown_event()
