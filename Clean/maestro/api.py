'''before running the code, install pip and to install cmake it is recommended to install 
an official copy of cmake rather than using pip. here is the following methods to install cmake:
              - cmake.org (this is how windows users should get cmake)
              - apt install cmake (for Ubuntu or Debian based systems)
              - yum install cmake (for Redhat or CenOS based systems)
              '''

from fastapi import FastAPI, UploadFile, File, Form
import requests
from fastapi.responses import JSONResponse
import logging
import time
import os

""""
uvicorn api:app --host 0.0.0.0 --port 8080 --log-config log.ini
"""


def logger_creation():
    log_dir = "../logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"maestro.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger


app = FastAPI()
logger = logger_creation()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    url = "http://127.0.0.1:8082/detect"
    url2 = "http://127.0.0.1:8083/recognize"

    # Read file into memory
    file_contents = await file.read()

    files = {"file": (file.filename, file_contents, file.content_type)}
    
    # Send the file to the first service
    response = requests.post(url, files=files)
    output = response.json()
    
    if output.get('faces_detected', 0) > 0:
        files = {"file": (file.filename, file_contents, file.content_type)}
        response = requests.post(url2, files=files)

    print(response.json())
    logger.debug(f"[maestro] Response from /detect has been recieved")
    return response.json()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    url = "http://127.0.0.1:8083/recognize"
    files = {"file": (file.filename, file.file, file.content_type)}
    time.sleep(2)
    response = requests.post(url, files=files)
    time.sleep(2)
    print(response.json())
    logger.debug(f"[maestro] Response from /recognize has been recieved")
    return {"match": True}


@app.post("/add-filters")
async def add_filters(image: UploadFile = File(...), person_name: str = Form(...)):
    """Forwards add-filters request to the add-filters FastAPI service."""
    url = "http://127.0.0.1:8086/add-filters"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"person_name": person_name}
    response = requests.post(url, files=files, data=data)
    logger.debug(f"[maestro] Response from /add_filters has been recieved")
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/store")
async def store_face(
    image: UploadFile = File(...),
    person_name: str = Form(...),
    apply_filter: str = Form("false"),
):
    """Forwards store request to the store FastAPI service."""
    url = "http://127.0.0.1:8084/store"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"person_name": person_name, "apply_filter": apply_filter}
    response = requests.post(url, files=files, data=data)
    logger.debug(f"[maestro] Response from /store has been recieved")
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/search")
async def search_face(image: UploadFile = File(...), k: int = Form(5)):
    """Forwards search request to the search FastAPI service."""
    url = "http://127.0.0.1:8085/search"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"k": k}
    response = requests.post(url, files=files, data=data)
    logger.debug(f"[maestro] Response from /search has been recieved")
    return JSONResponse(content=response.json(), status_code=response.status_code)
