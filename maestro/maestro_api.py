'''before running the code, install pip and to install cmake it is recommended to install 
an official copy of cmake rather than using pip. here is the following methods to install cmake:
              - cmake.org (this is how windows users should get cmake)
              - apt install cmake (for Ubuntu or Debian based systems)
              - yum install cmake (for Redhat or CenOS based systems)
              '''

from fastapi import FastAPI, UploadFile, File, Form
import requests
from fastapi.responses import JSONResponse
import subprocess
import time
import os

app = FastAPI()

def start_service(service_directory, port):
    """Starts the FastAPI service using uvicorn."""
    os.chdir(service_directory)
    subprocess.Popen(["uvicorn", "api:app", "--reload", "--host", "127.0.0.1", "--port", str(port)])
    time.sleep(5)  # Wait for the server to start
    os.chdir("..")  # Go back to the original directory

def stop_service():
    """Stops the FastAPI service."""
    subprocess.run(["pkill", "-f", "uvicorn api:app"])

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # try:
    #     start_service("../recognition_docker", 8081)
    # except Exception as e:
    #     print("\n\n", e, "\n\n")
    # try:
        url = "http://127.0.0.1:8081/detect"
        url2 = "http://127.0.0.1:8082/recognize"

        # Read file into memory
        file_contents = await file.read()

        files = {"file": (file.filename, file_contents, file.content_type)}
        
        # Send the file to the first service
        time.sleep(2)
        response = requests.post(url, files=files)
        time.sleep(2)
        output = response.json()
        
        if output.get('faces_detected', 0) > 0:
            start_service("../recognition_docker", 8082)
            # Resend the file 
            files = {"file": (file.filename, file_contents, file.content_type)}
            time.sleep(2)
            response = requests.post(url2, files=files)
            time.sleep(2)

            # stop_service()

        print(response.json())
        return response.json()
    # finally:
    #     stop_service()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    # start_service("../recognition_docker", 8082)
    # try:
        url = "http://127.0.0.1:8082/recognize"
        files = {"file": (file.filename, file.file, file.content_type)}
        time.sleep(2)
        response = requests.post(url, files=files)
        time.sleep(2)
        print(response.json())
        return {"match": True}
    # finally:
    #     stop_service()

@app.post("/add-filters")
async def add_filters(image: UploadFile = File(...), person_name: str = Form(...)):
    """Forwards add-filters request to the add-filters FastAPI service."""
    start_service("../db/add_filter_docker", 8083)
    try:
        url = "http://127.0.0.1:8083/add-filters"
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"person_name": person_name}
        time.sleep(2)
        response = requests.post(url, files=files, data=data)
        time.sleep(2)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    finally:
        stop_service()

@app.post("/store")
async def store_face(
    image: UploadFile = File(...),
    person_name: str = Form(...),
    apply_filter: str = Form("false"),
):
    """Forwards store request to the store FastAPI service."""
    start_service("../db/store_docker", 8085)
    try:
        url = "http://127.0.0.1:8085/store"
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"person_name": person_name, "apply_filter": apply_filter}
        time.sleep(2)
        response = requests.post(url, files=files, data=data)
        time.sleep(2)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    finally:
        stop_service()

@app.post("/search")
async def search_face(image: UploadFile = File(...), k: int = Form(5)):
    """Forwards search request to the search FastAPI service."""
    start_service("../db/search_docker", 8084)
    try:
        url = "http://127.0.0.1:8084/search"
        files = {"image": (image.filename, await image.read(), image.content_type)}
        data = {"k": k}
        time.sleep(2)
        response = requests.post(url, files=files, data=data)
        time.sleep(2)
        return JSONResponse(content=response.json(), status_code=response.status_code)
    finally:
        stop_service()