from fastapi import FastAPI, UploadFile, File, HTTPException
from motion_algorithm import *
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io
import logging
import os
"""
uvicorn api:app --host 0.0.0.0 --port 8081
docker build -t motion_detect .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -p 8081:8081 \
    motion_detect
"""


def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"motion_detection.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger

app = FastAPI()
logger = logger_creation()

def read_image(file: UploadFile) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("L")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    image1 = read_image(file1)
    image2 = read_image(file2)
    is_different = calculate_difference(image1, image2)
    #turns out np.bool is not the same as python's normal bool so JSON coudln't deal with it
    logger.debug(f"[motion_detection] is_different : {bool(is_different)}")
    return JSONResponse(content={"is_different": bool(is_different)})