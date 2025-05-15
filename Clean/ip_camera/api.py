from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import logging 
import os

from ip_camera import generate_images

"""
Note: I actually can't test if docker works because i am running this docker on macOS which
does allow for camera access to docker no matter what. Though it important to note this code 
works outside of docker
uvicorn api:app --host 0.0.0.0 --port 8079
docker build -t camera .
docker run \
    -v /Users/omarsalahwork/Documents/Codes/Capstone/door_lock_main_combined/Clean/logs:/logs \
    -p 8079:8079 \
    camera
"""

def logger_creation():
    log_dir = "logs"
    
    docker_logger = logging.getLogger()
    docker_logger.setLevel(logging.DEBUG)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, f"camera.log")
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    if not docker_logger.handlers: 
        docker_logger.addHandler(file_handler)
    
    return docker_logger

        
camera = cv2.VideoCapture(0)
app = FastAPI()
logger = logger_creation()
        
@app.get("/camera")
def video_feed():
    return StreamingResponse(generate_images(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")
    