from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import logging 
import os
import picamera2 as pi
import time
import requests

from ip_camera import generate_images

"""
Note: After alot of testing, this file doesn't work in docker due to camera access issues. 
uvicorn api:app --host 0.0.0.0 --port 8079 --log-config log.ini
"""

def logger_creation():
    log_dir = "../logs"
    
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

def run():
	try: 
		while True: 
			images = []
			
			for i in range(3):
				frame = cam.capture_array()
				
				filename = f"image_{i}.jpg"
				cv2.imwrite(filename, frame)
				images.append(filename)
				
				time.sleep(0.25)
				
			try: 
				files = [("file", (img, open(img, "rb"), "image/jpeg")) for img in images]
				response = requests.post(url, files=files)
				print(f"Server response: {response.status_code} - {response.text}")
			except Exception as e: 
				print(f"Error sending images: {e}")
			finally: 
				for img in images: 
					os.remove(img)

	except KeyboardInterrupt: 
		print("Interrupted by user. Exiting...")
		
	finally: 
		cam.stop()
		cv2.destroyAllWindows()

url = "http://localhost:8080/router"

cam = pi.Picamera2()
config = cam.create_preview_configuration(main={"format" : "RGB888", "size" : (640, 840)})
cam.configure(config)
cam.start()
app = FastAPI()
logger = logger_creation()
        
run()
@app.get("/camera")
def video_feed():
    return StreamingResponse(generate_images(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")
