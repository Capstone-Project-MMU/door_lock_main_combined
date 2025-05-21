import time
import requests
import cv2
import os
import picamera2 as pi

url = "http://localhost:8080/detect"

cam = pi.Picamera2()
config = cam.create_preview_configuration(main={"format" : "RGB888", "size" : (640, 840)})
cam.configure(config)
cam.start()

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
