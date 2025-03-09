import cv2
import time
import requests
import os

# Define the endpoint
url = "http://localhost:8080/detect"

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    images = []

    # Capture 3 images with a 0.25s delay
    for i in range(3):
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            continue
        
        filename = f"image_{i}.jpg"
        cv2.imwrite(filename, frame)
        images.append(filename)

        time.sleep(0.25)

    # Send images to the server
    try:
        files = [("images", (img, open(img, "rb"), "image/jpeg")) for img in images]
        response = requests.post(url, file=files)

        print(f"Server response: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"Error sending images: {e}")

    finally:
        # Clean up images after sending
        for img in images:
            os.remove(img)
