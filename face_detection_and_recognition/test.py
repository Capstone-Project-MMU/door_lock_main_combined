import requests

url_detect = "http://127.0.0.1:8000/detect"
url_recognize = "http://127.0.0.1:8000/recognize"

# Load image
file_path = "test_image1.png"
files = {"file": open(file_path, "rb")}

# Test Face Detection
response_detect = requests.post(url_detect, files=files)
print("Face Detection Response:", response_detect.json())

# Test Face Recognition
files = {"file": open(file_path, "rb")}
response_recognize = requests.post(url_recognize, files=files)
print("Face Recognition Response:", response_recognize.json())
