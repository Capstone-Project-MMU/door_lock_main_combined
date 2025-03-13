
from api import app
import requests

### uvicorn api:app --reload --host 127.0.0.1 --port 8080

test_image = "moh.png"
URL = "http://127.0.0.1:8080"

def test_detect():
    with open(test_image, "rb") as file:
        response = requests.post(f"{URL}/detect", files={"file": (test_image, file, "image/jpeg")})
    return response

def test_recognize():
    with open(test_image, "rb") as file:
        response = requests.post(f"{URL}/recognize", files={"file": (test_image, file, "image/jpeg")})
    return response

def test_add_filters():
    with open(test_image, "rb") as image:
        response = requests.post(f"{URL}/add-filters", files={"image": (test_image, image, "image/jpeg")}, data={"person_name": "John Doe"})
    return response

def test_store_face():
    with open(test_image, "rb") as image:
        response = requests.post(f"{URL}/store", files={"image": (test_image, image, "image/jpeg")}, data={"person_name": "John Doe", "apply_filter": "true"})
    return response

def test_search_face():
    with open(test_image, "rb") as image:
        response = requests.post(f"{URL}/search", files={"image": (test_image, image, "image/jpeg")}, data={"k": 5})
    return response

# test_add_filters()
# test_store_face()
# test_search_face()