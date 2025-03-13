import pytest
import httpx
from fastapi.testclient import TestClient
from maestro.api import app

client = TestClient(app)

test_image = "moh.png"

def test_detect():
    with open(test_image, "rb") as file:
        response = client.post("/detect", files={"file": (test_image, file, "image/jpeg")})
    assert response.status_code == 200
    assert "faces_detected" in response.json()

def test_recognize():
    with open(test_image, "rb") as file:
        response = client.post("/recognize", files={"file": (test_image, file, "image/jpeg")})
    assert response.status_code == 200
    assert "match" in response.json()

def test_add_filters():
    with open(test_image, "rb") as image:
        response = client.post("/add-filters", files={"image": (test_image, image, "image/jpeg")}, data={"person_name": "John Doe"})
    assert response.status_code == 200
    assert "message" in response.json()

def test_store_face():
    with open(test_image, "rb") as image:
        response = client.post("/store", files={"image": (test_image, image, "image/jpeg")}, data={"person_name": "John Doe", "apply_filter": "false"})
    assert response.status_code == 200
    assert "message" in response.json()

def test_search_face():
    with open(test_image, "rb") as image:
        response = client.post("/search", files={"image": (test_image, image, "image/jpeg")}, data={"k": 5})
    assert response.status_code == 200
    assert "result" in response.json()