import pytest
import requests
import os

API_URL = "http://127.0.0.1:8083"  

TEST_IMAGE_PATH = "moh.png"  
TEST_PERSON_NAME = "Mohamed"

@pytest.fixture
def test_image():
    """Ensure the test image exists before running tests."""
    assert os.path.exists(TEST_IMAGE_PATH), f"Test image '{TEST_IMAGE_PATH}' not found!"
    return open(TEST_IMAGE_PATH, "rb")

def test_add_filters(test_image):
    """Test the /add-filters endpoint."""
    url = f"{API_URL}/add-filters"
    files = {"image": test_image}
    data = {"person_name": TEST_PERSON_NAME}

    response = requests.post(url, files=files, data=data)
    
    assert response.status_code == 200
    assert f"Filters applied for {TEST_PERSON_NAME}" in response.json()["message"]

def test_store_face_with_filters(test_image):
    """Test the /store endpoint with filters."""
    url = f"{API_URL}/store"
    files = {"image": test_image}
    data = {"person_name": TEST_PERSON_NAME, "apply_filter": "true"}  

    response = requests.post(url, files=files, data=data)

    assert response.status_code == 200
    assert f"Stored filtered images for {TEST_PERSON_NAME}" in response.json()["message"]

def test_store_face_with_filters(test_image):
    """Test the /store endpoint with filters."""
    url = f"{API_URL}/store"
    files = {"image": test_image}
    data = {"person_name": TEST_PERSON_NAME, "apply_filter": "true"}  

    response = requests.post(url, files=files, data=data)

    assert response.status_code == 200
    assert f"Stored filtered images for {TEST_PERSON_NAME}" in response.json()["message"]


def test_search_face(test_image):
    """Test the /search endpoint."""
    url = f"{API_URL}/search?k=5"
    files = {"image": test_image}

    response = requests.post(url, files=files)
    json_response = response.json()

    assert response.status_code == 200
    assert "Search completed" in json_response["message"]
    assert "matches" in json_response["result"]  
    assert isinstance(json_response["result"]["matches"], list)  
    assert len(json_response["result"]["matches"]) > 0  
    assert "image_path" in json_response["result"]["matches"][0]  



if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"])
