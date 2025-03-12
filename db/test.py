import requests

API_URL = "http://127.0.0.1:8085"  # Change this if needed

def send_post_request(endpoint, image_path, data):
    """Sends a POST request with an image to a given endpoint."""
    files = {"image": image_path}
    response = requests.post(f"{API_URL}/{endpoint}", files=files, data=data)
    
    print(f"Response from {endpoint}: {response.status_code}")
    print(response.json())

# Provide the full path to the image
# TEST_IMAGE_PATH = "moh.png"  # Replace with your image path
# TEST_PERSON_NAME = "Mohamed"
# files = {"image": TEST_IMAGE_PATH}
# data = {"person_name": TEST_PERSON_NAME}

# Test API endpoints
# send_post_request("add-filters", image_path)
# send_post_request("store", TEST_IMAGE_PATH, data=data)
url = "http://127.0.0.1:8085/store"
files = {"image": open("moh.png", "rb")}
data = {"person_name": "mohtady", "apply_filter": "false"}

response = requests.post(url, files=files, data=data)

print(response.text)
# send_post_request("search", image_path, params={"k": 5})

