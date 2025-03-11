import requests

API_URL = "http://127.0.0.1:8083"  # Change this if needed

def send_post_request(endpoint, image_path, params=None):
    """Sends a POST request with an image to a given endpoint."""
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}
        response = requests.post(f"{API_URL}/{endpoint}", files=files, params=params)
    
    print(f"Response from {endpoint}: {response.status_code}")
    print(response.json())

# Provide the full path to the image
image_path = "moh.png"  # Replace with your image path

# Test API endpoints
send_post_request("add-filters", image_path)
send_post_request("store", image_path)
send_post_request("search", image_path, params={"k": 5})
