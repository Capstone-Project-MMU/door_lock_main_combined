import requests

API_URL = "http://127.0.0.1:8084"  # Change this if needed

def send_post_request(endpoint, image_path, data=None, params=None):
    """Sends a POST request with an image to a given endpoint."""
    with open(image_path, "rb") as file:
        files = {"image": file}
        response = requests.post(f"{API_URL}/{endpoint}", files=files, data=data, params=params)
    
    print(f"Response from {endpoint}: {response.status_code}")
    print(response.json())

# Provide the full path to the image
TEST_IMAGE_PATH = "moh.png"  # Replace with your image path
# TEST_PERSON_NAME = "Mohamed"
# files = {"image": open(TEST_IMAGE_PATH, "rb")}
# store_data = {"person_name": TEST_PERSON_NAME, "apply_filter": "true"}
# # add_filters_data = {"person_name": "Mohamed"}
# store_url = "http://127.0.0.1:8085/store"



# Test API endpoints
# send_post_request("add-filters", TEST_IMAGE_PATH, add_filters_data)
# send_post_request("store", TEST_IMAGE_PATH, data=data)
'''results in an internal server error because it can't open faiss_db/filtered_face_index.faiss 
so i left it alone. Does save correctly image correclty'''
# response = requests.post(store_url, files=files, data=store_data) 
send_post_request("search", TEST_IMAGE_PATH, params={"k": 5})

