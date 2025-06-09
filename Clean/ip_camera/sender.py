import time
import requests
import os

url = "http://localhost:8080/detect"
index = 0

def send_image(filepath):
    try:
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f, "image/jpeg")}  # ? key must be "file"
            response = requests.post(url, files=files)
            print(f"[{filepath}] Sent {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error sending {filepath}: {e}")


if __name__ == "__main__":
    while True:
        filename = f"image_{index % 3}.jpg"
        if os.path.exists(filename):
            send_image(filename)
        else:
            print(f"{filename} not found, skipping...")
        index += 1
        time.sleep(0.5)
