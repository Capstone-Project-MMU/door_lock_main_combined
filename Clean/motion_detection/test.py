import requests

url = "http://127.0.0.1:8081/compare"

files = {
    "file1": open("image1.png", "rb"),
    "file2": open("image2.png", "rb"),
}

response = requests.post(url, files=files)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
