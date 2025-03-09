from fastapi import FastAPI, UploadFile, File
import requests

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    url = "127.0.0.1:8081/detect"
    response = requests.post(url, file)
    print(response.json())
    return {"faces_detected": 1}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    url = "127.0.0.1:8081/recognize"
    response = requests.post(url, file)
    print(response.json())
    return {"match": True}