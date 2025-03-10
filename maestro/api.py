from fastapi import FastAPI, UploadFile, File
import requests
#uvicorn api:app --host 0.0.0.0 --port 8080

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    url = "http://127.0.0.1:8081/detect"
    url2 = "http://127.0.0.1:8082/recognize"

    # Read file into memory
    file_contents = await file.read()

    files = {"file": (file.filename, file_contents, file.content_type)}
    
    # Send the file to the first service
    response = requests.post(url, files=files)
    output = response.json()
    
    if output.get('faces_detected', 0) > 0:
        # Resend the file 
        files = {"file": (file.filename, file_contents, file.content_type)}
        response = requests.post(url2, files=files)

    print(response.json())
    return response.json()

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    url = "http://127.0.0.1:8082/recognize"
    files = {"file": (file.filename, file.file, file.content_type)}
    response = requests.post(url, files=files)
    print(response.json())
    return {"match": True}