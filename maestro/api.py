from fastapi import FastAPI, UploadFile, File, Form
import requests
from fastapi.responses import JSONResponse

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

# New endpoints
@app.post("/add-filters")
async def add_filters(image: UploadFile = File(...), person_name: str = Form(...)):
    """Forwards add-filters request to the add-filters Docker service."""
    url = "http://127.0.0.1:8083/add-filters"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"person_name": person_name}
    response = requests.post(url, files=files, data=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/store")
async def store_face(
    image: UploadFile = File(...),
    person_name: str = Form(...),
    apply_filter: str = Form("false"),
):
    """Forwards store request to the store Docker service."""
    url = "http://127.0.0.1:8085/store"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"person_name": person_name, "apply_filter": apply_filter}
    response = requests.post(url, files=files, data=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/search")
async def search_face(image: UploadFile = File(...), k: int = Form(5)):
    """Forwards search request to the search Docker service."""
    url = "http://127.0.0.1:8084/search"
    files = {"image": (image.filename, await image.read(), image.content_type)}
    data = {"k": k}
    response = requests.post(url, files=files, data=data)
    return JSONResponse(content=response.json(), status_code=response.status_code)