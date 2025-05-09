from fastapi import FastAPI, UploadFile, File, HTTPException
from motion_algorithm import *
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import io

app = FastAPI()

def read_image(file: UploadFile) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("L")
        return np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

@app.post("/compare")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    image1 = read_image(file1)
    image2 = read_image(file2)
    is_different = calculate_difference(image1, image2)
    return JSONResponse(content={"is_different": is_different})