import os
import cv2
import numpy as np
import onnxruntime as ort
import faiss
from insightface.app import FaceAnalysis
from PIL import Image, ImageEnhance
from store_faces import store_face
from search_faces import search_face
from add_filters import apply_filters
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/add-filters")
def add_filters(image_path: str, output_dir: str):
    return apply_filters("moh.png", "images_with_filters")

@app.post("/store")
def store_Face(image_path: str):
    return store_face("moh.png")

@app.post("/search")
def search_Face(query_image_path: str, k: int):
    return search_face("moh.png", 5)



