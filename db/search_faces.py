import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Directory where filtered images are stored
IMAGE_DIR = "images_with_filters"
INDEX_FILE = "fais_db/filtered_face_index.faiss"
METADATA_FILE = "fais_db/metadata.txt"
DIMENSION = 512  

# Load FAISS index and metadata
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        image_names = [line.strip() for line in f.readlines()]
else:
    raise Exception("No stored faces found. Run store_faces.py first.")

# Load Face Analysis model
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=-1)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    faces = app.get(img)
    if len(faces) == 0:
        return None

    return np.array(faces[0].normed_embedding, dtype=np.float32)

def search_face(query_image_path, k=5):
    query_embedding = get_embedding(query_image_path)
    if query_embedding is None:
        return {"error": "No face detected in the query image"}

    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)

    matches = []
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        
        matched_filename = image_names[idx]  # Get file name from metadata
        matched_image_path = os.path.join(IMAGE_DIR, matched_filename)

        matches.append({
            "rank": rank+1,
            "filename": matched_filename,
            "image_path": matched_image_path,
            "distance": float(distance)
        })

    return {"message": "Search completed", "matches": matches}
