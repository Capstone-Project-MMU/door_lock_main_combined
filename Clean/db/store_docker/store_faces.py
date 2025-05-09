import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Directory for storing face images
IMAGE_DIR = "../images_with_filters"
os.makedirs(IMAGE_DIR, exist_ok=True)

# FAISS index setup
DIMENSION = 512  # Embedding size
INDEX_FILE = "fais_db/filtered_face_index.faiss"
METADATA_FILE = "fais_db/metadata.txt"

# Load existing FAISS index or create a new one
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "r") as f:
        image_names = [line.strip() for line in f.readlines()]
else:
    index = faiss.IndexFlatL2(DIMENSION)
    image_names = []

# Load Face Analysis model
app = FaceAnalysis(name='buffalo_l')  
app.prepare(ctx_id=-1)

def get_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    faces = app.get(img)
    if len(faces) == 0:
        return None, None

    return np.array(faces[0].normed_embedding, dtype=np.float32), img

def store_face(image_path):
    embedding, img = get_embedding(image_path)
    if embedding is None:
        return {"error": "No face detected."}

    filename = os.path.basename(image_path)
    face_path = os.path.join(IMAGE_DIR, filename)

    cv2.imwrite(face_path, img)  # Save image
    index.add(np.expand_dims(embedding, axis=0))  # Add to FAISS
    image_names.append(filename)  # Store metadata

    # Save FAISS index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w") as f:
        f.writelines("\n".join(image_names))

    return {"message": f"Face stored as {filename}"}
if __name__ == "__main__":
    image_path = "moh.png" 
    # Store the face image
    response = store_face(image_path)
    print(response)