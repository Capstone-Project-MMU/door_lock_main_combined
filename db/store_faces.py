import os
import cv2
import numpy as np
import onnxruntime as ort
import faiss
from insightface.app import FaceAnalysis

# Directory for storing face images
IMAGE_DIR = "faces"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Load Face Analysis model
app = FaceAnalysis(name='buffalo_l')  # Lightweight face model
app.prepare(ctx_id=-1)  # Use CPU

# FAISS Index
DIMENSION = 512  # Embedding size
INDEX_FILE = "fais_db/face_index.faiss"

# Load existing FAISS index or create a new one
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatL2(DIMENSION)

# Function to extract embeddings
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if len(faces) == 0:
        print("No face detected.")
        return None, None

    face = faces[0]  # Take the first detected face
    embedding = np.array(face.normed_embedding, dtype=np.float32)
    return embedding, img

# Function to store an image and embedding
def store_face(image_path):
    embedding, img = get_embedding(image_path)
    if embedding is None:
        return

    # Save the image with a unique name
    face_id = len(os.listdir(IMAGE_DIR))
    new_filename = f"{face_id}.jpg"
    face_path = os.path.join(IMAGE_DIR, new_filename)
    cv2.imwrite(face_path, img)

    # Add embedding to FAISS index
    index.add(np.expand_dims(embedding, axis=0))

    # Save FAISS index
    faiss.write_index(index, INDEX_FILE)
    print(f"Stored: {new_filename}")

# Example Usage
# image_path = "moh.png"
# store_face(image_path)

if __name__ == "__main__":
    pass
