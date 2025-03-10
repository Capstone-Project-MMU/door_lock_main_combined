import os
import cv2
import numpy as np
import faiss
from insightface.app import FaceAnalysis

# Directory where face images are stored
IMAGE_DIR = "faces"
INDEX_FILE = "fais_db/face_index.faiss"
DIMENSION = 512  # Embedding size

# Load FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    print("No stored faces found. Run store_faces.py first.")
    exit()

# Load Face Analysis model
app = FaceAnalysis(name='buffalo_l')  # Lightweight model
app.prepare(ctx_id=-1)  # Use CPU

# Function to extract embeddings
def get_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if len(faces) == 0:
        print("No face detected.")
        return None

    face = faces[0]  # Take first detected face
    return np.array(face.normed_embedding, dtype=np.float32)

# Function to search nearest 5 faces
def search_face(query_image_path, k=5):
    query_embedding = get_embedding(query_image_path)
    if query_embedding is None:
        return

    query_embedding = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query_embedding, k)

    print(f"\nTop {k} Matches:")
    for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx == -1:
            continue
        matched_image_path = os.path.join(IMAGE_DIR, f"{idx}.jpg")
        print(f"{rank+1}. {matched_image_path} - Distance: {distance}")
        
        # Display the matched image
        matched_img = cv2.imread(matched_image_path)
        cv2.imshow(f"Match {rank+1}", matched_img)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # Example Usage
image_path = "moh.png"
search_face(image_path)

# if __name__ == "__main__":
#     pass