import cv2
import numpy as np
from deepface import DeepFace

# Load reference image (Moh)
reference_image = "moh.png"

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]  # Crop face region

        # Save the detected face as temp image
        temp_image_path = "temp_face.jpg"
        cv2.imwrite(temp_image_path, face)

        try:
            # Compare detected face with Moh's image
            result = DeepFace.verify(temp_image_path, reference_image, model_name="Facenet", enforce_detection=False)
            
            # Print result based on verification
            if result["verified"]:
                print("Moh")
                label = "Moh"
                color = (0, 255, 0)  # Green for match
            else:
                print("False")
                label = "False"
                color = (0, 0, 255)  # Red for mismatch
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        except Exception as e:
            print("Face verification failed:", e)

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()