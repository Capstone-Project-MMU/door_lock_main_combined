import cv2
import mediapipe as mp
from deepface import DeepFace

# Load reference image (Moh)
reference_image = "moh.png"

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (Mediapipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    # If faces are detected
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, width, height = (
                int(bboxC.xmin * w),
                int(bboxC.ymin * h),
                int(bboxC.width * w),
                int(bboxC.height * h),
            )

            # Ensure valid bounding box
            x, y, width, height = max(0, x), max(0, y), max(1, width), max(1, height)

            # Crop the detected face
            face = frame[y : y + height, x : x + width]

            # Save the detected face as "human.png"
            human_image_path = "human.png"
            cv2.imwrite(human_image_path, face)

            try:
                # Compare detected face with Moh's image
                result = DeepFace.verify(
                    human_image_path, reference_image, model_name="Facenet", enforce_detection=False
                )

                # Determine if the face matches Moh
                if result["verified"]:
                    print("Moh")
                    label = "Moh"
                    color = (0, 255, 0)  # Green for match
                else:
                    print("False")
                    label = "False"
                    color = (0, 0, 255)  # Red for mismatch

                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            except Exception as e:
                print("Face verification failed:", e)

    # Show frame
    cv2.imshow("Face Detection & Verification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
