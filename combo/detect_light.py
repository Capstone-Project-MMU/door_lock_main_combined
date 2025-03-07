import cv2
import mediapipe as mp
import face_recognition

# Load reference image (Moh)
reference_image = face_recognition.load_image_file("moh.png")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

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

            # Convert face to RGB
            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # Detect face locations for encoding
            face_locations = face_recognition.face_locations(rgb_face)
            face_encodings = face_recognition.face_encodings(rgb_face, face_locations)

            # If a face was found, compare with Moh
            if face_encodings:
                match = face_recognition.compare_faces([reference_encoding], face_encodings[0])

                if match[0]:
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

    # Show frame
    cv2.imshow("Face Detection & Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
