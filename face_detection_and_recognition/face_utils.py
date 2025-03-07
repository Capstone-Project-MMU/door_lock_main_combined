import cv2
import mediapipe as mp
import face_recognition

def detect_faces(frame):
    """Detect faces in a frame using Mediapipe."""
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    faces = []
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
            
            face = frame[y : y + height, x : x + width]
            faces.append((x, y, width, height, face))
    
    return faces

def recognize_face(face, reference_encoding):
    """Recognize a face by comparing with a reference encoding."""
    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_face)
    face_encodings = face_recognition.face_encodings(rgb_face, face_locations)
    
    if face_encodings:
        match = face_recognition.compare_faces([reference_encoding], face_encodings[0])
        return match[0]
    return False

def main():
    """Main function to test face detection and recognition."""
    reference_image = face_recognition.load_image_file("moh.png")
    reference_encoding = face_recognition.face_encodings(reference_image)[0]
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(frame)
        for x, y, width, height, face in faces:
            is_moh = recognize_face(face, reference_encoding)
            label = "Moh" if is_moh else "False"
            color = (0, 255, 0) if is_moh else (0, 0, 255)
            
            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("Face Detection & Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
