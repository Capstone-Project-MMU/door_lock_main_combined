import cv2
import numpy as np
import mediapipe as mp

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

def detect_faces_movement(frame, face_detection):
    """Detect faces in a frame using Mediapipe."""
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
            x, y, width, height = max(0, x), max(0, y), max(1, width), max(1, height)
            face = frame[y : y + height, x : x + width]
            faces.append((x, y, width, height, face))
    return faces

# to overlay the bounding box on the frame
def draw_faces(frame, faces):
    """Draw bounding boxes around detected faces."""
    for (x, y, width, height, _) in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return frame

# get two frames then detect face in each and draw the bounding box, then see how much did the face move
def calculate_movement(face1, face2):
    """Calculate the movement of the face between two frames."""
    if not face1 or not face2:
        return 0

    x1, y1, w1, h1, _ = face1[0]
    x2, y2, w2, h2, _ = face2[0]

    # Calculate the center of the bounding boxes
    center1 = (x1 + w1 // 2, y1 + h1 // 2)
    center2 = (x2 + w2 // 2, y2 + h2 // 2)

    # Calculate Euclidean distance between centers
    distance = np.linalg.norm(np.array(center1) - np.array(center2))
    return distance

def normalize_horizontal_position(x, width, frame_width):
    """
    Normalize the horizontal position of a face so that:
    - center of the frame is 0
    - right edge is +100
    - left edge is -100
    """
    face_center_x = x + width // 2
    offset_from_center = face_center_x - frame_width // 2
    normalized = (offset_from_center / (frame_width / 2)) * 100
    return np.clip(normalized, -100, 100)


def test():
    cap = cv2.VideoCapture(0)
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.4) as face_detection:
        # Warm-up
        for _ in range(20):
            cap.read()
        ret, frame1 = cap.read()
        if not ret:
            print("Failed to capture first frame")
            cap.release()
            exit()

        faces1 = detect_faces_movement(frame1, face_detection)
        frame1 = draw_faces(frame1, faces1)
        cv2.imshow("Frame 1", frame1)
        cv2.waitKey(5000)

        cv2.waitKey(1000)
        ret, frame2 = cap.read()
        if not ret:
            print("Failed to capture second frame")
            cap.release()
            exit()

        faces2 = detect_faces_movement(frame2, face_detection)
        frame2 = draw_faces(frame2, faces2)
        cv2.imshow("Frame 2", frame2)
        cv2.waitKey(5000)

        movement = calculate_movement(faces1, faces2)
        print(f"Face movement distance: {movement:.2f}")

        if faces2:
            x, y, w, h, _ = faces2[0]
            normalized_position = normalize_horizontal_position(x, w, frame2.shape[1])
            print(f"Normalized horizontal position (center=0): {normalized_position:.2f}")

    cap.release()
    cv2.destroyAllWindows()


# Example usage:
if __name__ == "__main__":
    test()