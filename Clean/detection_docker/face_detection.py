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

# get one frame only and calculate how far it is from the center of the frame
def get_face_position(frame, face):
    """
    Get the horizontal position of a face in a frame.
    Returns a value between -100 (left edge) and +100 (right edge).
    """
    x, y, width, height, _ = face
    frame_height, frame_width, _ = frame.shape

    # Normalize the horizontal position
    normalized_x = normalize_horizontal_position(x, width, frame_width)
    
    return normalized_x
# Normalize the horizontal position of a face
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

# to test the function
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    for _ in range(20):
            cap.read()
    ret, frame = cap.read()
    
    # Detect faces in the image
    faces = detect_faces(frame)
    
    for face in faces:
        position = get_face_position(frame, face)
        print(f"Face position: {position}")
        
        # Draw rectangle around the face
        x, y, width, height, _ = face
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # to normalize the position
        normalized_position = normalize_horizontal_position(x, width, frame.shape[1])
        cv2.putText(frame, f"Position: {normalized_position:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # wait a bit to see the result
        cv2.waitKey(1000)
        #save the face with the position
        face_image = face[4]
        cv2.imwrite(f"face_position_{normalized_position:.2f}.jpg", face_image)
    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


