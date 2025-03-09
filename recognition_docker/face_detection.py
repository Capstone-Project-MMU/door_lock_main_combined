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
