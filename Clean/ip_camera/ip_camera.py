import cv2
import picamera2 as pi
import os

cam = pi.Picamera2()
config = cam.create_preview_configuration(main={"format" : "RGB888", "size" : (640, 840)})
cam.configure(config)
cam.start()

frame_index = 0

# Function responsible for creating jpeg image stream and formatting it
def generate_images():
    global frame_index
    while True:
        frame = cam.capture_array()

        # Save latest 3 frames
        filename = f"image_{frame_index % 3}.jpg"
        cv2.imwrite(filename, frame)
        frame_index += 1

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')






