import cv2
import picamera2 as pi

cam = pi.Picamera2()
config = cam.create_preview_configuration(main={"format" : "RGB888", "size" : (640, 840)})
cam.configure(config)
cam.start()

# Function responsible for creating jpeg image stream and formatting it
def generate_images():
    while True: 
        sucess, frame = cam.read()
        if not sucess: 
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')