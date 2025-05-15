import cv2

camera = cv2.VideoCapture(0)

# Function responsible for creating jpeg image stream and formatting it
def generate_images():
    while True: 
        sucess, frame = camera.read()
        if not sucess: 
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')