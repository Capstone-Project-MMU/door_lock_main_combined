from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
from ip_camera import generate_images

app = FastAPI()
# uvicorn api:app --host 0.0.0.0 --port 8079

@app.get("/camera")
def video_feed():
    return StreamingResponse(generate_images(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")
