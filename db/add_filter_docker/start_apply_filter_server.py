import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.post("/shutdown")
async def shutdown():
    """Shutdown the server."""
    shutdown_event = app.router.shutdown_event
    if shutdown_event:
        await shutdown_event()

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8083, reload=True)
