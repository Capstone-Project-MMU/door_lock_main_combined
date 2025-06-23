from fastapi import FastAPI
from pydantic import BaseModel
import paho.mqtt.publish as publish

#uvicorn api:app --host 0.0.0.0 --port 8000

app = FastAPI() 


MQTT_BROKER = "192.168.10.111"
MQTT_TOPIC = "home/text_message"
MQTT_USER = "myuser"    
MQTT_PASS = "mypassword"

class Message(BaseModel):
    word: str

@app.post("/send-word")
def send_word(msg: Message):
    publish.single(
        topic=MQTT_TOPIC,
        payload=msg.word,
        hostname=MQTT_BROKER,
        auth={'username': MQTT_USER, 'password': MQTT_PASS}
    )
    return {"status": "sent", "word": msg.word}
