from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Message(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Hello from Railway + FastAPI"}

@app.post("/echo")
def echo_message(msg: Message):
    return {"you_sent": msg.text}
