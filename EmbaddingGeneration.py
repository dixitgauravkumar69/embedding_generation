from fastapi import FastAPI
from pydantic import BaseModel
import hashlib

app = FastAPI()

class TextRequest(BaseModel):
    text: str

def get_embedding(text):
    hash_obj = hashlib.sha256(text.encode())
    return [b / 255.0 for b in hash_obj.digest()]

@app.post("/embedding")
def embedding(req: TextRequest):
    return get_embedding(req.text)

@app.get("/")
def home():
    return {"message": "API is running 🚀"}