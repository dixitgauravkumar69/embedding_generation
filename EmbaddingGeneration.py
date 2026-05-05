from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load model once
model = SentenceTransformer('all-MiniLM-L6-v2')

class TextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Welcome to Embedding API 🚀"}

@app.post("/embedding")
def get_embedding(request: TextRequest):
    embedding = model.encode(request.text)
    
    # ONLY return embedding
    return embedding.tolist()