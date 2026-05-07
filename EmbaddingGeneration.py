from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cpu'
)

class TextRequest(BaseModel):
    text: str

@app.post("/embedding")
def embedding(req: TextRequest):

    text = req.text.lower().strip()

    embedding_vector = model.encode(text).tolist()

    return {
        "embedding": embedding_vector
    }

@app.get("/")
def home():
    return {
        "message": "Embedding API Running 🚀"
    }
