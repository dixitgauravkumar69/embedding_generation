from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

# Load AI Embedding Model
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

class TextRequest(BaseModel):
    text: str


@app.post("/embedding")
def embedding(req: TextRequest):

    # Clean text
    text = req.text.lower().strip()

    # Generate semantic embedding
    embedding_vector = model.encode(text).tolist()

    return {
        "embedding": embedding_vector
    }


@app.get("/")
def home():
    return {
        "message": "AI Embedding API Running 🚀"
    }
