"""Original FastAPI endpoints plus the Assignment 2 CNN classifier."""
from fastapi import FastAPI
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
from app.cnn_router import router as cnn_router

app = FastAPI(title="GenAI Assignments 1-2")

# Original sample corpus, preserved from the GitHub repository.
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)
    return {"word": request.word, "embedding": embedding.tolist()}


app.include_router(cnn_router)
