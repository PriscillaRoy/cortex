"""Cortex API.

Run with: uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask

app = FastAPI(title="Cortex", description="Ask questions over your notes.")


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None


class RetrievedChunkResponse(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class AskResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunkResponse]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest) -> AskResponse:
    kwargs = {}
    if request.top_k is not None:
        kwargs["top_k"] = request.top_k

    result = ask(request.query, **kwargs)

    return AskResponse(
        answer=result.answer,
        retrieved_chunks=[
            RetrievedChunkResponse(
                text=c.text, source=c.source, chunk_index=c.chunk_index, score=c.score
            )
            for c in result.retrieved_chunks
        ],
    )
