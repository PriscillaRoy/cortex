"""Core RAG pipeline: retrieve relevant chunks, build a prompt, generate
an answer.

Phase 0 keeps retrieve() and generate_answer() as plain functions. Phase 1
wraps these with tracing spans + latency histograms without changing
their signatures - that's the point of keeping them separate and small.
"""

from dataclasses import dataclass

from app.config import TOP_K
from app.embeddings import embed_query
from app.llm import generate
from app.vector_store import get_client, search


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    score: float


@dataclass
class AskResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


def retrieve(query: str, top_k: int = TOP_K) -> list[RetrievedChunk]:
    client = get_client()
    query_embedding = embed_query(query)
    hits = search(client, query_embedding, top_k)

    return [
        RetrievedChunk(
            text=hit["entity"]["text"],
            source=hit["entity"]["source"],
            chunk_index=hit["entity"]["chunk_index"],
            score=hit["distance"],
        )
        for hit in hits
    ]


def build_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n---\n\n".join(
        f"[Source: {c.source}, chunk {c.chunk_index}]\n{c.text}" for c in chunks
    )

    return f"""You are a helpful assistant answering questions based on the user's \
personal notes. Use ONLY the context below to answer the question. If the \
context doesn't contain the answer, say so clearly rather than guessing.

Context:
{context}

Question: {query}

Answer:"""


def ask(query: str, top_k: int = TOP_K) -> AskResult:
    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)
    answer = generate(prompt)

    return AskResult(answer=answer, retrieved_chunks=chunks)
