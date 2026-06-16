"""Tool: search_notes

Semantic search over note chunks using the existing RAG retrieval
pipeline. Returns top-k chunks most relevant to the query.

This is the same retrieval that /ask uses — exposed here as a named
tool so the agent can call it explicitly as one step in a multi-step
reasoning chain.
"""

from app.config import TOP_K
from app.rag import RetrievedChunk, retrieve


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": (
            "Semantically search your personal notes for content relevant "
            "to a query. Returns the most relevant text chunks. Use this "
            "when you need to find specific facts, explanations, or details "
            "from the notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query — what you're looking for in the notes.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return (default 4, max 8).",
                },
            },
            "required": ["query"],
        },
    },
}


def run(query: str, top_k: int = TOP_K) -> dict:
    """Execute the search and return a structured result dict."""
    try:
        top_k = int(top_k)  # LLM sometimes sends "6" instead of 6
    except (TypeError, ValueError):
        top_k = TOP_K
    top_k = min(top_k, 8)
    chunks, fallback = retrieve(query, top_k=top_k)

    return {
        "query": query,
        "count": len(chunks),
        "fallback": fallback,
        "chunks": [
            {
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
            }
            for c in chunks
        ],
    }


def format_for_llm(result: dict) -> str:
    """Format the tool result as text for the LLM's next context window."""
    if not result["chunks"]:
        return f"No results found for query: {result['query']!r}"

    lines = [f"Search results for {result['query']!r} ({result['count']} chunks):"]
    if result.get("fallback"):
        lines.append("[Note: vector search timed out, using keyword fallback]")

    for i, chunk in enumerate(result["chunks"], 1):
        lines.append(f"\n[{i}] Source: {chunk['source']} (score: {chunk['score']})")
        lines.append(chunk["text"][:400] + ("..." if len(chunk["text"]) > 400 else ""))

    return "\n".join(lines)
