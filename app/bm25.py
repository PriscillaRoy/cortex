"""BM25 keyword search over note chunks — retrieval fallback.

Used when vector search times out. Pure Python, no extra dependencies.
Loaded lazily at first use, kept in memory after that.

BM25 (Best Match 25) ranks documents by term frequency, adjusted for
document length. For short focused notes it performs reasonably well on
exact/near-exact keyword queries — not as good as semantic search for
conceptual questions, but far better than returning an error.

The key property we need: it works without Milvus. If vector search
times out (Milvus overloaded, index corruption, etc.), this path
doesn't touch the vector store at all — it reads raw text from the
chunks we already have in memory.
"""

import math
import re
from collections import Counter

from app.rag import RetrievedChunk


class BM25Index:
    """In-memory BM25 index over a flat list of text chunks.

    Parameters follow the standard BM25+ defaults:
      k1=1.5  — term frequency saturation (higher = more weight to TF)
      b=0.75  — document length normalization (1.0 = full, 0 = none)
    """

    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks  # list of {"text", "source", "chunk_index"}
        self.k1 = k1
        self.b = b

        # Tokenize all chunks
        self.tokenized = [self._tokenize(c["text"]) for c in chunks]

        # Document frequency (how many chunks contain each term)
        self.df: dict[str, int] = Counter()
        for tokens in self.tokenized:
            for term in set(tokens):
                self.df[term] += 1

        self.n_docs = len(chunks)
        self.avgdl = (
            sum(len(t) for t in self.tokenized) / self.n_docs
            if self.n_docs > 0
            else 1.0
        )

    def _tokenize(self, text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, drop short tokens."""
        return [
            t for t in re.split(r"[^a-z0-9]+", text.lower())
            if len(t) > 1
        ]

    def search(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        """Return top-k chunks ranked by BM25 score."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        scores = []
        for i, tokens in enumerate(self.tokenized):
            tf_map = Counter(tokens)
            dl = len(tokens)
            score = 0.0

            for term in query_terms:
                if term not in tf_map:
                    continue

                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                df = self.df.get(term, 0)
                idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

                # TF with length normalization
                tf = tf_map[term]
                tf_norm = (
                    tf * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
                )

                score += idf * tf_norm

            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        top = scores[:top_k]

        return [
            RetrievedChunk(
                text=self.chunks[i]["text"],
                source=self.chunks[i]["source"],
                chunk_index=self.chunks[i]["chunk_index"],
                score=round(score, 4),
            )
            for i, score in top
            if score > 0  # don't return zero-score chunks
        ]


# Module-level singleton — built once, reused.
_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    """Return the singleton BM25 index, building it if needed.

    Reads all chunks currently stored in Milvus. This means the BM25
    index always reflects the same data as the vector store — no separate
    sync needed.
    """
    global _index
    if _index is not None:
        return _index

    from app.vector_store import get_client

    client = get_client()
    # Query all chunks (no filter, high limit)
    results = client.query(
        collection_name="notes",
        filter="chunk_index >= 0",
        output_fields=["text", "source", "chunk_index"],
        limit=10_000,
    )

    chunks = [
        {
            "text": r["text"],
            "source": r["source"],
            "chunk_index": r["chunk_index"],
        }
        for r in results
    ]

    _index = BM25Index(chunks)
    return _index


def bm25_search(query: str, top_k: int = 4) -> list[RetrievedChunk]:
    """Convenience wrapper used by the fallback path in app/rag.py."""
    return get_bm25_index().search(query, top_k=top_k)
