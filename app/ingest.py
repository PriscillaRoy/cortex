"""Ingest pipeline: load notes -> chunk -> embed -> store in Milvus.

Run with: python -m app.ingest
"""

from app.chunking import load_and_chunk_notes
from app.config import NOTES_DIR
from app.embeddings import embed_texts
from app.vector_store import get_client, insert_chunks, reset_collection


def main() -> None:
    print(f"Loading and chunking notes from {NOTES_DIR}...")
    chunks = load_and_chunk_notes(NOTES_DIR)
    print(f"  -> {len(chunks)} chunks from {len({c.source for c in chunks})} files")

    print("Embedding chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    print(f"  -> {len(embeddings)} embeddings of dim {len(embeddings[0])}")

    print("Storing in Milvus...")
    client = get_client()
    reset_collection(client)
    insert_chunks(client, chunks, embeddings)
    print(f"  -> stored {len(chunks)} chunks in collection")

    print("Done.")


if __name__ == "__main__":
    main()
