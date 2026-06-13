"""Thin wrapper around Milvus Lite for storing and querying chunk
embeddings.

Milvus Lite runs embedded (single file on disk, no separate server) but
uses the same client API as full Milvus. The `connect_to_db` / collection
setup here is the part that changes if migrating to a real Milvus
deployment (Docker/Zilliz Cloud) later - everything else (insert, search)
stays the same.
"""

from pymilvus import MilvusClient, DataType

from app.chunking import Chunk
from app.config import COLLECTION_NAME, EMBEDDING_DIM, MILVUS_DB_PATH


def get_client() -> MilvusClient:
    return MilvusClient(MILVUS_DB_PATH)


def ensure_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=4000)
    schema.add_field("source", DataType.VARCHAR, max_length=256)
    schema.add_field("chunk_index", DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )


def reset_collection(client: MilvusClient) -> None:
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    ensure_collection(client)


def insert_chunks(client: MilvusClient, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    rows = [
        {
            "vector": emb,
            "text": chunk.text,
            "source": chunk.source,
            "chunk_index": chunk.chunk_index,
        }
        for chunk, emb in zip(chunks, embeddings)
    ]
    client.insert(collection_name=COLLECTION_NAME, data=rows)


def search(client: MilvusClient, query_embedding: list[float], top_k: int) -> list[dict]:
    client.load_collection(COLLECTION_NAME)

    results = client.search(
        collection_name=COLLECTION_NAME,
        data=[query_embedding],
        limit=top_k,
        output_fields=["text", "source", "chunk_index"],
    )
    # results is a list (one per query vector) of lists of hits
    return results[0]
