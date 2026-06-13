"""Central configuration for Cortex. Keep all tunables here so later
phases (latency budgets, eval thresholds) have one place to look."""

import os

# --- Paths ---
NOTES_DIR = os.environ.get("CORTEX_NOTES_DIR", "data/notes")
MILVUS_DB_PATH = os.environ.get("CORTEX_MILVUS_PATH", "data/cortex.db")

# --- Embedding model ---
# all-MiniLM-L6-v2: 384-dim, ~80MB, fast on CPU. Good default for a
# memory-constrained deployment story later.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Chunking ---
CHUNK_SIZE_CHARS = 800
CHUNK_OVERLAP_CHARS = 100

# --- Retrieval ---
TOP_K = 4

# --- Milvus collection ---
COLLECTION_NAME = "cortex_chunks"

# --- LLM (Ollama) ---
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")
