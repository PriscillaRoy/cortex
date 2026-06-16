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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:latest")
NOTES_DIR = os.environ.get("NOTES_DIR", "data/notes")

# Fixed seed for app/baseline.py - makes save/check runs deterministic
# (same prompt -> same output), so "did the answer change" reflects
# changes we made (prompt/model/chunks), not LLM sampling randomness.
# Normal /ask and /ask/compare calls do NOT use a seed - those should
# behave like a real LLM (naturally varied), only baseline comparisons
# need determinism.
BASELINE_SEED = 42

# Per-stage latency budgets (seconds).
# Based on real p50/p95 measurements from app/latency_report.py:
#   embed_query  p95 ~900ms (warm), but cold-start ~8-30s (not budgeted
#                here — warmup at startup handles that)
#   vector_search p95 ~120ms
#   generate      p95 ~3.7s (v1 prompt), ~2.4s (v2 prompt)
#
# Budgets are generous (2-3x p95) to avoid false timeouts under normal
# load, while still bounding the worst-case hang.
TIMEOUT_EMBED_S   = 2.0   # embedding a single query string
TIMEOUT_RETRIEVE_S = 3.0  # embed + vector search combined
TIMEOUT_GENERATE_S = 12.0 # LLM generation (generous — local Ollama varies)
TIMEOUT_TOTAL_S   = 15.0  # end-to-end request ceiling
