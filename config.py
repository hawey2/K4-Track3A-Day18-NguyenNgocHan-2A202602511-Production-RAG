"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
# all-MiniLM-L6-v2: 384-dim, đã cache sẵn trên máy (bge-m3 cần tải ~2.3GB).
# EMBEDDING_DIM suy ra tự động từ model ở DenseSearch nếu để None —
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Reranking ---
# ⚠️ FlashRank chỉ có ms-marco-TinyBERT-L-2-v2 (English-only). Test với query
# tiếng Anh trên passage tiếng Việt cho score 0.0000 toàn bộ -> dùng cho
# tiếng Việt là mù hoàn toàn, phá hỏng ranking đúng của BM25/Dense.
# Vì vậy mặc định "auto": thử cross-encoder đa ngôn ngữ trước.
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_BACKEND = "auto"  # "cross-encoder" | "flashrank" | "auto"

# --- LLM ---
LLM_MODEL = "gpt-4o-mini"

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
