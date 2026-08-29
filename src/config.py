# ==============================================================================
# config.py — Centralized Configuration & Path Management
# ==============================================================================
# All project settings, paths, and environment variable loading are handled here.
# This module auto-creates required directories and validates the environment.
# ==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
load_dotenv(override=True)

# ── Project Root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"F:\data\GEN AI PROJECTS\market research insight")

# ── Directory Paths ──────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
CORPUS_DIR = DATA_DIR / "corpus"
EMBEDDINGS_CACHE_DIR = DATA_DIR / "embeddings_cache"
INSIGHTS_DIR = DATA_DIR / "insights"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODEL_DIR = Path(os.getenv("LOCAL_MODEL_PATH", str(PROJECT_ROOT / "model")))

# Auto-create all required directories
for _dir in [DATA_DIR, FAISS_INDEX_DIR, CORPUS_DIR, EMBEDDINGS_CACHE_DIR,
             INSIGHTS_DIR, LOGS_DIR, CONFIGS_DIR, MODEL_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── LLM Provider Settings ───────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash")

# GROQ
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# HuggingFace API
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL_NAME = os.getenv("HUGGINGFACE_MODEL_NAME",
                                    "mistralai/Mistral-7B-Instruct-v0.3")

# ── Embedding Model Settings ────────────────────────────────────────────────
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME",
                                  "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_FALLBACK_MODEL = os.getenv("EMBEDDING_FALLBACK_MODEL",
                                      "sentence-transformers/paraphrase-MiniLM-L3-v2")

# ── News API ─────────────────────────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# ── RAG Pipeline Hyperparameters ─────────────────────────────────────────────
# Chunk settings — optimized for short financial/news paragraphs.
# 512 tokens preserves causal reasoning chains while staying small enough for
# high-precision similarity search.  64-token overlap prevents sentence
# splitting at chunk boundaries.
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval settings
RETRIEVAL_TOP_K = 15            # Number of candidate chunks to retrieve
RELEVANCE_THRESHOLD = 0.25      # Minimum cosine-similarity for inclusion
EMBEDDING_BATCH_SIZE = 64       # Batch size for embedding computation

# ── News Date Window ─────────────────────────────────────────────────────────
NEWS_LOOKBACK_DAYS = 10

# ── SteelOrbis Config ────────────────────────────────────────────────────────
STEELORBIS_URL = "https://www.steelorbis.com/steel-news/latest-news/"
REQUEST_TIMEOUT = 30            # seconds
MAX_RETRIES = 3

# ── LLM Generation Settings ─────────────────────────────────────────────────
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.4           # Slightly creative but factual
