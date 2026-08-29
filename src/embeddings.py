# ==============================================================================
# embeddings.py — HuggingFace Embedding Engine
# ==============================================================================
# Features:
#   - GPU auto-detection with CPU fallback
#   - Batch embedding computation
#   - Disk-based caching of computed embeddings
#   - Fallback model if primary model fails to load
#   - L2 normalization for cosine similarity via FAISS inner product
# ==============================================================================

import os
import hashlib
import pickle
import numpy as np
from typing import List, Optional

from src.config import (
    EMBEDDING_MODEL_NAME, EMBEDDING_FALLBACK_MODEL,
    EMBEDDING_BATCH_SIZE, EMBEDDINGS_CACHE_DIR,
)
from src.logger import get_logger

logger = get_logger("Embeddings")

# Module-level singleton to avoid reloading the model on every call
_model_instance = None
_model_name_loaded = None


def _get_device() -> str:
    """Auto-detect GPU; fall back to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
            return "cuda"
    except ImportError:
        pass
    logger.info("Using CPU for embeddings.")
    return "cpu"


def _load_model(model_name: str):
    """Load a SentenceTransformer model with error handling."""
    global _model_instance, _model_name_loaded
    
    if _model_instance is not None and _model_name_loaded == model_name:
        return _model_instance
    
    try:
        from sentence_transformers import SentenceTransformer
        device = _get_device()
        logger.info("Loading embedding model: %s on %s", model_name, device)
        _model_instance = SentenceTransformer(model_name, device=device)
        _model_name_loaded = model_name
        logger.info("Embedding model loaded successfully.")
        return _model_instance
    except Exception as e:
        logger.error("Failed to load model '%s': %s", model_name, e)
        return None


def get_embedding_model():
    """
    Load the primary embedding model; if it fails, try the fallback.
    
    Returns:
        SentenceTransformer model instance.
    
    Raises:
        RuntimeError: If both primary and fallback models fail to load.
    """
    model = _load_model(EMBEDDING_MODEL_NAME)
    if model is not None:
        return model

    logger.warning("Primary model failed. Trying fallback: %s",
                    EMBEDDING_FALLBACK_MODEL)
    model = _load_model(EMBEDDING_FALLBACK_MODEL)
    if model is not None:
        return model

    raise RuntimeError(
        f"Both embedding models failed to load: "
        f"primary='{EMBEDDING_MODEL_NAME}', fallback='{EMBEDDING_FALLBACK_MODEL}'. "
        f"Please check your internet connection or install the models manually."
    )


def _cache_key(texts: List[str], model_name: str) -> str:
    """Generate a deterministic cache key from texts + model name."""
    content = model_name + "||" + "||".join(texts)
    return hashlib.sha256(content.encode()).hexdigest()


def _load_from_cache(key: str) -> Optional[np.ndarray]:
    """Load cached embeddings from disk if they exist."""
    cache_file = EMBEDDINGS_CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
            logger.info("Loaded embeddings from cache: %s", cache_file.name)
            return embeddings
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
    return None


def _save_to_cache(key: str, embeddings: np.ndarray) -> None:
    """Save computed embeddings to disk cache."""
    cache_file = EMBEDDINGS_CACHE_DIR / f"{key}.pkl"
    try:
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info("Embeddings cached: %s", cache_file.name)
    except Exception as e:
        logger.warning("Cache save failed: %s", e)


def embed_texts(texts: List[str], use_cache: bool = True) -> np.ndarray:
    """
    Compute normalized embeddings for a list of texts.

    Args:
        texts: List of text strings to embed.
        use_cache: Whether to check/save disk cache.

    Returns:
        np.ndarray of shape (len(texts), embedding_dim), L2-normalized.
        Normalization allows FAISS IndexFlatIP to compute cosine similarity
        directly via inner product.

    Raises:
        RuntimeError: If embedding model cannot be loaded.
        ValueError: If texts list is empty.
    """
    if not texts:
        raise ValueError("Cannot embed an empty list of texts.")

    model = get_embedding_model()
    model_name = _model_name_loaded or EMBEDDING_MODEL_NAME

    # ── Check cache ──────────────────────────────────────────────────────────
    if use_cache:
        key = _cache_key(texts, model_name)
        cached = _load_from_cache(key)
        if cached is not None:
            return cached

    # ── Batch embedding ──────────────────────────────────────────────────────
    logger.info("Computing embeddings for %d texts (batch_size=%d)...",
                len(texts), EMBEDDING_BATCH_SIZE)
    try:
        embeddings = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2 normalize for cosine sim
            convert_to_numpy=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        logger.info("Embeddings computed: shape=%s", embeddings.shape)

        # ── Save to cache ────────────────────────────────────────────────────
        if use_cache:
            _save_to_cache(key, embeddings)

        return embeddings

    except Exception as e:
        logger.error("Embedding computation failed: %s", e, exc_info=True)
        raise RuntimeError(f"Embedding computation failed: {e}") from e


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Args:
        query: Query text.

    Returns:
        np.ndarray of shape (1, embedding_dim), L2-normalized.
    """
    return embed_texts([query], use_cache=False)
