# ==============================================================================
# retriever.py — RAG Retrieval & Classification
# ==============================================================================
# Orchestrates the retrieval pipeline:
#   1. Embed query
#   2. Search FAISS index
#   3. Normalize scores
#   4. Filter by relevance threshold
#   5. Classify chunks into SUPPLY vs DEMAND signals based on keywords
# ==============================================================================

import numpy as np
from typing import List, Dict, Any, Tuple

from src.config import RETRIEVAL_TOP_K, RELEVANCE_THRESHOLD
from src.embeddings import embed_query
from src.faiss_manager import FAISSManager
from src.logger import get_logger

logger = get_logger("Retriever")

# ── Signal Keywords ──────────────────────────────────────────────────────────
SUPPLY_KEYWORDS = {
    "production", "output", "inventory", "stockpile", "mine", "mining",
    "closure", "shutdown", "capacity", "shortage", "surplus", "tariff",
    "export", "import", "logistics", "shipping", "producer", "limit",
    "supply", "refinery", "smelter", "labor", "strike", "disruption"
}

DEMAND_KEYWORDS = {
    "consumption", "buying", "sales", "order", "automotive", "construction",
    "infrastructure", "housing", "manufacturing", "growth", "recession",
    "consumer", "spending", "demand", "purchase", "booking", "sentiment",
    "interest rate", "gdp", "slowdown", "recovery", "ev", "sector"
}


def _classify_chunk(text: str) -> str:
    """
    Classify text as 'SUPPLY', 'DEMAND', or 'GENERAL' based on keyword density.
    """
    text_lower = text.lower()
    supply_score = sum(1 for w in SUPPLY_KEYWORDS if w in text_lower)
    demand_score = sum(1 for w in DEMAND_KEYWORDS if w in text_lower)

    if supply_score > demand_score:
        return "SUPPLY"
    elif demand_score > supply_score:
        return "DEMAND"
    elif supply_score > 0 and demand_score > 0:
        return "BOTH"  # Useful for complex chunks
    else:
        return "GENERAL"


def retrieve_and_classify(
    query: str, faiss_manager: FAISSManager
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve relevant chunks and classify them.

    Args:
        query: Search query (e.g., "Steel market updates in India").
        faiss_manager: Initialized FAISS manager instance.

    Returns:
        Dict with keys "SUPPLY", "DEMAND" containing lists of scored chunks.
    """
    logger.info("Retrieving for query: '%s'", query)

    # 1. Embed query
    try:
        query_vec = embed_query(query)
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        return {"SUPPLY": [], "DEMAND": []}

    # 2. FAISS search
    scores, meta_list = faiss_manager.search(query_vec, k=RETRIEVAL_TOP_K)

    supply_chunks = []
    demand_chunks = []

    if len(scores) == 0:
        logger.warning("No relevant chunks found.")
        return {"SUPPLY": [], "DEMAND": []}

    # 3. Process results
    max_score = np.max(scores) if len(scores) > 0 else 1.0

    for score, meta in zip(scores, meta_list):
        # Normalize score (0-1 approx, though cosine sim is already -1 to 1)
        # Here we just use the raw score since it's cosine sim [0,1] for valid norms
        
        if score < RELEVANCE_THRESHOLD:
            continue

        chunk_text = meta.get("chunk_text", "")  # Note: text is not in meta by default in this design?
        # Wait, in chunker.py we included chunk_text IN separate field, but faiss add_vectors takes meta_list
        # The meta list passed to add_vectors usually includes the text to retrieve it back.
        # Let's verify chunker.py... Ah, chunker returns dict with 'chunk_text' and 'metadata'.
        # We need to flatten this when adding to FAISS.
        # I will handle this flattening in the pipeline setup.
        # Assuming meta here contains 'text' or 'chunk_text'
        
        # If 'chunk_text' is not in meta, we might have lost it.
        # Correction: We will ensure pipeline sends chunk_text INSIDE meta to FAISS.
        
        classification = _classify_chunk(chunk_text)

        result_item = {
            "text": chunk_text,
            "score": float(score),
            "source": meta.get("source"),
            "date": meta.get("published_at"),
            "url": meta.get("url"),
            "classification": classification
        }

        if classification in ["SUPPLY", "BOTH"]:
            supply_chunks.append(result_item)
        if classification in ["DEMAND", "BOTH", "GENERAL"]:
             # General news often reflects macro demand sentiment
            demand_chunks.append(result_item)

    # Sort by score descending
    supply_chunks.sort(key=lambda x: x["score"], reverse=True)
    demand_chunks.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        "Retrieval complete. Supply chunks: %d, Demand chunks: %d",
        len(supply_chunks), len(demand_chunks)
    )

    return {
        "SUPPLY": supply_chunks,
        "DEMAND": demand_chunks
    }
