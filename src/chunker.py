# ==============================================================================
# chunker.py — Hybrid Semantic + Recursive Chunking
# ==============================================================================
# STRATEGY RATIONALE:
#
# Financial/commodity news articles are typically short (200-800 words) with
# dense causal reasoning ("prices fell because supply increased due to...").
# A naive fixed-size splitter would break these causal chains mid-sentence.
#
# Our hybrid approach:
#   1. RecursiveCharacterTextSplitter with paragraph → sentence → word
#      hierarchy, preserving natural document structure.
#   2. chunk_size=512 characters — small enough for precise retrieval but
#      large enough to preserve one or two complete causal reasoning chains.
#   3. chunk_overlap=64 characters — ensures sentences at chunk boundaries
#      appear in both adjacent chunks, preventing information loss.
#   4. Per-chunk metadata — each chunk carries its parent article's metadata
#      (source, date, metal, country) so the retriever can filter/boost.
#
# This outperforms sentence-level splitting (too fragmented for reasoning)
# and document-level splitting (too diluted for precise retrieval).
# ==============================================================================

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from src.logger import get_logger

logger = get_logger("Chunker")


def chunk_articles(
    cleaned_articles: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Split cleaned articles into semantically meaningful chunks with metadata.

    Uses RecursiveCharacterTextSplitter which splits on these separators
    in order of priority:
        "\\n\\n" → paragraph break
        "\\n"   → line break
        ". "    → sentence break
        " "     → word break

    This hierarchy ensures we split at the most natural boundary first.

    Args:
        cleaned_articles: List of cleaned article dicts from data_cleaner.

    Returns:
        List of chunk dicts, each containing:
            - chunk_text: The text content of the chunk
            - chunk_id: Unique identifier (article_index:chunk_index)
            - metadata: Inherited from parent article
    """
    logger.info("Chunking %d articles (size=%d, overlap=%d)...",
                len(cleaned_articles), CHUNK_SIZE, CHUNK_OVERLAP)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: List[Dict[str, Any]] = []

    for art_idx, article in enumerate(cleaned_articles):
        text = article.get("full_text", "").strip()
        if not text or len(text) < 30:
            continue

        try:
            splits = splitter.split_text(text)

            for chunk_idx, chunk_text in enumerate(splits):
                chunk_text = chunk_text.strip()
                if len(chunk_text) < 20:
                    continue  # Skip tiny fragments

                all_chunks.append({
                    "chunk_text": chunk_text,
                    "chunk_id": f"{art_idx}:{chunk_idx}",
                    "metadata": {
                        "title": article.get("title", ""),
                        "source": article.get("source", ""),
                        "published_at": article.get("published_at", ""),
                        "url": article.get("url", ""),
                        "metal": article.get("metal", ""),
                        "country": article.get("country", ""),
                        "article_index": art_idx,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(splits),
                    },
                })

        except Exception as e:
            logger.warning("Failed to chunk article %d: %s", art_idx, e)
            continue

    logger.info("Chunking complete: %d articles → %d chunks.",
                len(cleaned_articles), len(all_chunks))
    return all_chunks
