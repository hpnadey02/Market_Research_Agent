# ==============================================================================
# data_cleaner.py — News Corpus Cleaning & Normalization
# ==============================================================================
# Responsible for:
#   - HTML tag stripping
#   - Encoding normalization (UTF-8)
#   - Removal of advertisements, navigation text, boilerplate
#   - Deduplication by title similarity
#   - Structured metadata extraction (date, source, country, metal)
#   - Persistence of cleaned corpus to disk
# ==============================================================================

import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from src.config import CORPUS_DIR
from src.logger import get_logger

logger = get_logger("DataCleaner")

# ── Patterns for noise removal ───────────────────────────────────────────────
_AD_PATTERNS = re.compile(
    r"(advertisement|subscribe now|click here|sign up|newsletter|cookie|"
    r"privacy policy|terms of use|all rights reserved|powered by|"
    r"read more\.\.\.|share this article|follow us|download our app)",
    re.IGNORECASE,
)

_NAV_PATTERNS = re.compile(
    r"^(home|about|contact|login|register|menu|search|sitemap|"
    r"previous|next|page \d+|back to top)$",
    re.IGNORECASE,
)

_WHITESPACE = re.compile(r"\s+")
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def _strip_html(text: str) -> str:
    """Remove all HTML tags and decode entities."""
    if not text:
        return ""
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        # Fallback regex-based stripping
        return re.sub(r"<[^>]+>", " ", text)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and encoding."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _is_noise(text: str) -> bool:
    """Check if text is advertisement, navigation, or boilerplate."""
    if len(text) < 20:
        return True
    if _AD_PATTERNS.search(text):
        return True
    if _NAV_PATTERNS.match(text.strip()):
        return True
    return False


def _extract_date(date_str: str) -> str:
    """Try to parse various date formats into ISO format."""
    if not date_str:
        return ""
    # Try ISO format first (from NewsAPI)
    for fmt in [
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%d %b %Y",
        "%B %d, %Y",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
    ]:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            continue
    return date_str.strip()


def _content_hash(text: str) -> str:
    """Generate a hash for deduplication."""
    normalized = text.lower().strip()
    # Remove common stop words and punctuation for fuzzy dedup
    normalized = re.sub(r"[^\w\s]", "", normalized)
    normalized = " ".join(normalized.split()[:15])  # First 15 words
    return hashlib.md5(normalized.encode()).hexdigest()


def clean_and_deduplicate(
    raw_articles: List[Dict[str, Any]],
    metal: str,
    country: str,
) -> List[Dict[str, Any]]:
    """
    Clean, normalize, and deduplicate the raw article corpus.

    Processing pipeline:
        1. Strip HTML from all text fields
        2. Normalize encoding and whitespace
        3. Remove advertisement/navigation noise
        4. Deduplicate by title similarity hash
        5. Attach structured metadata

    Args:
        raw_articles: List of raw article dicts from news_retriever.
        metal: Metal name for metadata tagging.
        country: Country name for metadata tagging.

    Returns:
        List of cleaned, deduplicated article dicts with metadata.
    """
    logger.info("Cleaning %d raw articles...", len(raw_articles))
    seen_hashes = set()
    cleaned = []

    for art in raw_articles:
        try:
            # ── Strip HTML ───────────────────────────────────────────────────
            title = _strip_html(art.get("title", ""))
            description = _strip_html(art.get("description", ""))
            content = _strip_html(art.get("content", ""))

            # ── Normalize ────────────────────────────────────────────────────
            title = _normalize_text(title)
            description = _normalize_text(description)
            content = _normalize_text(content)

            # ── Skip noise ───────────────────────────────────────────────────
            combined_text = f"{title} {description} {content}".strip()
            if _is_noise(combined_text):
                continue

            # ── Deduplicate ──────────────────────────────────────────────────
            h = _content_hash(title if title else combined_text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # ── Build cleaned entry with metadata ────────────────────────────
            cleaned_entry = {
                "title": title,
                "description": description,
                "content": content if content else description,
                "full_text": combined_text,
                "url": art.get("url", ""),
                "published_at": _extract_date(art.get("published_at", "")),
                "source": art.get("source", "Unknown"),
                "metal": metal,
                "country": country,
                "cleaned_at": datetime.utcnow().isoformat(),
            }
            cleaned.append(cleaned_entry)

        except Exception as e:
            logger.warning("Skipping malformed article: %s — Error: %s",
                           art.get("title", "?")[:50], e)
            continue

    logger.info(
        "Cleaning complete: %d → %d articles (removed %d duplicates/noise).",
        len(raw_articles), len(cleaned), len(raw_articles) - len(cleaned),
    )

    # ── Persist cleaned corpus ───────────────────────────────────────────────
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        corpus_file = CORPUS_DIR / f"corpus_{metal}_{country}_{timestamp}.json"
        with open(corpus_file, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2, ensure_ascii=False)
        logger.info("Cleaned corpus saved to %s", corpus_file)
    except Exception as e:
        logger.error("Failed to save corpus: %s", e)

    return cleaned
