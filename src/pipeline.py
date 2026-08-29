# ==============================================================================
# pipeline.py — End-to-End RAG Pipeline Orchestrator
# ==============================================================================
# Coordinates the entire flow:
#   1. Validate Input
#   2. Retrieve News (API + Scraper)
#   3. Clean & Deduplicate
#   4. Chunk & Embed
#   5. Index to FAISS
#   6. Retrieve & Classify
#   7. Generate Insights (Supply, Demand, Summary)
#
# Returns a comprehensive result dictionary for the UI.
# ==============================================================================

import time
from typing import Dict, Any

from src.validators import validate_inputs
from src.news_retriever import retrieve_news
from src.data_cleaner import clean_and_deduplicate
from src.chunker import chunk_articles
from src.embeddings import embed_texts
from src.faiss_manager import FAISSManager
from src.retriever import retrieve_and_classify
from src.llm_engine import generate_insight
from src.prompts import (
    SUPPLY_PROMPT_TEMPLATE, 
    DEMAND_PROMPT_TEMPLATE, 
    SUMMARY_PROMPT_TEMPLATE
)
from src.logger import get_logger

logger = get_logger("Pipeline")


def run_pipeline(
    metal: str, country: str, price: str, weekly_change: str, progress_callback=None
) -> Dict[str, Any]:
    """
    Execute the full market research pipeline.

    Args:
        metal: Metal name.
        country: Country name.
        price: Current price.
        weekly_change: Weekly percentage change.
        progress_callback: Optional function to update UI progress (0-100).

    Returns:
        Dict containing:
            - status: "success" or "error"
            - error_message: str (if error)
            - supply_insight: str
            - demand_insight: str
            - summary: str
            - retrieval_results: Dict (for UI display)
            - stats: Dict (article counts, etc.)
    """
    start_time = time.time()
    result = {"status": "success", "error_message": ""}

    def update_progress(msg: str, value: int):
        logger.info(f"Stage: {msg}")
        if progress_callback:
            progress_callback(msg, value)

    try:
        # ── 1. Validation ───────────────────────────────────────────────────
        update_progress("Validating inputs...", 10)
        is_valid, data, val_err = validate_inputs(metal, country, price, weekly_change)
        if not is_valid:
            return {"status": "error", "error_message": f"Validation Failed: {val_err}"}

        # ── 2. Retrieval ────────────────────────────────────────────────────
        update_progress("Retrieving news (NewsAPI + SteelOrbis)...", 20)
        raw_articles = retrieve_news(data["metal"], data["country"])
        
        if not raw_articles:
            # We can still proceed if no news found, but insights will be weak.
            # However, for a robust app, we might want to warn.
            logger.warning("No articles found!")
            result["warning"] = "No recent news found. Insights may be generic."

        # ── 3. Cleaning ─────────────────────────────────────────────────────
        update_progress("Cleaning & deduplicating corpus...", 35)
        cleaned = clean_and_deduplicate(raw_articles, data["metal"], data["country"])
        
        # ── 4. Chunking ─────────────────────────────────────────────────────
        update_progress("Chunking content...", 45)
        chunks = chunk_articles(cleaned)
        
        if not chunks:
            logger.warning("No chunks available after processing.")
            if not result.get("warning"):
                result["warning"] = "Content processing yielded no usable text."

        # ── 5. Embedding & Indexing ─────────────────────────────────────────
        update_progress("Generating embeddings & building index...", 60)
        
        faiss_mgr = FAISSManager()
        faiss_mgr.create_index()  # Start fresh for each query to prioritize relevance
        
        if chunks:
            texts = [c["chunk_text"] for c in chunks]
            embeddings = embed_texts(texts, use_cache=True)
            
            # Add metadata (include text for retrieval display)
            meta_list = []
            for c in chunks:
                m = c["metadata"].copy()
                m["chunk_text"] = c["chunk_text"]
                meta_list.append(m)
                
            faiss_mgr.add_vectors(embeddings, meta_list)
        
        # ── 6. Retrieval & Classification ───────────────────────────────────
        update_progress("Retrieving & classifying signals...", 75)
        query = f"{data['metal']} market trends supply demand {data['country']}"
        retrieved = retrieve_and_classify(query, faiss_mgr)
        
        result["retrieval_results"] = retrieved
        
        # Prepare context strings
        supply_ctx = "\n\n".join(
            [f"- {item['text']} ({item['date']})" for item in retrieved["SUPPLY"][:5]]
        )
        demand_ctx = "\n\n".join(
            [f"- {item['text']} ({item['date']})" for item in retrieved["DEMAND"][:5]]
        )
        
        # Fallback if contexts are empty
        if not supply_ctx: supply_ctx = "No specific supply-side news found."
        if not demand_ctx: demand_ctx = "No specific demand-side news found."

        # ── 7. Insight Generation ───────────────────────────────────────────
        update_progress("Generating analyst insights (Supply)...", 85)
        
        # Supply Insight
        supply_prompt = SUPPLY_PROMPT_TEMPLATE.format(
            metal=data["metal"],
            country=data["country"],
            price=data["price"],
            change=data["weekly_change"],
            context=supply_ctx
        )
        supply_insight = generate_insight(supply_prompt)
        
        update_progress("Generating analyst insights (Demand)...", 90)
        # Demand Insight
        demand_prompt = DEMAND_PROMPT_TEMPLATE.format(
            metal=data["metal"],
            country=data["country"],
            price=data["price"],
            change=data["weekly_change"],
            context=demand_ctx
        )
        demand_insight = generate_insight(demand_prompt)
        
        update_progress("Generating executive summary...", 95)
        # Summary
        summary_prompt = SUMMARY_PROMPT_TEMPLATE.format(
            metal=data["metal"],
            country=data["country"],
            price=data["price"],
            change=data["weekly_change"],
            supply_insight=supply_insight,
            demand_insight=demand_insight
        )
        summary = generate_insight(summary_prompt)

        # ── Finalize ────────────────────────────────────────────────────────
        result["supply_insight"] = supply_insight
        result["demand_insight"] = demand_insight
        result["summary"] = summary
        result["stats"] = {
            "raw_articles": len(raw_articles),
            "cleaned_articles": len(cleaned),
            "total_chunks": len(chunks),
            "supply_signals": len(retrieved["SUPPLY"]),
            "demand_signals": len(retrieved["DEMAND"])
        }
        
        update_progress("Complete!", 100)
        return result

    except Exception as e:
        logger.error("Pipeline breakdown: %s", e, exc_info=True)
        return {"status": "error", "error_message": f"System Error: {str(e)}"}
