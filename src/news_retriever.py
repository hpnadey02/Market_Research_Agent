# ==============================================================================
# news_retriever.py — Multi-Source News Retrieval Engine
# ==============================================================================
# Retrieves commodity/metal news from two sources:
#   1. NewsAPI.org — structured API with date filtering
#   2. SteelOrbis   — web scraping of latest news page
#
# Both sources are wrapped in fail-safe try/except blocks with retry logic
# and graceful degradation (if one fails, the other still returns results).
# ==============================================================================

import time
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
from bs4 import BeautifulSoup

from src.config import (
    NEWS_API_KEY, STEELORBIS_URL, NEWS_LOOKBACK_DAYS,
    REQUEST_TIMEOUT, MAX_RETRIES,
)
from src.logger import get_logger

logger = get_logger("NewsRetriever")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NewsAPI Retrieval
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _fetch_newsapi(metal: str, country: str) -> List[Dict[str, Any]]:
    """
    Fetch articles from NewsAPI.org using the query [COUNTRY] + [METAL] + news.
    Filters to the last NEWS_LOOKBACK_DAYS days.

    Returns a list of article dicts with keys:
        title, description, content, url, published_at, source
    """
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY not set — skipping NewsAPI source.")
        return []

    query = f"{country} {metal} news"
    from_date = (datetime.utcnow() - timedelta(days=NEWS_LOOKBACK_DAYS)).strftime(
        "%Y-%m-%d"
    )
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 50,
        "apiKey": NEWS_API_KEY,
    }

    articles_out: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "NewsAPI request (attempt %d/%d): query='%s'",
                attempt, MAX_RETRIES, query,
            )
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            if data.get("status") != "ok":
                logger.error("NewsAPI returned non-ok status: %s", data)
                return []

            for art in data.get("articles", []):
                articles_out.append({
                    "title": art.get("title", ""),
                    "description": art.get("description", ""),
                    "content": art.get("content", ""),
                    "url": art.get("url", ""),
                    "published_at": art.get("publishedAt", ""),
                    "source": art.get("source", {}).get("name", "NewsAPI"),
                })

            logger.info("NewsAPI returned %d articles.", len(articles_out))
            return articles_out

        except requests.exceptions.Timeout:
            logger.warning("NewsAPI timeout (attempt %d/%d)", attempt, MAX_RETRIES)
        except requests.exceptions.ConnectionError:
            logger.warning("NewsAPI connection error (attempt %d/%d)", attempt, MAX_RETRIES)
        except requests.exceptions.HTTPError as e:
            logger.error("NewsAPI HTTP error: %s", e)
            return []  # Non-retryable (likely 401/429)
        except Exception as e:
            logger.error("NewsAPI unexpected error: %s", e, exc_info=True)
            return []

        time.sleep(2 * attempt)  # Exponential back-off

    logger.error("NewsAPI failed after %d retries.", MAX_RETRIES)
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SteelOrbis Web Scraper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _scrape_steelorbis() -> List[Dict[str, Any]]:
    """
    Scrape the SteelOrbis latest-news page for article headlines, dates,
    and snippet text.

    Returns a list of article dicts with keys:
        title, description, content, url, published_at, source
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    articles_out: List[Dict[str, Any]] = []

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "SteelOrbis scrape (attempt %d/%d): %s",
                attempt, MAX_RETRIES, STEELORBIS_URL,
            )
            resp = requests.get(
                STEELORBIS_URL, headers=headers, timeout=REQUEST_TIMEOUT
            )
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")

            # ── Parse article cards ──────────────────────────────────────────
            # SteelOrbis typically lists news in <div> or <article> containers
            # with headline links and date spans.  We try multiple selectors
            # to be robust against minor layout changes.
            news_items = (
                soup.select("div.news-item")
                or soup.select("div.news-list-item")
                or soup.select("article")
                or soup.select("div.col-md-12 div.row")
            )

            if not news_items:
                # Fallback: grab all substantial links on the page
                all_links = soup.select("a[href*='/steel-news/']")
                for link in all_links:
                    title_text = link.get_text(strip=True)
                    if len(title_text) > 30:  # Filter out nav links
                        articles_out.append({
                            "title": title_text,
                            "description": title_text,
                            "content": title_text,
                            "url": f"https://www.steelorbis.com{link.get('href', '')}",
                            "published_at": "",
                            "source": "SteelOrbis",
                        })
            else:
                for item in news_items:
                    # Title
                    title_tag = item.find("a") or item.find("h2") or item.find("h3")
                    title_text = title_tag.get_text(strip=True) if title_tag else ""

                    # URL
                    href = ""
                    if title_tag and title_tag.name == "a":
                        href = title_tag.get("href", "")
                    elif title_tag:
                        a_tag = title_tag.find("a")
                        if a_tag:
                            href = a_tag.get("href", "")
                    if href and not href.startswith("http"):
                        href = f"https://www.steelorbis.com{href}"

                    # Date
                    date_tag = item.find("span", class_="date") or item.find("time")
                    date_text = date_tag.get_text(strip=True) if date_tag else ""

                    # Snippet / description
                    desc_tag = item.find("p") or item.find("div", class_="summary")
                    desc_text = desc_tag.get_text(strip=True) if desc_tag else title_text

                    if title_text and len(title_text) > 10:
                        articles_out.append({
                            "title": title_text,
                            "description": desc_text,
                            "content": desc_text,
                            "url": href,
                            "published_at": date_text,
                            "source": "SteelOrbis",
                        })

            logger.info("SteelOrbis returned %d articles.", len(articles_out))
            return articles_out

        except requests.exceptions.Timeout:
            logger.warning(
                "SteelOrbis timeout (attempt %d/%d)", attempt, MAX_RETRIES
            )
        except requests.exceptions.ConnectionError:
            logger.warning(
                "SteelOrbis connection error (attempt %d/%d)", attempt, MAX_RETRIES
            )
        except Exception as e:
            logger.error("SteelOrbis unexpected error: %s", e, exc_info=True)
            return []

        time.sleep(2 * attempt)

    logger.error("SteelOrbis scraping failed after %d retries.", MAX_RETRIES)
    return []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def retrieve_news(metal: str, country: str) -> List[Dict[str, Any]]:
    """
    Orchestrate multi-source news retrieval with graceful degradation.

    If one source fails, the other still contributes.  If both fail,
    an empty list is returned and the pipeline will raise a stage error.

    Args:
        metal: Metal name (e.g., 'Steel').
        country: Country name (e.g., 'India').

    Returns:
        Combined list of raw article dicts from all sources.
    """
    logger.info("Starting news retrieval for metal='%s', country='%s'", metal, country)

    newsapi_articles = _fetch_newsapi(metal, country)
    steelorbis_articles = _scrape_steelorbis()

    combined = newsapi_articles + steelorbis_articles
    logger.info(
        "Total raw articles retrieved: %d (NewsAPI=%d, SteelOrbis=%d)",
        len(combined), len(newsapi_articles), len(steelorbis_articles),
    )
    return combined
