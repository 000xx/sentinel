"""
GDELT DOC 2.0 API News Ingestion

Searches global news via GDELT's full-text search API. Unlike Finnhub,
GDELT searches by keyword (company name, brand terms) rather than
ticker symbol. It covers 250K+ outlets in 152 languages.

GDELT is the primary news source for ETFs and macro instruments (Tiers 3, 4)
and serves as a supplement for equities when Finnhub returns sparse results.
"""

import time
import logging
import urllib.parse
from typing import Optional

import requests

logger = logging.getLogger("sentinel.ingest.gdelt")

BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
RATE_LIMIT_DELAY = 1.5  # courtesy delay between requests
MAX_RETRIES = 2


def fetch_news(
    query: str,
    timespan: str = "60min",
    max_records: int = 50,
    source_country: Optional[str] = None,
    tone_threshold: Optional[float] = None,
) -> list[dict]:
    """
    Search GDELT for articles matching the query.

    Args:
        query: Search terms (company name, keywords, etc.)
        timespan: How far back to look ("60min", "24h", "7d")
        max_records: Max articles to return (cap 250)
        source_country: Optional filter e.g. "US"
        tone_threshold: If set, filter for articles with toneabs > this value

    Returns list of dicts with keys:
    - title: str
    - url: str
    - source: str (domain)
    - datetime: str
    - language: str
    - country: str
    - tone: float (GDELT sentiment score, negative=bearish positive=bullish)
    - news_source: "gdelt"
    """
    # Build query string with optional operators
    full_query = query
    if source_country:
        full_query += f" sourcecountry:{source_country}"
    if tone_threshold:
        full_query += f" toneabs>{tone_threshold}"

    params = {
        "query": full_query,
        "mode": "artlist",
        "maxrecords": min(max_records, 250),
        "format": "json",
        "timespan": timespan,
        "sort": "datedesc",
    }

    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(RATE_LIMIT_DELAY)
            resp = requests.get(url, timeout=15)

            if resp.status_code == 429:
                logger.warning("GDELT rate limited, backing off...")
                time.sleep(5 * (attempt + 1))
                continue

            if resp.status_code != 200:
                logger.warning(f"GDELT returned {resp.status_code} for query: {query[:50]}...")
                return []

            data = resp.json()
            raw_articles = data.get("articles", [])

            if not raw_articles:
                logger.debug(f"  GDELT: no articles for query: {query[:50]}...")
                return []

            articles = []
            for art in raw_articles:
                articles.append({
                    "title": art.get("title", ""),
                    "url": art.get("url", ""),
                    "source": art.get("domain", "unknown"),
                    "datetime": art.get("seendate", ""),
                    "language": art.get("language", ""),
                    "country": art.get("sourcecountry", ""),
                    "tone": _parse_tone(art.get("tone", "")),
                    "news_source": "gdelt",
                })

            logger.info(f"  GDELT: '{query[:40]}...' → {len(articles)} articles")
            return articles

        except requests.exceptions.Timeout:
            logger.warning(f"GDELT timeout (attempt {attempt + 1}) for: {query[:50]}...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(3)
        except requests.exceptions.RequestException as e:
            logger.error(f"GDELT request failed: {e}")
            return []
        except (ValueError, KeyError) as e:
            logger.error(f"GDELT response parse error: {e}")
            return []

    return []


def _parse_tone(tone_str) -> Optional[float]:
    """Parse GDELT tone value (may be string or float)."""
    if tone_str is None or tone_str == "":
        return None
    try:
        return round(float(tone_str), 2)
    except (ValueError, TypeError):
        return None


def build_query_for_ticker(ticker_info: dict) -> str:
    """
    Build a GDELT search query from ticker config.

    For ETFs/macro (tiers 3, 4): use the gdelt_query field from config.
    For equities (tiers 1, 2, 5): use the company name.
    """
    # If the config has a pre-built GDELT query, use it
    if "gdelt_query" in ticker_info:
        return ticker_info["gdelt_query"]

    # Otherwise construct from company name
    name = ticker_info.get("name", ticker_info.get("symbol", ""))

    # Strip common suffixes that add noise
    for suffix in [" Inc", " Corp", " Co", " Ltd", " PLC", " SA", " SE",
                   " NV", " AG", " Holdings", " Group"]:
        name = name.replace(suffix, "")

    return name.strip()


def get_average_tone(articles: list[dict]) -> Optional[float]:
    """Calculate average GDELT tone score across articles."""
    tones = [a["tone"] for a in articles if a.get("tone") is not None]
    if not tones:
        return None
    return round(sum(tones) / len(tones), 2)
