"""
Finnhub News Ingestion

Fetches company-specific news articles for tickers that have been
flagged as anomalous by Kronos. Only fires for tickers with good
Finnhub coverage (Tiers 1, 2, 5).
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import requests

logger = logging.getLogger("sentinel.ingest.finnhub")

BASE_URL = "https://finnhub.io/api/v1"
RATE_LIMIT_DELAY = 1.1  # slightly over 1s to stay under 60/min


def _get_api_key() -> str:
    key = os.environ.get("FINNHUB_API_KEY")
    if not key:
        raise EnvironmentError("FINNHUB_API_KEY must be set")
    return key


def fetch_company_news(
    symbol: str,
    lookback_hours: int = 24,
    max_articles: int = 15,
) -> list[dict]:
    """
    Fetch recent news articles for a single ticker from Finnhub.

    Returns list of dicts with keys:
    - title: str
    - summary: str
    - source: str
    - url: str
    - datetime: str (ISO format)
    - news_source: "finnhub" (tag for downstream merge)
    """
    api_key = _get_api_key()

    from_date = (datetime.utcnow() - timedelta(hours=lookback_hours)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"{BASE_URL}/company-news",
            params={
                "symbol": symbol,
                "from": from_date,
                "to": to_date,
                "token": api_key,
            },
            timeout=10,
        )

        if resp.status_code == 429:
            logger.warning(f"Finnhub rate limited for {symbol}, waiting...")
            time.sleep(5)
            return []

        resp.raise_for_status()
        raw_articles = resp.json()

        if not isinstance(raw_articles, list):
            logger.warning(f"Finnhub returned unexpected format for {symbol}")
            return []

        # Normalize to our standard format
        articles = []
        for art in raw_articles[:max_articles]:
            articles.append({
                "title": art.get("headline", ""),
                "summary": art.get("summary", ""),
                "source": art.get("source", "unknown"),
                "url": art.get("url", ""),
                "datetime": datetime.fromtimestamp(
                    art.get("datetime", 0)
                ).isoformat() if art.get("datetime") else "",
                "news_source": "finnhub",
            })

        logger.info(f"  Finnhub: {symbol} → {len(articles)} articles")
        return articles

    except requests.exceptions.RequestException as e:
        logger.error(f"Finnhub request failed for {symbol}: {e}")
        return []


def fetch_general_news(max_articles: int = 10) -> list[dict]:
    """
    Fetch general market news (not ticker-specific).
    Useful as fallback context for macro tickers.
    """
    api_key = _get_api_key()

    try:
        resp = requests.get(
            f"{BASE_URL}/news",
            params={"category": "general", "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()

        articles = []
        for art in raw[:max_articles]:
            articles.append({
                "title": art.get("headline", ""),
                "summary": art.get("summary", ""),
                "source": art.get("source", "unknown"),
                "url": art.get("url", ""),
                "datetime": datetime.fromtimestamp(
                    art.get("datetime", 0)
                ).isoformat() if art.get("datetime") else "",
                "news_source": "finnhub",
            })

        return articles

    except requests.exceptions.RequestException as e:
        logger.error(f"Finnhub general news failed: {e}")
        return []


def batch_fetch_news(
    symbols: list[str],
    lookback_hours: int = 24,
) -> dict[str, list[dict]]:
    """
    Fetch news for multiple symbols with rate limiting.
    Returns dict mapping symbol -> list of article dicts.
    """
    results = {}
    for i, symbol in enumerate(symbols):
        results[symbol] = fetch_company_news(symbol, lookback_hours)

        # Rate limit: ~60 calls/min → 1 call/sec
        if i < len(symbols) - 1:
            time.sleep(RATE_LIMIT_DELAY)

    return results
