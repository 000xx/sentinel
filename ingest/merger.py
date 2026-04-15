"""
News Merger — Dual-Source Waterfall

Implements the Finnhub-first / GDELT-fallback logic for equities,
and GDELT-first for ETFs and macro instruments. Merges, deduplicates,
and caps articles for LLM context.
"""

import logging
from typing import Optional

from ingest import finnhub, gdelt

logger = logging.getLogger("sentinel.ingest.merger")

# Minimum articles from primary source before triggering secondary
MIN_ARTICLES_THRESHOLD = 3

# Max articles to pass to the Narrator LLM
MAX_ARTICLES_FOR_LLM = 10


def fetch_news_for_anomaly(
    ticker_info: dict,
    tier: int,
) -> dict:
    """
    Fetch news for an anomalous ticker using the dual-source waterfall.

    Args:
        ticker_info: dict with symbol, name, sector, and optionally gdelt_query
        tier: integer tier (1-5)

    Returns dict:
    {
        "articles": list[dict],      # merged, deduped, capped
        "primary_source": str,       # "finnhub" or "gdelt"
        "secondary_used": bool,
        "gdelt_tone": float | None,  # average GDELT tone if available
        "total_found": int,
    }
    """
    symbol = ticker_info["symbol"]
    articles = []
    primary_source = "finnhub"
    secondary_used = False
    gdelt_tone = None

    if tier in (3, 4):
        # ETFs and macro: GDELT first (Finnhub has no ETF coverage)
        primary_source = "gdelt"
        gdelt_query = gdelt.build_query_for_ticker(ticker_info)
        gdelt_articles = gdelt.fetch_news(gdelt_query, timespan="60min")
        articles.extend(gdelt_articles)

        gdelt_tone = gdelt.get_average_tone(gdelt_articles)

        # Optionally supplement with Finnhub general news
        if len(articles) < MIN_ARTICLES_THRESHOLD:
            secondary_used = True
            general = finnhub.fetch_general_news(max_articles=5)
            articles.extend(general)

    else:
        # Equities (Tiers 1, 2, 5): Finnhub first
        primary_source = "finnhub"
        fh_articles = finnhub.fetch_company_news(symbol, lookback_hours=24)
        articles.extend(fh_articles)

        # If Finnhub returned sparse results, supplement with GDELT
        if len(fh_articles) < MIN_ARTICLES_THRESHOLD:
            secondary_used = True
            gdelt_query = gdelt.build_query_for_ticker(ticker_info)
            gdelt_articles = gdelt.fetch_news(
                gdelt_query,
                timespan="24h",
                source_country="US" if tier in (1, 2) else None,
            )
            articles.extend(gdelt_articles)
            gdelt_tone = gdelt.get_average_tone(gdelt_articles)

    total_found = len(articles)

    # Deduplicate by URL
    articles = _deduplicate(articles)

    # Sort by recency (newest first)
    articles = sorted(
        articles,
        key=lambda a: a.get("datetime", ""),
        reverse=True,
    )

    # Cap for LLM context
    articles = articles[:MAX_ARTICLES_FOR_LLM]

    source_label = primary_source
    if secondary_used:
        source_label = "both"

    logger.info(
        f"  News merge for {symbol}: {total_found} found, "
        f"{len(articles)} after dedup/cap (src: {source_label})"
    )

    return {
        "articles": articles,
        "primary_source": primary_source,
        "secondary_used": secondary_used,
        "news_source": source_label,
        "gdelt_tone": gdelt_tone,
        "total_found": total_found,
    }


def _deduplicate(articles: list[dict]) -> list[dict]:
    """Remove duplicate articles by URL."""
    seen_urls = set()
    unique = []
    for art in articles:
        url = art.get("url", "")
        if url and url in seen_urls:
            continue
        seen_urls.add(url)
        unique.append(art)
    return unique
