"""
Agent 2: The Narrator

Takes raw news articles for an anomalous ticker and synthesizes them
into a concise 3-bullet catalyst summary using Groq (primary) or
Gemini (fallback).

Output format:
{
    "verdict": str,       # single-sentence catalyst assessment
    "catalysts": [str],   # exactly 3 bullet points
    "sentiment": str,     # BULLISH | BEARISH | UNCERTAIN | EVENT-DRIVEN
}
"""

import os
import json
import time
import logging
import re
from typing import Optional

import requests

logger = logging.getLogger("sentinel.agents.narrator")

SYSTEM_PROMPT = """You are a senior financial analyst at a quantitative hedge fund. You write
concise, actionable intelligence briefs. No speculation beyond what the data supports.

You MUST respond in valid JSON with exactly these keys:
{
  "verdict": "single sentence — is this a real catalyst or noise?",
  "catalysts": ["bullet 1 (max 25 words)", "bullet 2", "bullet 3"],
  "sentiment": "BULLISH | BEARISH | UNCERTAIN | EVENT-DRIVEN"
}

Do NOT include any text outside the JSON object. No markdown, no explanation."""

USER_PROMPT_TEMPLATE = """The stock {ticker} ({company}) is experiencing abnormal volatility.
Kronos forecasting model has flagged a {sigma}σ deviation above normal expected price range.

Current price action:
- Last close: ${last_close}
- Forecasted range: ${forecast_high} - ${forecast_low}
- Historical average range: ${baseline}

Below are the {n} most recent news articles from the past 24 hours:

{articles_json}

Based ONLY on the provided headlines and summaries, produce your analysis as JSON."""

NO_NEWS_PROMPT = """The stock {ticker} ({company}) is experiencing abnormal volatility.
Kronos forecasting model has flagged a {sigma}σ deviation above normal expected price range.

Current price action:
- Last close: ${last_close}
- Forecasted range: ${forecast_high} - ${forecast_low}
- Historical average range: ${baseline}

No news articles were found for this ticker in the past 24 hours from either
Finnhub or GDELT sources.

Respond with JSON. State that no catalyst was identified and suggest checking
SEC filings or EDGAR for unreported events."""


def narrate(
    ticker: str,
    company: str,
    sigma: float,
    price_data: dict,
    articles: list[dict],
) -> dict:
    """
    Generate a 3-bullet catalyst narrative for an anomalous ticker.

    Args:
        ticker: symbol
        company: company name
        sigma: z-score
        price_data: dict with last_close, forecast_high, forecast_low, baseline_mean
        articles: list of article dicts from the news merger

    Returns dict with verdict, catalysts, sentiment.
    Falls back to raw headlines if LLM fails.
    """
    # Try Groq first, then Gemini
    result = _call_groq(ticker, company, sigma, price_data, articles)
    if result:
        return result

    logger.warning(f"Groq failed for {ticker}, trying Gemini fallback...")
    result = _call_gemini(ticker, company, sigma, price_data, articles)
    if result:
        return result

    # Both failed — return raw headline fallback
    logger.error(f"All LLM providers failed for {ticker}, using headline fallback")
    return _headline_fallback(articles)


def _build_prompt(
    ticker: str,
    company: str,
    sigma: float,
    price_data: dict,
    articles: list[dict],
) -> str:
    """Build the user prompt."""
    if not articles:
        return NO_NEWS_PROMPT.format(
            ticker=ticker,
            company=company,
            sigma=sigma,
            last_close=price_data.get("last_close", 0),
            forecast_high=price_data.get("forecast_high", 0),
            forecast_low=price_data.get("forecast_low", 0),
            baseline=price_data.get("baseline_mean", 0),
        )

    # Slim down articles for the prompt
    slim_articles = []
    for a in articles[:10]:
        slim_articles.append({
            "title": a.get("title", "")[:200],
            "source": a.get("source", ""),
            "summary": a.get("summary", "")[:300],
        })

    return USER_PROMPT_TEMPLATE.format(
        ticker=ticker,
        company=company,
        sigma=sigma,
        last_close=price_data.get("last_close", 0),
        forecast_high=price_data.get("forecast_high", 0),
        forecast_low=price_data.get("forecast_low", 0),
        baseline=price_data.get("baseline_mean", 0),
        n=len(slim_articles),
        articles_json=json.dumps(slim_articles, indent=2),
    )


def _call_groq(
    ticker: str,
    company: str,
    sigma: float,
    price_data: dict,
    articles: list[dict],
) -> Optional[dict]:
    """Call Groq API for narration."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.debug("GROQ_API_KEY not set, skipping Groq")
        return None

    prompt = _build_prompt(ticker, company, sigma, price_data, articles)

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama3-70b-8192",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 400,
            },
            timeout=30,
        )

        if resp.status_code == 429:
            logger.warning("Groq rate limited")
            return None

        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return _parse_llm_response(content)

    except Exception as e:
        logger.error(f"Groq API error for {ticker}: {e}")
        return None


def _call_gemini(
    ticker: str,
    company: str,
    sigma: float,
    price_data: dict,
    articles: list[dict],
) -> Optional[dict]:
    """Call Google Gemini API as fallback."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.debug("GEMINI_API_KEY not set, skipping Gemini")
        return None

    prompt = _build_prompt(ticker, company, sigma, price_data, articles)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"

    try:
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": full_prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 400,
                },
            },
            timeout=30,
        )

        if resp.status_code == 429:
            logger.warning("Gemini rate limited")
            return None

        resp.raise_for_status()
        data = resp.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return _parse_llm_response(content)

    except Exception as e:
        logger.error(f"Gemini API error for {ticker}: {e}")
        return None


def _parse_llm_response(content: str) -> Optional[dict]:
    """Parse LLM JSON response, handling markdown fences and quirks."""
    # Strip markdown code fences
    content = content.strip()
    content = re.sub(r"^```json\s*", "", content)
    content = re.sub(r"^```\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    try:
        parsed = json.loads(content)

        # Validate structure
        verdict = parsed.get("verdict", "No verdict provided.")
        catalysts = parsed.get("catalysts", [])
        sentiment = parsed.get("sentiment", "UNCERTAIN")

        # Ensure exactly 3 catalysts
        if not isinstance(catalysts, list):
            catalysts = [str(catalysts)]
        while len(catalysts) < 3:
            catalysts.append("No additional catalyst identified.")
        catalysts = catalysts[:3]

        # Validate sentiment
        valid_sentiments = {"BULLISH", "BEARISH", "UNCERTAIN", "EVENT-DRIVEN"}
        if sentiment not in valid_sentiments:
            sentiment = "UNCERTAIN"

        return {
            "verdict": verdict,
            "catalysts": catalysts,
            "sentiment": sentiment,
        }

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response: {e}")
        logger.debug(f"Raw content: {content[:500]}")
        return None


def _headline_fallback(articles: list[dict]) -> dict:
    """
    Fallback when all LLM providers fail.
    Returns raw headlines as catalyst bullets.
    """
    if not articles:
        return {
            "verdict": "No catalyst identified — check SEC filings / EDGAR.",
            "catalysts": [
                "LLM narration unavailable this cycle.",
                "No news articles found from Finnhub or GDELT.",
                "Monitor for follow-up in next pipeline cycle.",
            ],
            "sentiment": "UNCERTAIN",
        }

    headlines = [a.get("title", "Untitled")[:80] for a in articles[:3]]
    while len(headlines) < 3:
        headlines.append("No additional headline available.")

    return {
        "verdict": "LLM narration unavailable — raw headlines shown below.",
        "catalysts": headlines,
        "sentiment": "UNCERTAIN",
    }
