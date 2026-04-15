"""
Delivery Layer

Two outputs:
1. Append alerts to site/alerts.json (for the Sentinel web dashboard)
2. Optionally push to Discord webhook (if URL is configured)

The JSON file is the primary delivery mechanism — it accumulates
alerts over time and is served by GitHub Pages. The Discord webhook
is a secondary notification channel.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("sentinel.delivery")

ALERTS_FILE = Path("site/alerts.json")
MAX_DISCORD_RETRIES = 3


def deliver_alerts(alerts: list[dict]) -> dict:
    """
    Deliver all alerts for this cycle.
    Writes to JSON file and optionally pushes to Discord.

    Args:
        alerts: list of fully-formed alert dicts

    Returns summary dict with counts.
    """
    if not alerts:
        logger.info("No alerts to deliver this cycle")
        return {"json_written": 0, "discord_pushed": 0, "discord_failed": 0}

    # Write to JSON file
    json_count = _append_to_json(alerts)

    # Push to Discord
    discord_url = os.environ.get("DISCORD_WEBHOOK_URL")
    discord_ok = 0
    discord_fail = 0

    if discord_url:
        for alert in alerts:
            success = _push_to_discord(alert, discord_url)
            if success:
                discord_ok += 1
            else:
                discord_fail += 1
            time.sleep(0.5)  # respect Discord rate limits

    logger.info(
        f"Delivery complete: {json_count} to JSON, "
        f"{discord_ok} to Discord ({discord_fail} failed)"
    )

    return {
        "json_written": json_count,
        "discord_pushed": discord_ok,
        "discord_failed": discord_fail,
    }


def _append_to_json(alerts: list[dict]) -> int:
    """
    Append alerts to the JSON file. Never deletes — only accumulates.
    Creates the file if it doesn't exist.
    """
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if ALERTS_FILE.exists():
        try:
            with open(ALERTS_FILE) as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, IOError):
            existing = []

    # Prepend new alerts (newest first in the file)
    combined = alerts + existing

    with open(ALERTS_FILE, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    logger.info(f"  JSON: appended {len(alerts)} alerts (total: {len(combined)})")
    return len(alerts)


def build_alert_payload(
    anomaly: dict,
    ticker_info: dict,
    narrative: dict,
    news_result: dict,
    tier: int,
) -> dict:
    """
    Build the standardized alert payload used by both the JSON file
    and Discord webhook.
    """
    return {
        "id": f"alert-{anomaly['symbol']}-{int(time.time() * 1000)}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "ticker": anomaly["symbol"],
        "company": ticker_info.get("name", anomaly["symbol"]),
        "sector": ticker_info.get("sector", "Unknown"),
        "tier": tier,
        "sigma": anomaly["sigma"],
        "sentiment": narrative.get("sentiment", "UNCERTAIN"),
        "newsSource": news_result.get("news_source", "unknown"),
        "lastClose": anomaly["last_close"],
        "forecastHigh": str(anomaly["forecast_high"]),
        "forecastLow": str(anomaly["forecast_low"]),
        "baselineRange": str(anomaly["baseline_mean"]),
        "catalysts": narrative.get("catalysts"),
        "verdict": narrative.get("verdict", ""),
        "isEscalating": anomaly.get("is_escalating", False),
        "gdeltTone": news_result.get("gdelt_tone"),
        "articleCount": news_result.get("total_found", 0),
    }


# ──────────────────────────────────────────────────────────────────
# Discord Webhook
# ──────────────────────────────────────────────────────────────────

SENTIMENT_COLORS = {
    "BULLISH": 0x00E676,
    "BEARISH": 0xFF1744,
    "UNCERTAIN": 0xFFAB00,
    "EVENT-DRIVEN": 0x00B0FF,
}

TIER_LABELS = {
    1: "T1·BELL",
    2: "T2·BETA",
    3: "T3·ETF",
    4: "T4·MACRO",
    5: "T5·SMCAP",
}


def _push_to_discord(alert: dict, webhook_url: str) -> bool:
    """Push a single alert to Discord as a rich embed."""
    sentiment = alert.get("sentiment", "UNCERTAIN")
    color = SENTIMENT_COLORS.get(sentiment, 0xFFAB00)
    tier_label = TIER_LABELS.get(alert.get("tier", 0), "?")

    # Build catalyst text
    catalysts = alert.get("catalysts")
    if catalysts and isinstance(catalysts, list):
        catalyst_text = "\n".join(f"▸ {c}" for c in catalysts)
    else:
        catalyst_text = "_No catalyst identified_"

    title = f"{'⚡ ESCALATING' if alert.get('isEscalating') else '🚨 ANOMALY'}: {alert['ticker']}"

    # Build fields
    fields = [
        {
            "name": "Signal",
            "value": (
                f"**{alert['sigma']}σ** above baseline\n"
                f"Range: ${alert['forecastLow']} — ${alert['forecastHigh']}\n"
                f"Baseline: ${alert['baselineRange']}"
            ),
            "inline": True,
        },
        {
            "name": "Sentiment",
            "value": f"**{sentiment}**",
            "inline": True,
        },
        {
            "name": "Analysis",
            "value": f"_{alert.get('verdict', '')}_\n\n{catalyst_text}",
            "inline": False,
        },
    ]

    if alert.get("gdeltTone") is not None:
        fields[1]["value"] += f"\nGDELT Tone: {alert['gdeltTone']}"

    embed = {
        "embeds": [{
            "title": title,
            "description": f"{alert.get('company', '')} · {alert.get('sector', '')}",
            "color": color,
            "fields": fields,
            "footer": {
                "text": f"Sentinel V1 · {tier_label} · src:{alert.get('newsSource', '?')} · {alert.get('articleCount', 0)} articles"
            },
            "timestamp": alert.get("timestamp", datetime.utcnow().isoformat()),
        }],
    }

    for attempt in range(MAX_DISCORD_RETRIES):
        try:
            resp = requests.post(webhook_url, json=embed, timeout=10)

            if resp.status_code == 204:
                return True

            if resp.status_code == 429:
                retry_after = resp.json().get("retry_after", 2)
                logger.warning(f"Discord rate limited, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            logger.warning(f"Discord webhook returned {resp.status_code}")
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"Discord push failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_DISCORD_RETRIES - 1:
                time.sleep(2 ** attempt)

    return False
