"""
Alpaca Market Data Ingestion Layer

Fetches OHLCV candlestick data for the ticker universe via Alpaca's
multi-bar REST endpoint. Batches requests to stay within rate limits.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger("sentinel.ingest.alpaca")

BASE_URL = "https://data.alpaca.markets/v2"
BARS_ENDPOINT = f"{BASE_URL}/stocks/bars"

# Alpaca allows up to ~10k symbols per multi-bar request, but we batch
# in groups of 100 to keep response sizes manageable and retries cheap.
BATCH_SIZE = 100
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds


def _get_headers() -> dict:
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Accept": "application/json",
    }


def _fetch_bars_batch(
    symbols: list[str],
    timeframe: str = "15Min",
    lookback_bars: int = 200,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV bars for a batch of symbols.
    Returns dict mapping symbol -> DataFrame with columns:
    [timestamp, open, high, low, close, volume]
    """
    headers = _get_headers()

    # Calculate start time: lookback_bars * timeframe in minutes
    # 200 bars of 15-min data = 3000 minutes = ~50 hours
    timeframe_minutes = int(timeframe.replace("Min", ""))
    start = datetime.utcnow() - timedelta(minutes=lookback_bars * timeframe_minutes)
    end = datetime.utcnow()

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": lookback_bars,
        "adjustment": "raw",
        "feed": "iex",  # free tier uses IEX feed
        "sort": "asc",
    }

    all_bars = {}
    page_token = None

    for attempt in range(MAX_RETRIES):
        try:
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(BARS_ENDPOINT, headers=headers, params=params, timeout=30)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", RETRY_DELAY))
                logger.warning(f"Rate limited by Alpaca, waiting {retry_after}s")
                time.sleep(retry_after)
                continue

            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars", {})
            for symbol, bar_list in bars.items():
                if symbol not in all_bars:
                    all_bars[symbol] = []
                all_bars[symbol].extend(bar_list)

            # Handle pagination
            page_token = data.get("next_page_token")
            if page_token:
                continue

            break  # success, no more pages

        except requests.exceptions.RequestException as e:
            logger.error(f"Alpaca request failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                logger.error(f"Alpaca batch failed after {MAX_RETRIES} attempts")

    # Convert to DataFrames
    result = {}
    for symbol, bar_list in all_bars.items():
        if not bar_list:
            continue
        df = pd.DataFrame(bar_list)
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                 "l": "low", "c": "close", "v": "volume"})
        # Keep only needed columns (Alpaca returns 'n', 'vw' etc)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in cols if c in df.columns]]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        result[symbol] = df

    return result


def fetch_all_tickers(
    symbols: list[str],
    timeframe: str = "15Min",
    lookback_bars: int = 200,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all symbols, batched to respect rate limits.
    Returns dict mapping symbol -> DataFrame.
    Skips tickers with fewer than 50 bars.
    """
    logger.info(f"Fetching bars for {len(symbols)} tickers (batch size={BATCH_SIZE})")
    start_time = time.time()

    all_data = {}
    skipped = 0

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(symbols) + BATCH_SIZE - 1) // BATCH_SIZE
        logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} symbols")

        batch_data = _fetch_bars_batch(batch, timeframe, lookback_bars)

        for symbol, df in batch_data.items():
            if len(df) < 50:
                logger.debug(f"  Skipping {symbol}: only {len(df)} bars (need 50+)")
                skipped += 1
                continue
            all_data[symbol] = df

        # Small delay between batches to be nice to the API
        if i + BATCH_SIZE < len(symbols):
            time.sleep(0.5)

    elapsed = time.time() - start_time
    logger.info(
        f"Alpaca fetch complete: {len(all_data)} tickers with data, "
        f"{skipped} skipped, {elapsed:.1f}s elapsed"
    )
    return all_data


def is_market_hours() -> bool:
    """
    Quick check if US market is likely open.
    Not precise (doesn't account for holidays), but prevents
    wasted runs on weekends and overnight.
    """
    now = datetime.utcnow()
    # Weekends
    if now.weekday() >= 5:
        return False
    # Market hours: 9:30 AM - 4:00 PM ET = 13:30 - 20:00 UTC
    # Give 30-min buffer on each side for pre/post
    hour_utc = now.hour + now.minute / 60
    if hour_utc < 13.0 or hour_utc > 20.5:
        return False
    return True


def check_data_freshness(data: dict[str, pd.DataFrame], max_stale_minutes: int = 60) -> bool:
    """
    Check if the most recent bar across all tickers is reasonably fresh.
    If the newest data point is older than max_stale_minutes, the data
    is probably stale (weekend/holiday).
    """
    if not data:
        return False

    latest = None
    for df in data.values():
        if len(df) > 0:
            last_ts = df["timestamp"].iloc[-1]
            if latest is None or last_ts > latest:
                latest = last_ts

    if latest is None:
        return False

    age = datetime.utcnow() - latest.to_pydatetime().replace(tzinfo=None)
    if age > timedelta(minutes=max_stale_minutes):
        logger.warning(f"Data appears stale: newest bar is {age} old")
        return False

    return True
