"""
Config Loader

Reads ticker_universe.json and provides helpers for:
- Getting all symbols (optionally filtered by tier)
- Looking up ticker metadata (name, sector, tier, gdelt_query)
- Determining which tickers to scan this cycle based on frequency
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sentinel.config")

CONFIG_FILE = Path("config/ticker_universe.json")

_config = None
_ticker_lookup = None


def load_config() -> dict:
    """Load and cache the ticker universe config."""
    global _config, _ticker_lookup

    if _config is not None:
        return _config

    with open(CONFIG_FILE) as f:
        _config = json.load(f)

    # Build fast lookup: symbol -> {name, sector, tier, gdelt_query, ...}
    _ticker_lookup = {}
    for tier_key, tier_data in _config.get("tiers", {}).items():
        tier_num = _tier_key_to_num(tier_key)
        for ticker in tier_data.get("tickers", []):
            symbol = ticker["symbol"]
            _ticker_lookup[symbol] = {
                **ticker,
                "tier": tier_num,
                "scan_frequency": tier_data.get("scan_frequency", "*/15 * * * *"),
                "news_source_priority": tier_data.get("news_source_priority", ["finnhub", "gdelt"]),
            }

    total = len(_ticker_lookup)
    logger.info(f"Config loaded: {total} tickers across {len(_config.get('tiers', {}))} tiers")
    return _config


def get_all_symbols() -> list[str]:
    """Get all symbols from all tiers."""
    load_config()
    return list(_ticker_lookup.keys())


def get_symbols_for_cycle() -> list[str]:
    """
    Get symbols that should be scanned this cycle.

    Tiers 1-4: every 15 minutes (always included)
    Tier 5: every 60 minutes (only at :00 minutes)
    """
    load_config()

    now = datetime.utcnow()
    is_hourly = now.minute < 15  # include tier 5 at the top of each hour

    symbols = []
    for symbol, info in _ticker_lookup.items():
        tier = info["tier"]
        if tier == 5 and not is_hourly:
            continue
        symbols.append(symbol)

    logger.info(
        f"Cycle symbols: {len(symbols)} "
        f"({'including' if is_hourly else 'excluding'} Tier 5)"
    )
    return symbols


def get_ticker_info(symbol: str) -> dict:
    """Look up metadata for a single ticker."""
    load_config()
    return _ticker_lookup.get(symbol, {"symbol": symbol, "name": symbol, "tier": 0})


def get_tier(symbol: str) -> int:
    """Get the tier number for a symbol."""
    return get_ticker_info(symbol).get("tier", 0)


def _tier_key_to_num(key: str) -> int:
    """Convert tier key name to number."""
    mapping = {
        "tier_1_bellwethers": 1,
        "tier_2_high_beta": 2,
        "tier_3_sector_etfs": 3,
        "tier_4_macro_indicators": 4,
        "tier_5_small_cap_movers": 5,
    }
    return mapping.get(key, 0)
