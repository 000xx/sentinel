"""
Sentinel V1 — Main Pipeline Orchestrator

The 15-Minute Anomaly Tracker

Execution flow:
1. Load config + state
2. Check market hours (skip if closed)
3. Fetch OHLCV bars from Alpaca
4. Load Kronos-mini and run inference
5. Detect anomalies against rolling baselines
6. For each anomaly: fetch news (Finnhub + GDELT), narrate (Groq/Gemini)
7. Deliver alerts (JSON file + Discord)
8. Save state
"""

import sys
import time
import logging
from datetime import datetime

# ── Logging ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("sentinel")

# ── Imports ──────────────────────────────────────────────────────
from config_loader import load_config, get_symbols_for_cycle, get_ticker_info, get_tier
from state_manager import load_state, save_state
from ingest.alpaca import fetch_all_tickers, is_market_hours, check_data_freshness
from ingest.merger import fetch_news_for_anomaly
from agents.quant import load_model, run_inference, detect_anomalies
from agents.narrator import narrate
from delivery.output import deliver_alerts, build_alert_payload

# ── Config ───────────────────────────────────────────────────────
SIGMA_THRESHOLD = 2.0          # minimum σ to flag as anomaly
MAX_NARRATIONS_PER_CYCLE = 10  # cap LLM calls per run
LOOKBACK_BARS = 200            # bars of history to fetch per ticker
FORECAST_STEPS = 4             # bars to forecast forward


def main():
    run_start = time.time()
    logger.info("=" * 60)
    logger.info("SENTINEL V1 — Pipeline cycle starting")
    logger.info(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    logger.info("=" * 60)

    # ── Step 1: Load config + state ──────────────────────────────
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    state = load_state()

    # ── Step 2: Market hours check ───────────────────────────────
    # if not is_market_hours():
    #     logger.info("Market is closed — skipping cycle")
    #     # Still save state to update last_run timestamp
    #     save_state(state)
    #     _log_summary(run_start, 0, 0, 0, skipped=True)
    #     return

    # ── Step 3: Determine which tickers to scan ──────────────────
    symbols = get_symbols_for_cycle()
    if not symbols:
        logger.warning("No symbols to scan this cycle")
        save_state(state)
        return

    # ── Step 4: Fetch market data from Alpaca ────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info("PHASE 1: DATA INGESTION")
    logger.info(f"{'─' * 40}")

    try:
        ohlcv_data = fetch_all_tickers(
            symbols=symbols,
            timeframe="15Min",
            lookback_bars=LOOKBACK_BARS,
        )
    except Exception as e:
        logger.error(f"Alpaca fetch failed: {e}")
        save_state(state)
        _log_summary(run_start, 0, 0, 0, error="alpaca_down")
        return

    if not ohlcv_data:
        logger.warning("Alpaca returned no data — aborting cycle")
        save_state(state)
        return

    if not check_data_freshness(ohlcv_data, max_stale_minutes=120):
        logger.warning("Data appears stale (weekend/holiday?) — skipping cycle")
        save_state(state)
        _log_summary(run_start, len(ohlcv_data), 0, 0, skipped=True)
        return

    logger.info(f"Valid data for {len(ohlcv_data)} / {len(symbols)} tickers")

    # ── Step 5: Load Kronos and run inference ────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info("PHASE 2: KRONOS INFERENCE")
    logger.info(f"{'─' * 40}")

    if not load_model(device="cpu", max_context=2048):
        logger.error("Kronos model failed to load — aborting")
        save_state(state)
        _log_summary(run_start, len(ohlcv_data), 0, 0, error="kronos_load_failed")
        return

    try:
        inference_results = run_inference(ohlcv_data, forecast_steps=FORECAST_STEPS)
    except Exception as e:
        logger.error(f"Kronos inference failed: {e}")
        save_state(state)
        _log_summary(run_start, len(ohlcv_data), 0, 0, error="kronos_inference_failed")
        return

    # ── Step 6: Detect anomalies ─────────────────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info("PHASE 3: ANOMALY DETECTION")
    logger.info(f"{'─' * 40}")

    anomalies = detect_anomalies(
        inference_results=inference_results,
        ohlcv_data=ohlcv_data,
        state=state,
        sigma_threshold=SIGMA_THRESHOLD,
    )

    if not anomalies:
        logger.info("No anomalies detected this cycle")
        save_state(state)
        _log_summary(run_start, len(ohlcv_data), len(inference_results), 0)
        return

    logger.info(f"Anomalies detected: {len(anomalies)}")
    for a in anomalies[:20]:
        esc = " ⚡ESCALATING" if a["is_escalating"] else ""
        logger.info(f"  {a['symbol']:8s}  σ={a['sigma']:5.2f}  "
                     f"range=${a['forecast_range']:.4f}  "
                     f"close=${a['last_close']:.2f}{esc}")

    # ── Step 7: News + Narration for top anomalies ───────────────
    logger.info(f"\n{'─' * 40}")
    logger.info("PHASE 4: NEWS FETCH + NARRATION")
    logger.info(f"{'─' * 40}")

    alert_payloads = []
    narration_count = 0

    for anomaly in anomalies:
        symbol = anomaly["symbol"]
        ticker_info = get_ticker_info(symbol)
        tier = ticker_info.get("tier", 0)

        # Fetch news (dual-source waterfall)
        try:
            news_result = fetch_news_for_anomaly(ticker_info, tier)
        except Exception as e:
            logger.error(f"News fetch failed for {symbol}: {e}")
            news_result = {
                "articles": [],
                "news_source": "none",
                "gdelt_tone": None,
                "total_found": 0,
            }

        # Narrate (capped at MAX_NARRATIONS_PER_CYCLE)
        if narration_count < MAX_NARRATIONS_PER_CYCLE:
            try:
                narrative = narrate(
                    ticker=symbol,
                    company=ticker_info.get("name", symbol),
                    sigma=anomaly["sigma"],
                    price_data={
                        "last_close": anomaly["last_close"],
                        "forecast_high": anomaly["forecast_high"],
                        "forecast_low": anomaly["forecast_low"],
                        "baseline_mean": anomaly["baseline_mean"],
                    },
                    articles=news_result["articles"],
                )
                narration_count += 1
            except Exception as e:
                logger.error(f"Narration failed for {symbol}: {e}")
                narrative = {
                    "verdict": "Narration failed this cycle.",
                    "catalysts": [
                        "LLM call failed — raw signal only.",
                        f"Sigma: {anomaly['sigma']}σ above baseline.",
                        "Check next cycle for narrative update.",
                    ],
                    "sentiment": "UNCERTAIN",
                }
        else:
            # Over narration cap — signal-only alert
            narrative = {
                "verdict": "QUEUED — narrative pending (LLM budget exceeded this cycle).",
                "catalysts": None,
                "sentiment": "UNCERTAIN",
            }
            logger.info(f"  {symbol}: narration skipped (over cap)")

        # Build payload
        payload = build_alert_payload(
            anomaly=anomaly,
            ticker_info=ticker_info,
            narrative=narrative,
            news_result=news_result,
            tier=tier,
        )
        alert_payloads.append(payload)

    # ── Step 8: Deliver ──────────────────────────────────────────
    logger.info(f"\n{'─' * 40}")
    logger.info("PHASE 5: DELIVERY")
    logger.info(f"{'─' * 40}")

    delivery_result = deliver_alerts(alert_payloads)

    # ── Step 9: Save state ───────────────────────────────────────
    save_state(state)

    # ── Summary ──────────────────────────────────────────────────
    _log_summary(
        run_start,
        tickers_scanned=len(ohlcv_data),
        inferences=len(inference_results),
        anomalies_found=len(anomalies),
        alerts_delivered=len(alert_payloads),
        narrations=narration_count,
        delivery=delivery_result,
    )


def _log_summary(
    run_start: float,
    tickers_scanned: int = 0,
    inferences: int = 0,
    anomalies_found: int = 0,
    alerts_delivered: int = 0,
    narrations: int = 0,
    delivery: dict = None,
    skipped: bool = False,
    error: str = None,
):
    """Print a clean run summary."""
    elapsed = time.time() - run_start

    logger.info(f"\n{'=' * 60}")
    logger.info("SENTINEL V1 — Cycle Complete")
    logger.info(f"{'=' * 60}")

    if skipped:
        logger.info(f"  Status:           SKIPPED (market closed or stale data)")
    elif error:
        logger.info(f"  Status:           ERROR ({error})")
    else:
        logger.info(f"  Status:           OK")

    logger.info(f"  Duration:         {elapsed:.1f}s")
    logger.info(f"  Tickers scanned:  {tickers_scanned}")
    logger.info(f"  Inferences:       {inferences}")
    logger.info(f"  Anomalies:        {anomalies_found}")
    logger.info(f"  Alerts delivered: {alerts_delivered}")
    logger.info(f"  Narrations:       {narrations}")

    if delivery:
        logger.info(f"  JSON written:     {delivery.get('json_written', 0)}")
        logger.info(f"  Discord pushed:   {delivery.get('discord_pushed', 0)}")
        if delivery.get("discord_failed", 0) > 0:
            logger.info(f"  Discord failed:   {delivery['discord_failed']}")

    logger.info(f"{'=' * 60}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline crashed: {e}")
        sys.exit(1)
