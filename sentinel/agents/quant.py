"""
Agent 1: The Quant (Kronos)

Loads Kronos-mini foundation model and runs volatility forecasting
on OHLCV candlestick data. Detects anomalies by comparing forecasted
price range against a rolling historical baseline.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sentinel.agents.quant")

# Global model references (loaded once per run)
_predictor = None
_model_loaded = False


def load_model(device: str = "cpu", max_context: int = 2048) -> bool:
    """
    Load Kronos-mini model and tokenizer from Hugging Face cache.
    Returns True if successful.
    """
    global _predictor, _model_loaded

    if _model_loaded:
        return True

    logger.info("Loading Kronos-mini model...")
    start = time.time()

    try:
        # Kronos model code must be on PYTHONPATH (cloned from GitHub)
        from model import Kronos, KronosTokenizer, KronosPredictor

        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")

        _predictor = KronosPredictor(
            model, tokenizer,
            device=device,
            max_context=max_context,
        )
        _model_loaded = True

        elapsed = time.time() - start
        logger.info(f"Kronos-mini loaded in {elapsed:.1f}s (device={device})")
        return True

    except ImportError as e:
        logger.error(
            f"Cannot import Kronos model code: {e}. "
            "Ensure kronos_src is on PYTHONPATH."
        )
        return False
    except Exception as e:
        logger.error(f"Kronos model load failed: {e}")
        return False


def run_inference(
    ohlcv_data: dict[str, pd.DataFrame],
    forecast_steps: int = 4,
) -> dict[str, dict]:
    """
    Run Kronos inference on all tickers.

    Args:
        ohlcv_data: dict mapping symbol -> DataFrame (open, high, low, close, volume, timestamp)
        forecast_steps: number of future bars to predict

    Returns dict mapping symbol -> {
        "forecast_df": pd.DataFrame,
        "forecast_range": float,  # predicted high - low for next bar
        "last_close": float,
    }
    """
    if not _model_loaded or _predictor is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    logger.info(f"Running Kronos inference on {len(ohlcv_data)} tickers...")
    start = time.time()

    results = {}
    errors = 0

    # Process tickers individually for robustness
    # (predict_batch requires consistent shapes; individual is safer for V1)
    for symbol, df in ohlcv_data.items():
        try:
            # Prepare input DataFrame (Kronos expects: open, high, low, close)
            input_df = df[["open", "high", "low", "close"]].copy()
            if "volume" in df.columns:
                input_df["volume"] = df["volume"]

            # Timestamps
            x_timestamp = df["timestamp"]

            # Future timestamps: extend by forecast_steps bars
            last_ts = df["timestamp"].iloc[-1]
            freq_delta = timedelta(minutes=15)
            y_timestamp = pd.Series([
                last_ts + freq_delta * (i + 1) for i in range(forecast_steps)
            ])

            # Run prediction
            forecast_df = _predictor.predict(
                df=input_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
            )

            # Extract the first forecast bar's range
            if forecast_df is not None and len(forecast_df) > 0:
                forecast_high = float(forecast_df["high"].iloc[0])
                forecast_low = float(forecast_df["low"].iloc[0])
                forecast_range = forecast_high - forecast_low

                results[symbol] = {
                    "forecast_df": forecast_df,
                    "forecast_range": forecast_range,
                    "forecast_high": forecast_high,
                    "forecast_low": forecast_low,
                    "last_close": float(df["close"].iloc[-1]),
                }
            else:
                logger.debug(f"  {symbol}: empty forecast, skipping")

        except Exception as e:
            errors += 1
            logger.debug(f"  {symbol}: inference error: {e}")
            continue

    elapsed = time.time() - start
    logger.info(
        f"Kronos inference complete: {len(results)} results, "
        f"{errors} errors, {elapsed:.1f}s elapsed "
        f"({elapsed / max(len(ohlcv_data), 1):.2f}s/ticker avg)"
    )
    return results


def detect_anomalies(
    inference_results: dict[str, dict],
    ohlcv_data: dict[str, pd.DataFrame],
    state: dict,
    sigma_threshold: float = 2.0,
    dedup_sigma_delta: float = 0.5,
) -> list[dict]:
    """
    Compare Kronos forecasts against rolling baselines to detect anomalies.

    Uses exponential moving average (α=0.05) for baseline stability.
    Applies deduplication rules against active alerts in state.

    Returns list of anomaly dicts sorted by sigma (descending):
    [
        {
            "symbol": str,
            "sigma": float,
            "forecast_range": float,
            "forecast_high": float,
            "forecast_low": float,
            "last_close": float,
            "baseline_mean": float,
            "baseline_std": float,
            "is_escalating": bool,
        },
        ...
    ]
    """
    anomalies = []
    baselines = state.get("rolling_baselines", {})
    active_alerts = state.get("active_alerts", {})
    alpha = 0.05  # EMA smoothing factor

    for symbol, result in inference_results.items():
        df = ohlcv_data.get(symbol)
        if df is None or len(df) < 50:
            continue

        # Compute historical ranges from actual data
        historical_ranges = (df["high"] - df["low"]).values
        window_mean = float(np.mean(historical_ranges[-50:]))
        window_std = float(np.std(historical_ranges[-50:]))

        if window_std < 1e-8:
            continue  # flat stock, no meaningful volatility

        # Blend with persisted baseline (EMA)
        if symbol in baselines:
            stored = baselines[symbol]
            baseline_mean = (1 - alpha) * stored["mean"] + alpha * window_mean
            baseline_std = (1 - alpha) * stored["std"] + alpha * window_std
        else:
            baseline_mean = window_mean
            baseline_std = window_std

        # Update baseline in state
        baselines[symbol] = {
            "mean": round(baseline_mean, 6),
            "std": round(baseline_std, 6),
            "updated": datetime.utcnow().isoformat(),
        }

        # Compute sigma
        forecast_range = result["forecast_range"]
        sigma = (forecast_range - baseline_mean) / baseline_std

        if sigma < sigma_threshold:
            continue

        # Deduplication check
        is_escalating = False
        if symbol in active_alerts:
            prev_sigma = active_alerts[symbol].get("last_sigma", 0)
            if sigma - prev_sigma < dedup_sigma_delta:
                logger.debug(
                    f"  {symbol}: suppressed (σ={sigma:.2f}, "
                    f"prev={prev_sigma:.2f}, delta < {dedup_sigma_delta})"
                )
                # Still update the active alert tracking
                active_alerts[symbol]["last_sigma"] = round(sigma, 2)
                continue
            is_escalating = True

        anomalies.append({
            "symbol": symbol,
            "sigma": round(sigma, 2),
            "forecast_range": round(forecast_range, 4),
            "forecast_high": round(result["forecast_high"], 2),
            "forecast_low": round(result["forecast_low"], 2),
            "last_close": round(result["last_close"], 2),
            "baseline_mean": round(baseline_mean, 4),
            "baseline_std": round(baseline_std, 4),
            "is_escalating": is_escalating,
        })

        # Update active alerts
        active_alerts[symbol] = {
            "first_flagged": active_alerts.get(symbol, {}).get(
                "first_flagged", datetime.utcnow().isoformat()
            ),
            "last_sigma": round(sigma, 2),
            "alert_count": active_alerts.get(symbol, {}).get("alert_count", 0) + 1,
        }

    # Purge stale active alerts (older than 2 hours)
    cutoff = (datetime.utcnow() - timedelta(hours=2)).isoformat()
    stale = [s for s, a in active_alerts.items() if a.get("first_flagged", "") < cutoff]
    for s in stale:
        del active_alerts[s]

    # Sort by sigma descending
    anomalies.sort(key=lambda a: a["sigma"], reverse=True)

    # Write back to state
    state["rolling_baselines"] = baselines
    state["active_alerts"] = active_alerts

    logger.info(
        f"Anomaly detection: {len(anomalies)} anomalies "
        f"(threshold={sigma_threshold}σ, {len(active_alerts)} active)"
    )

    return anomalies
