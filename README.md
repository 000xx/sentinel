# SENTINEL V1 — The 15-Minute Anomaly Tracker

A zero-cost market anomaly detection pipeline that runs on GitHub Actions every 15 minutes.

## Architecture

```
Alpaca (OHLCV) → Kronos-mini (volatility forecast) → Anomaly Detection (σ threshold)
                                                            │
                                              ┌─────────────┼─────────────┐
                                              ▼             ▼             ▼
                                          Finnhub        GDELT       Narrator
                                        (company news)  (global news)  (Groq LLM)
                                              │             │             │
                                              └──────┬──────┘             │
                                                     ▼                    ▼
                                              News Merger          3-Bullet Summary
                                                     │                    │
                                                     └────────┬──────────┘
                                                              ▼
                                                    ┌─────────────────┐
                                                    │  site/alerts.json│ ← GitHub Pages
                                                    │  Discord Webhook │ ← optional
                                                    └─────────────────┘
```

## Ticker Universe

506 tickers across 5 tiers:

| Tier | Count | Frequency | Description |
|------|-------|-----------|-------------|
| 1 — Bellwethers | 103 | 15 min | S&P 100 core + key large-caps |
| 2 — High Beta | 107 | 15 min | Meme stocks, growth, crypto-adjacent |
| 3 — Sector ETFs | 44 | 15 min | Sector SPDRs, thematic ETFs |
| 4 — Macro | 27 | 15 min | Bonds, commodities, currencies, VIX |
| 5 — Small-Cap | 225 | 60 min | High-volume small/mid-caps |

## Setup

### 1. Fork this repo (must be public for free Actions minutes)

### 2. Add GitHub Secrets

Go to Settings → Secrets and Variables → Actions:

| Secret | Service | Required |
|--------|---------|----------|
| `ALPACA_API_KEY` | [Alpaca](https://alpaca.markets) (free) | Yes |
| `ALPACA_SECRET_KEY` | Alpaca | Yes |
| `FINNHUB_API_KEY` | [Finnhub](https://finnhub.io) (free) | Yes |
| `GROQ_API_KEY` | [Groq](https://groq.com) (free) | Yes |
| `DISCORD_WEBHOOK_URL` | Discord server webhook | Optional |

GDELT requires no API key.

### 3. Enable GitHub Pages

Go to Settings → Pages → Source: "Deploy from a branch" → Branch: `main` → Folder: `/site`

Your dashboard will be live at `https://<username>.github.io/<repo>/`

### 4. Enable GitHub Actions

The workflow runs automatically via cron. Trigger a manual run from the Actions tab to test.

## Project Structure

```
sentinel/
├── .github/workflows/sentinel.yml   # 15-min cron workflow
├── config/ticker_universe.json      # 506 tickers, tiers, GDELT queries
├── agents/
│   ├── quant.py                     # Kronos model + anomaly detection
│   └── narrator.py                  # Groq/Gemini LLM synthesis
├── ingest/
│   ├── alpaca.py                    # Market data fetching
│   ├── finnhub.py                   # Finnhub news client
│   ├── gdelt.py                     # GDELT DOC API client
│   └── merger.py                    # Dual-source waterfall + dedup
├── delivery/
│   └── output.py                    # JSON file + Discord webhook
├── site/
│   ├── index.html                   # Dashboard (GitHub Pages)
│   └── alerts.json                  # Accumulated alerts (never deleted)
├── state.json                       # Rolling baselines + dedup state
├── main.py                          # Pipeline orchestrator
└── requirements.txt
```

## Configuration

Edit these in `main.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SIGMA_THRESHOLD` | 2.0 | Min σ to flag anomaly. Lower = more alerts. |
| `MAX_NARRATIONS_PER_CYCLE` | 10 | Cap on LLM calls per run (Groq free tier) |
| `LOOKBACK_BARS` | 200 | Bars of history per ticker (~50 hours) |
| `FORECAST_STEPS` | 4 | Bars to forecast forward (1 hour) |

## How It Works

Every 15 minutes:

1. **Ingest**: Fetch 15-min OHLCV candles from Alpaca for all active tickers
2. **Forecast**: Kronos-mini predicts next 4 bars of price action
3. **Detect**: Compare forecasted volatility against rolling baseline (EMA α=0.05)
4. **Deduplicate**: Suppress re-alerts unless σ increases by ≥ 0.5
5. **News**: For anomalies — Finnhub (equities) + GDELT (ETFs/macro/fallback)
6. **Narrate**: Groq LLM synthesizes headlines into 3-bullet catalyst summary
7. **Deliver**: Append to `site/alerts.json` + push Discord embed
8. **Persist**: Commit updated state back to repo

## License

MIT
