"""
State Manager

Handles persistent state across ephemeral GitHub Actions runs.
State is stored in state.json and committed back to the repo
at the end of each cycle.

State contents:
- last_run: ISO timestamp of last successful run
- active_alerts: dict of currently-active anomaly tickers (for dedup)
- rolling_baselines: dict of per-ticker mean/std for sigma calculation
- run_count: total number of successful runs
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("sentinel.state")

STATE_FILE = Path("state.json")

DEFAULT_STATE = {
    "last_run": None,
    "active_alerts": {},
    "rolling_baselines": {},
    "run_count": 0,
}


def load_state() -> dict:
    """Load state from file, or return default if missing/corrupt."""
    if not STATE_FILE.exists():
        logger.info("No state file found, starting fresh")
        return DEFAULT_STATE.copy()

    try:
        with open(STATE_FILE) as f:
            state = json.load(f)
        logger.info(
            f"State loaded: {len(state.get('active_alerts', {}))} active alerts, "
            f"{len(state.get('rolling_baselines', {}))} baselines, "
            f"run #{state.get('run_count', 0)}"
        )
        return state

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"State file corrupt, starting fresh: {e}")
        return DEFAULT_STATE.copy()


def save_state(state: dict) -> None:
    """Save state to file."""
    state["last_run"] = datetime.utcnow().isoformat() + "Z"
    state["run_count"] = state.get("run_count", 0) + 1

    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info(f"State saved (run #{state['run_count']})")
    except IOError as e:
        logger.error(f"Failed to save state: {e}")
