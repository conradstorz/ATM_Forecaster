
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class AppConfig(BaseModel):
    """Application configuration for the ATM forecaster."""

    data_path: Path = Path("data/atm_daily_normalized.csv")
    holidays_path: Path = Path("data/holidays.csv")
    events_path: Path = Path("data/special_events.csv")
    model_dir: Path = Path("models")
    history_path: Path = Path("memory/predictions_history.jsonl")
    metrics_path: Path = Path("memory/metrics.json")
    horizons: List[int] = [7, 14, 21, 28]

    model_config = ConfigDict(arbitrary_types_allowed=True)


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a JSON file if provided, else defaults.

    Parameters
    ----------
    path:
        Optional path to a JSON file containing an :class:`AppConfig` dump.

    Returns
    -------
    AppConfig
        Loaded configuration.
    """
    if path is not None and path.exists():
        return AppConfig.model_validate_json(path.read_text(encoding="utf-8"))
    return AppConfig()
