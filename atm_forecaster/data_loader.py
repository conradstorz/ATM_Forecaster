
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import AppConfig


def load_transactions(config: Optional[AppConfig] = None) -> pd.DataFrame:
    """Load normalized ATM daily transactions.

    The expected columns are:
    - ``atm_id``: unique ATM identifier (from Terminal field)
    - ``Location``: human-readable location name
    - ``date``: date of settlement (datetime-like or string)
    - ``amount``: cash moved / withdrawn for that ATM on that date

    Parameters
    ----------
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame with a normalized schema.
    """
    from .config import load_config  # local import to avoid cycles

    if config is None:
        config = load_config()

    path: Path = config.data_path
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)
    expected_cols = {"atm_id", "Location", "date", "amount"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Data file {path} is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], utc=False)
    df = df.sort_values(["atm_id", "date"]).reset_index(drop=True)
    return df
