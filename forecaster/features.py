
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .config import AppConfig


def _load_calendar_flags(path: Path, date_col: str = "date") -> pd.DataFrame:
    """Load a simple date-based flag file if it exists.

    The file is expected to contain at least a ``date`` column.

    Parameters
    ----------
    path:
        Path to a CSV file. If it does not exist, an empty frame is returned.
    date_col:
        Name of the date column in the file.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a normalized ``date`` column or empty if missing.
    """
    if not path.exists():
        return pd.DataFrame(columns=["date"])
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Calendar file {path} must contain column '{date_col}'.")
    df["date"] = pd.to_datetime(df[date_col])
    return df


def add_calendar_features(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> pd.DataFrame:
    """Add calendar- and event-based features to the transaction data.

    Features added:
    - ``dow``: day of week (0=Mon, 6=Sun)
    - ``is_weekend``: 1 if Saturday/Sunday, else 0
    - ``dom``: day of month
    - ``month``: month of year
    - ``year``: calendar year
    - ``weekofyear``: ISO week number
    - ``is_month_start``: 1 if first day of month
    - ``is_month_end``: 1 if last day of month
    - ``is_holiday``: 1 if date appears in holidays file
    - ``has_event``: 1 if date appears in special events file

    Parameters
    ----------
    df:
        Transactions DataFrame with a ``date`` column.
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with additional feature columns.
    """
    from .config import load_config  # local import to avoid cycles

    if config is None:
        config = load_config()

    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column.")

    date = pd.to_datetime(df["date"])
    df["dow"] = date.dt.weekday
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["dom"] = date.dt.day
    df["month"] = date.dt.month
    df["year"] = date.dt.year
    df["weekofyear"] = date.dt.isocalendar().week.astype(int)
    df["is_month_start"] = date.dt.is_month_start.astype(int)
    df["is_month_end"] = date.dt.is_month_end.astype(int)

    # Holidays
    holidays = _load_calendar_flags(config.holidays_path)
    holidays = holidays.drop_duplicates("date") if not holidays.empty else holidays
    holidays["is_holiday"] = 1
    if not holidays.empty:
        df = df.merge(
            holidays[["date", "is_holiday"]],
            how="left",
            on="date",
        )
    else:
        df["is_holiday"] = 0
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)

    # Special events
    events = _load_calendar_flags(config.events_path)
    if not events.empty:
        events = events.drop_duplicates("date")
        events["has_event"] = 1
        df = df.merge(
            events[["date", "has_event"]],
            how="left",
            on="date",
        )
    else:
        df["has_event"] = 0
    df["has_event"] = df["has_event"].fillna(0).astype(int)

    return df


FEATURE_COLUMNS = [
    "dow",
    "is_weekend",
    "dom",
    "month",
    "year",
    "weekofyear",
    "is_month_start",
    "is_month_end",
    "is_holiday",
    "has_event",
]
