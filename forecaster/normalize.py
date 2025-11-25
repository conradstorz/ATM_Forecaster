from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from loguru import logger

from .config import AppConfig, load_config


def _parse_amount(val) -> float:
    """
    Parse currency strings like '$1,720.00' or '(1,234.56)' into floats.
    """
    if pd.isna(val):
        return math.nan
    if not isinstance(val, str):
        return float(val)

    s = val.strip()
    negative = False

    # Handle parentheses-style negatives: (123.45)
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1]

    # Strip currency formatting
    s = s.replace("$", "").replace(",", "").strip()
    if s in ("", ".", "-"):
        return math.nan

    try:
        num = float(s)
    except ValueError:
        logger.warning(f"Could not parse amount value {val!r}; treating as NaN.")
        return math.nan

    return -num if negative else num


def _normalize_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a raw 'Funds Movement by ATM by Day' DataFrame and return
    normalized daily rows with columns: atm_id, Location, date, amount.
    """
    required_cols = {
        "Terminal",
        "Location",
        "Settlement Date",
        "Settlement Type",
        "Amount",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")

    # Filter to transaction rows only (ignore adjustments, etc.)
    df = df[df["Settlement Type"] == "Transaction"].copy()

    # Parse amount
    df["amount"] = df["Amount"].apply(_parse_amount)

    # Parse dates (m/d/yy from your sample)
    df["date"] = pd.to_datetime(df["Settlement Date"], format="%m/%d/%y")

    # Use Terminal as atm_id
    df["atm_id"] = df["Terminal"]

    grouped = (
        df.groupby(["atm_id", "Location", "date"], as_index=False)["amount"]
        .sum()
        .sort_values(["atm_id", "date"])
        .reset_index(drop=True)
    )
    return grouped


def normalize_funds_csv(
    input_path: Path,
    output_path: Optional[Path] = None,
    config: Optional[AppConfig] = None,
) -> Path:
    """
    Normalize a raw 'Funds Movement by ATM by Day' CSV into atm_daily_normalized.csv.

    Parameters
    ----------
    input_path:
        Path to the raw CSV exported from your network processor.
    output_path:
        Destination CSV path. If omitted, uses ``config.data_path``.
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.

    Returns
    -------
    Path
        The path to the normalized CSV that was written.
    """
    if config is None:
        config = load_config()

    input_path = Path(input_path)
    if output_path is None:
        output_path = config.data_path
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Raw funds CSV not found: {input_path}")

    logger.info(f"Loading raw funds CSV from {input_path}")
    raw_df = pd.read_csv(input_path)
    grouped = _normalize_raw_frame(raw_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path, index=False)

    logger.info(
        f"Normalized {len(raw_df)} raw rows into {len(grouped)} daily rows -> {output_path}"
    )
    return output_path


def auto_normalize_new_raw_files(
    data_dir: Path,
    config: Optional[AppConfig] = None,
) -> Tuple[int, int]:
    """
    Scan a data directory for new raw CSV files, normalize them, append
    non-duplicate rows into the main normalized CSV, and move the raw
    files into data/archive.

    A "duplicate" is any row with the same (atm_id, Location, date)
    as an existing row in the normalized CSV. Existing rows win.

    Parameters
    ----------
    data_dir:
        Directory to scan for raw CSV files (usually config.data_path.parent).
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.

    Returns
    -------
    (processed_files, new_rows_added)
    """
    if config is None:
        config = load_config()

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = data_dir / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = config.data_path
    # If config.data_path is relative (e.g. "data/atm_daily_normalized.csv"),
    # leave it as-is; we assume cwd = project root.
    normalized_path = Path(normalized_path)

    # Load existing normalized data (if any)
    if normalized_path.exists():
        existing = pd.read_csv(normalized_path)
        if not existing.empty:
            existing["date"] = pd.to_datetime(existing["date"])
    else:
        existing = pd.DataFrame(columns=["atm_id", "Location", "date", "amount"])

    key_cols = ["atm_id", "Location", "date"]

    # Identify candidate raw files: any CSV in data_dir that is
    # not the normalized file, not holidays, not special_events, and
    # not already in archive/.
    protected_names = {
        normalized_path.name,
        "holidays.csv",
        "special_events.csv",
    }

    raw_files = [
        p
        for p in data_dir.glob("*.csv")
        if p.name not in protected_names
    ]

    processed_files = 0
    new_rows_added = 0

    if not raw_files:
        logger.info(f"No new raw CSV files found in {data_dir}.")
        return processed_files, new_rows_added

    # Build an index of existing keys for fast duplicate detection
    if not existing.empty:
        existing_index = pd.MultiIndex.from_frame(existing[key_cols])
    else:
        existing_index = None

    for raw_path in raw_files:
        logger.info(f"Processing raw file {raw_path}")
        raw_df = pd.read_csv(raw_path)
        grouped = _normalize_raw_frame(raw_df)
        grouped["date"] = pd.to_datetime(grouped["date"])

        if existing_index is not None and not grouped.empty:
            new_index = pd.MultiIndex.from_frame(grouped[key_cols])
            mask_new = ~new_index.isin(existing_index)
            grouped_new = grouped[mask_new].copy()
        else:
            grouped_new = grouped

        if not grouped_new.empty:
            existing = pd.concat([existing, grouped_new], ignore_index=True)
            new_rows_added += len(grouped_new)
            # Update existing_index to include the newly added keys
            existing_index = pd.MultiIndex.from_frame(existing[key_cols])
            logger.info(
                f"Added {len(grouped_new)} new daily rows from {raw_path.name}"
            )
        else:
            logger.info(f"No new daily rows from {raw_path.name} (all duplicates).")

        # Move raw file to archive
        dest = archive_dir / raw_path.name
        dest.exists() and dest.unlink()
        raw_path.rename(dest)
        logger.info(f"Moved {raw_path.name} -> {dest}")
        processed_files += 1

    # Write back normalized CSV only if we added anything
    if new_rows_added > 0:
        existing["date"] = pd.to_datetime(existing["date"])
        existing = (
            existing.sort_values(["atm_id", "date"])
            .reset_index(drop=True)
        )
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(normalized_path, index=False)
        logger.info(
            f"Normalized data updated at {normalized_path} "
            f"({len(existing)} total rows; {new_rows_added} new)."
        )
    else:
        logger.info("No new rows added; normalized CSV unchanged.")

    return processed_files, new_rows_added
