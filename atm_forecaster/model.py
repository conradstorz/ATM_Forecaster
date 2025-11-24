
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestRegressor

from .config import AppConfig, load_config
from .data_loader import load_transactions
from .features import FEATURE_COLUMNS, add_calendar_features


def _safe_model_path(config: AppConfig, atm_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", atm_id)
    return config.model_dir / f"model_{safe}.pkl"


def train_models(
    config: Optional[AppConfig] = None,
    min_points: int = 30,
) -> Dict[str, Path]:
    """Train a separate model for each ATM.

    The model is a simple :class:`RandomForestRegressor` that maps
    calendar + holiday/event features -> daily amount.

    Parameters
    ----------
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.
    min_points:
        Minimum number of data points per ATM required to train a model.

    Returns
    -------
    dict
        Mapping from ``atm_id`` to the path of the saved model.
    """
    if config is None:
        config = load_config()

    config.model_dir.mkdir(parents=True, exist_ok=True)
    df = load_transactions(config)
    df = add_calendar_features(df, config)

    models: Dict[str, Path] = {}
    for atm_id, group in df.groupby("atm_id"):
        if len(group) < min_points:
            logger.warning(
                f"Skipping ATM {atm_id!r}: only {len(group)} points (< {min_points})."
            )
            continue

        X = group[FEATURE_COLUMNS].values
        y = group["amount"].values.astype(float)

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)

        path = _safe_model_path(config, atm_id)
        joblib.dump(model, path)
        logger.info(f"Trained model for ATM {atm_id!r} with {len(group)} rows -> {path}")
        models[atm_id] = path

    return models


def _future_dates(
    start_date: datetime,
    horizons: Iterable[int],
) -> List[datetime]:
    dates: List[datetime] = []
    for h in sorted(set(horizons)):
        for d in range(1, h + 1):
            dates.append(start_date + timedelta(days=d))
    # Deduplicate while preserving order
    seen = set()
    unique_dates: List[datetime] = []
    for d in dates:
        if d not in seen:
            seen.add(d)
            unique_dates.append(d)
    return unique_dates


def forecast_for_atm(
    atm_id: str,
    config: Optional[AppConfig] = None,
    as_of_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate forecasts for a single ATM for all configured horizons.

    Parameters
    ----------
    atm_id:
        ATM identifier to forecast.
    config:
        Optional :class:`AppConfig`. If omitted, default config is used.
    as_of_date:
        Last date with known actual data. If omitted, uses the maximum
        date present in the transaction data for this ATM.

    Returns
    -------
    pandas.DataFrame
        Columns: ``atm_id``, ``date``, ``horizon_days``, ``prediction``.
    """
    if config is None:
        config = load_config()

    model_path = _safe_model_path(config, atm_id)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model for ATM {atm_id!r} not found at {model_path}. "
            "Run 'train' first."
        )
    model: RandomForestRegressor = joblib.load(model_path)

    df = load_transactions(config)
    df_atm = df[df["atm_id"] == atm_id].copy()
    if df_atm.empty:
        raise ValueError(f"No data found for ATM {atm_id!r}.")

    if as_of_date is None:
        as_of = df_atm["date"].max()
    else:
        as_of = pd.to_datetime(as_of_date)

    future_dates = _future_dates(as_of, config.horizons)
    future_df = pd.DataFrame({
        "atm_id": atm_id,
        "Location": df_atm["Location"].iloc[-1],
        "date": future_dates,
    })
    future_df = add_calendar_features(future_df, config)
    X_future = future_df[FEATURE_COLUMNS].values
    preds = model.predict(X_future)

    # Build a tidy output with horizon days
    records = []
    for d, y_hat in zip(future_df["date"], preds):
        delta_days = (d - as_of).days
        if delta_days in config.horizons:
            records.append({
                "atm_id": atm_id,
                "date": d.normalize(),
                "horizon_days": int(delta_days),
                "prediction": float(y_hat),
            })
    result = pd.DataFrame.from_records(records)
    return result.sort_values(["horizon_days", "date"]).reset_index(drop=True)


def forecast_all_atms(
    config: Optional[AppConfig] = None,
    as_of_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate forecasts for all ATMs with trained models."""
    if config is None:
        config = load_config()

    df = load_transactions(config)
    atm_ids = sorted(df["atm_id"].unique().tolist())

    all_predictions: List[pd.DataFrame] = []
    for atm_id in atm_ids:
        model_path = _safe_model_path(config, atm_id)
        if not model_path.exists():
            continue
        try:
            preds = forecast_for_atm(atm_id, config=config, as_of_date=as_of_date)
            all_predictions.append(preds)
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Failed to forecast for ATM {atm_id!r}: {exc}")

    if not all_predictions:
        return pd.DataFrame(columns=["atm_id", "date", "horizon_days", "prediction"])
    return pd.concat(all_predictions, ignore_index=True)
