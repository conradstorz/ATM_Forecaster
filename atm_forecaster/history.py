
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from .config import AppConfig, load_config
from .data_loader import load_transactions


def append_predictions_to_history(
    preds: pd.DataFrame,
    config: Optional[AppConfig] = None,
    run_ts: Optional[datetime] = None,
) -> None:
    """Append forecast records to the JSONL history file.

    Each record will contain:
    - ``atm_id``
    - ``date``
    - ``horizon_days``
    - ``prediction``
    - ``run_ts``
    - ``actual_amount`` (initially null)
    - ``error_abs`` (initially null)
    - ``error_pct`` (initially null)
    """
    if config is None:
        config = load_config()

    history_path: Path = config.history_path
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if run_ts is None:
        run_ts = datetime.utcnow()

    with history_path.open("a", encoding="utf-8") as f:
        for _, row in preds.iterrows():
            rec: Dict[str, Any] = {
                "atm_id": row["atm_id"],
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "horizon_days": int(row["horizon_days"]),
                "prediction": float(row["prediction"]),
                "run_ts": run_ts.isoformat(),
                "actual_amount": None,
                "error_abs": None,
                "error_pct": None,
            }
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Appended {len(preds)} prediction rows to {history_path}.")


def evaluate_history(
    config: Optional[AppConfig] = None,
) -> None:
    """Evaluate all unevaluated predictions against actuals and update metrics."""
    if config is None:
        config = load_config()

    history_path: Path = config.history_path
    if not history_path.exists():
        logger.warning("No history file found; nothing to evaluate.")
        return

    rows = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        logger.warning("Prediction history is empty; nothing to evaluate.")
        return

    hist_df = pd.DataFrame(rows)
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    tx = load_transactions(config)
    merged = hist_df.merge(
        tx[["atm_id", "date", "amount"]],
        how="left",
        on=["atm_id", "date"],
        suffixes=("", "_actual"),
    )
    has_actual = merged["amount"].notna()
    merged.loc[has_actual, "actual_amount"] = merged.loc[has_actual, "amount"]
    merged.loc[has_actual, "error_abs"] = (
        merged.loc[has_actual, "prediction"] - merged.loc[has_actual, "actual_amount"]
    ).abs()
    merged.loc[has_actual, "error_pct"] = (
        merged.loc[has_actual, "error_abs"] / merged.loc[has_actual, "actual_amount"].clip(lower=1e-6)
    ) * 100.0

    # Save updated history
    with history_path.open("w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            record = {k: (v if not isinstance(v, pd.Timestamp) else v.strftime("%Y-%m-%d"))
                      for k, v in row.to_dict().items()}
            f.write(json.dumps(record) + "\n")

    # Aggregate metrics
    evaluated = merged[merged["actual_amount"].notna()]
    if evaluated.empty:
        logger.info("No predictions have actuals yet; metrics not updated.")
        return

    metrics = {
        "updated_at": datetime.utcnow().isoformat(),
        "count": int(len(evaluated)),
        "mae": float(evaluated["error_abs"].mean()),
        "mape": float(evaluated["error_pct"].mean()),
        "rmse": float((evaluated["error_abs"] ** 2).mean() ** 0.5),
    }
    config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info(f"Updated metrics -> {config.metrics_path}: {metrics}")
