
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from loguru import logger

from .config import AppConfig, load_config
from .data_loader import load_transactions
from .history import append_predictions_to_history, evaluate_history
from .model import forecast_all_atms, forecast_for_atm, train_models
from .normalize import normalize_funds_csv, auto_normalize_new_raw_files


app = typer.Typer(help="ATM daily cash forecaster CLI.")


def _setup_logging(log_dir: Path = Path("logs")) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    logfile = log_dir / "atm_forecaster.log"
    logger.add(logfile, rotation="1 week", retention="4 weeks")

@app.command()
def normalize(
    input: Path = typer.Argument(
        ...,
        exists=True,
        readable=True,
        help="Raw 'Funds Movement by ATM by Day' CSV from the network.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Where to write the normalized daily CSV. "
             "Defaults to config.data_path (data/atm_daily_normalized.csv).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to a JSON config file.",
    ),
) -> None:
    """
    Normalize a raw funds-movement CSV into the training dataset.

    This reads the processor's per-ATM/per-day CSV, parses dates and amounts,
    aggregates by (atm_id, Location, date), and writes a clean
    `atm_daily_normalized.csv` ready for model training.
    """
    _setup_logging()
    config: AppConfig = load_config(config_path)

    out_path = normalize_funds_csv(
        input_path=input,
        output_path=output,
        config=config,
    )
    typer.echo(f"Normalized data written to {out_path}")

@app.command("normalize-auto")
def normalize_auto(
    config_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to a JSON config file. "
             "The data directory is derived from config.data_path.parent.",
    ),
) -> None:
    """
    Find all new raw CSV files in the data folder, normalize them, append
    non-duplicate rows into atm_daily_normalized.csv, and move the raw
    files into data/archive.

    You do not need to specify file names: just drop raw CSVs into `data/`
    and run this command.
    """
    _setup_logging()
    config: AppConfig = load_config(config_path)

    data_dir = config.data_path.parent
    processed, added = auto_normalize_new_raw_files(
        data_dir=data_dir,
        config=config,
    )

    typer.echo(
        f"Processed {processed} raw file(s); "
        f"added {added} new daily row(s)."
    )


@app.command()
def train(
    config_path: Optional[Path] = typer.Option(
        None, help="Optional path to a JSON config file."
    ),
    min_points: int = typer.Option(
        30, help="Minimum data points per ATM required for training."
    ),
) -> None:
    """Train models for all ATMs with sufficient data."""
    _setup_logging()
    config: AppConfig = load_config(config_path)
    models = train_models(config=config, min_points=min_points)
    typer.echo(f"Trained {len(models)} ATM models.")


@app.command()
def forecast(
    config_path: Optional[Path] = typer.Option(
        None, help="Optional path to a JSON config file."
    ),
    atm_id: Optional[str] = typer.Option(
        None, help="If provided, only forecast for this ATM ID."
    ),
    as_of: Optional[str] = typer.Option(
        None,
        help="As-of date (YYYY-MM-DD). If omitted, uses latest date per ATM.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Optional CSV output path. If omitted, writes to outputs/forecast_<timestamp>.csv.",
    ),
    record_history: bool = typer.Option(
        True,
        help="If true, append predictions to the JSONL history file.",
    ),
) -> None:
    """Generate forecasts for all ATMs (or a single ATM)."""
    _setup_logging()
    config: AppConfig = load_config(config_path)

    as_of_date = None
    if as_of:
        as_of_date = datetime.fromisoformat(as_of)

    if atm_id is not None:
        preds = forecast_for_atm(atm_id, config=config, as_of_date=as_of_date)
    else:
        preds = forecast_all_atms(config=config, as_of_date=as_of_date)

    if preds.empty:
        typer.echo("No predictions generated. Did you train models first?")
        return

    # Pivot to one row per atm_id, columns for each horizon
    pivot = preds.pivot(index="atm_id", columns="horizon_days", values="prediction")
    pivot = pivot.rename(columns={h: f"prediction_{h}" for h in pivot.columns})
    pivot = pivot.reset_index()

    config_path_dir = Path("outputs")
    config_path_dir.mkdir(parents=True, exist_ok=True)
    if output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = config_path_dir / f"forecast_{ts}.csv"

    pivot.to_csv(output, index=False)
    typer.echo(f"Wrote {len(pivot)} ATM forecasts to {output}.")

    if record_history:
        append_predictions_to_history(preds, config=config)
        typer.echo("Appended predictions to history.")


@app.command()
def evaluate(
    config_path: Optional[Path] = typer.Option(
        None, help="Optional path to a JSON config file."
    ),
) -> None:
    """Evaluate prediction history against actuals and update metrics."""
    _setup_logging()
    config: AppConfig = load_config(config_path)
    evaluate_history(config=config)
    typer.echo("Evaluation complete. Check metrics.json for updated metrics.")


@app.command()
def show_metrics(
    config_path: Optional[Path] = typer.Option(
        None, help="Optional path to a JSON config file."
    ),
) -> None:
    """Print the most recent metrics JSON, if available."""
    config: AppConfig = load_config(config_path)
    if not config.metrics_path.exists():
        typer.echo("No metrics file found yet. Run 'evaluate' after you have actuals.")
        return
    data = json.loads(config.metrics_path.read_text(encoding="utf-8"))
    typer.echo(json.dumps(data, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()
