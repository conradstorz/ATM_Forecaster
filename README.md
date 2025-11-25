
# ATM Forecaster

Daily cash movement forecasting for your ATM route, with calendar/holiday features
and self-evaluating prediction history.

## Layout

- `atm_forecaster/` – Python package (config, data loading, features, models, CLI).
- `data/atm_daily_normalized.csv` – normalized daily data (atm_id, Location, date, amount).
- `data/holidays.csv` – simple list of holidays (you can edit/extend).
- `data/special_events.csv` – special events that may affect volume (e.g. Derby). 
- `models/` – trained models, one per ATM.
- `memory/` – prediction history (`predictions_history.jsonl`) and metrics (`metrics.json`).
- `outputs/` – CSV forecast outputs from the CLI.
- `logs/` – Loguru logs from CLI operations.

## Basic usage with uv

From the project root (where `pyproject.toml` lives):

```bash
# 1) Create/refresh the environment, install dependencies AND this package
uv sync

# 2) Train models for all ATMs with at least 30 days of data
uv run -- atm-forecaster train --min-points 30

# 3) Forecast 7/14/21/28 days ahead for all ATMs
uv run -- atm-forecaster forecast

# 4) Forecast only for a single ATM
uv run -- atm-forecaster forecast --atm-id NW35983

# 5) After new actuals exist for forecasted dates, evaluate accuracy
uv run -- atm-forecaster evaluate

# 6) Show the current metrics JSON in a readable form
uv run -- atm-forecaster show-metrics

## Basic usage with uv


You can customize paths and horizons by writing a JSON config file containing
an `AppConfig` dump and passing `--config-path` to the CLI commands.
