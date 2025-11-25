import pytest
from typer.testing import CliRunner
from forecaster.cli import app

runner = CliRunner()


def test_train_command(monkeypatch):
    monkeypatch.setattr("forecaster.cli._setup_logging", lambda: None)
    monkeypatch.setattr("forecaster.cli.load_config", lambda path=None: object())
    monkeypatch.setattr("forecaster.cli.train_models", lambda config, min_points: [1, 2, 3])
    result = runner.invoke(app, ["train"])
    assert result.exit_code == 0
    assert "Trained 3 ATM models." in result.output


def test_forecast_command(monkeypatch):
    monkeypatch.setattr("forecaster.cli._setup_logging", lambda: None)
    monkeypatch.setattr("forecaster.cli.load_config", lambda path=None: object())
    import pandas as pd
    class DummyDF:
        empty = False
        def to_csv(self, output, index): pass
        def __len__(self): return 5
        def pivot(self, index, columns, values):
            # Return a DataFrame with the expected columns
            data = {
                "atm_id": ["ATM1", "ATM2"],
                "prediction_7": [1000, 2000],
                "prediction_14": [1500, 2500],
                "prediction_21": [1800, 2800],
                "prediction_28": [2100, 3100],
            }
            return pd.DataFrame(data)
    monkeypatch.setattr("forecaster.cli.forecast_all_atms", lambda config, as_of_date=None: DummyDF())
    monkeypatch.setattr("forecaster.cli.append_predictions_to_history", lambda preds, config: None)
    result = runner.invoke(app, ["forecast"])
    assert result.exit_code == 0
    assert "Wrote 2 ATM forecasts" in result.output
    assert "Appended predictions to history." in result.output


def test_evaluate_command(monkeypatch):
    monkeypatch.setattr("forecaster.cli._setup_logging", lambda: None)
    monkeypatch.setattr("forecaster.cli.load_config", lambda path=None: object())
    monkeypatch.setattr("forecaster.cli.evaluate_history", lambda config: None)
    result = runner.invoke(app, ["evaluate"])
    assert result.exit_code == 0
    assert "Evaluation complete" in result.output


def test_show_metrics_command(monkeypatch, tmp_path):
    class DummyConfig:
        metrics_path = tmp_path / "metrics.json"
    DummyConfig.metrics_path.write_text('{"accuracy": 0.95}', encoding="utf-8")
    monkeypatch.setattr("forecaster.cli.load_config", lambda path=None: DummyConfig())
    result = runner.invoke(app, ["show-metrics"])
    assert result.exit_code == 0
    assert '"accuracy": 0.95' in result.output


def test_show_metrics_no_file(monkeypatch):
    class DummyConfig:
        metrics_path = type("Path", (), {"exists": lambda self: False})()
    monkeypatch.setattr("forecaster.cli.load_config", lambda path=None: DummyConfig())
    result = runner.invoke(app, ["show-metrics"])
    assert result.exit_code == 0
    assert "No metrics file found yet" in result.output
