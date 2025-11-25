import pytest
import pandas as pd
from forecaster.data_loader import load_transactions

class DummyConfig:
    data_path = None

def test_load_transactions_file_not_found(tmp_path, monkeypatch):
    config = DummyConfig()
    config.data_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_transactions(config)

def test_load_transactions_missing_columns(tmp_path, monkeypatch):
    # Create a CSV missing 'amount' column
    csv_path = tmp_path / "atm.csv"
    df = pd.DataFrame({
        "atm_id": ["A1"],
        "Location": ["Loc1"],
        "date": ["2025-11-25"]
    })
    df.to_csv(csv_path, index=False)
    config = DummyConfig()
    config.data_path = csv_path
    with pytest.raises(ValueError) as exc:
        load_transactions(config)
    assert "missing columns" in str(exc.value)

def test_load_transactions_success(tmp_path, monkeypatch):
    # Create a valid CSV
    csv_path = tmp_path / "atm.csv"
    df = pd.DataFrame({
        "atm_id": ["A1", "A2"],
        "Location": ["Loc1", "Loc2"],
        "date": ["2025-11-25", "2025-11-26"],
        "amount": [100, 200]
    })
    df.to_csv(csv_path, index=False)
    config = DummyConfig()
    config.data_path = csv_path
    result = load_transactions(config)
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {"atm_id", "Location", "date", "amount"}
    assert len(result) == 2
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result.loc[0, "atm_id"] == "A1"
    assert result.loc[1, "amount"] == 200
