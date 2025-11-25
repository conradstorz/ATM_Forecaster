import pytest
import pandas as pd
from pathlib import Path
from forecaster.normalize import _parse_amount, _normalize_raw_frame, normalize_funds_csv

RAW_COLUMNS = ["Terminal", "Location", "Settlement Date", "Settlement Type", "Amount"]

def test_parse_amount_basic():
    assert _parse_amount("$1,234.56") == 1234.56
    assert _parse_amount("(1,234.56)") == -1234.56
    assert _parse_amount("-") != _parse_amount("0")  # Should be nan
    assert _parse_amount(None) != _parse_amount("0")  # Should be nan
    assert _parse_amount(100) == 100
    assert _parse_amount("$0.00") == 0.0

def test_normalize_raw_frame_success():
    df = pd.DataFrame([
        {"Terminal": "ATM1", "Location": "Loc1", "Settlement Date": "11/25/25", "Settlement Type": "Transaction", "Amount": "$100.00"},
        {"Terminal": "ATM1", "Location": "Loc1", "Settlement Date": "11/25/25", "Settlement Type": "Transaction", "Amount": "$50.00"},
        {"Terminal": "ATM2", "Location": "Loc2", "Settlement Date": "11/26/25", "Settlement Type": "Transaction", "Amount": "$200.00"},
        {"Terminal": "ATM2", "Location": "Loc2", "Settlement Date": "11/26/25", "Settlement Type": "Adjustment", "Amount": "$10.00"},
    ])
    result = _normalize_raw_frame(df)
    assert set(result.columns) == {"atm_id", "Location", "date", "amount"}
    assert len(result) == 2  # Only transaction rows
    assert result.loc[0, "amount"] == 150.0
    assert result.loc[1, "amount"] == 200.0

def test_normalize_raw_frame_missing_columns():
    df = pd.DataFrame({"Terminal": ["ATM1"], "Location": ["Loc1"]})
    with pytest.raises(ValueError):
        _normalize_raw_frame(df)

def test_normalize_funds_csv(tmp_path):
    raw_path = tmp_path / "raw.csv"
    out_path = tmp_path / "out.csv"
    df = pd.DataFrame([
        {"Terminal": "ATM1", "Location": "Loc1", "Settlement Date": "11/25/25", "Settlement Type": "Transaction", "Amount": "$100.00"},
        {"Terminal": "ATM2", "Location": "Loc2", "Settlement Date": "11/26/25", "Settlement Type": "Transaction", "Amount": "$200.00"},
    ])
    df.to_csv(raw_path, index=False)
    class DummyConfig:
        data_path = out_path
    result_path = normalize_funds_csv(raw_path, output_path=out_path, config=DummyConfig())
    assert result_path == out_path
    out_df = pd.read_csv(out_path)
    assert set(out_df.columns) == {"atm_id", "Location", "date", "amount"}
    assert len(out_df) == 2
    assert out_df.loc[0, "atm_id"] == "ATM1"
    assert out_df.loc[1, "amount"] == 200.0
