"""Unit tests for data validation logic in prodml.data."""

from pathlib import Path
import pandas as pd
import pytest

from prodml.data import validate_raw_data, REQUIRED_RAW_COLUMNS


@pytest.fixture
def sample_valid_parquet(tmp_path: Path) -> Path:
    """Create a minimal valid raw taxi dataset."""
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2024-01-01 10:00:00"] * 1050,
            "lpep_dropoff_datetime": ["2024-01-01 10:15:00"] * 1050,
            "PULocationID": [100] * 1050,
            "DOLocationID": [200] * 1050,
            "trip_distance": [2.5] * 1050,
        }
    )
    file_path = tmp_path / "valid_taxi.parquet"
    df.to_parquet(file_path)
    return file_path


def test_validate_raw_data_valid(sample_valid_parquet: Path) -> None:
    """Test that valid dataset passes schema and row constraints."""
    result = validate_raw_data(data_path=sample_valid_parquet, min_rows=1000)
    assert result["status"] == "valid"
    assert result["rows"] == 1050
    assert result["required_columns_present"] is True


def test_validate_raw_data_too_few_rows(tmp_path: Path) -> None:
    """Test that dataset with insufficient rows raises ValueError."""
    df = pd.DataFrame({col: [1] * 10 for col in REQUIRED_RAW_COLUMNS})
    file_path = tmp_path / "small_taxi.parquet"
    df.to_parquet(file_path)

    with pytest.raises(ValueError, match="at least 100"):
        validate_raw_data(data_path=file_path, min_rows=100)


def test_validate_raw_data_missing_column(tmp_path: Path) -> None:
    """Test that dataset missing a required schema column raises ValueError."""
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2024-01-01 10:00:00"] * 100,
            "lpep_dropoff_datetime": ["2024-01-01 10:15:00"] * 100,
            "PULocationID": [100] * 100,
            # DOLocationID missing
            "trip_distance": [2.5] * 100,
        }
    )
    file_path = tmp_path / "missing_col.parquet"
    df.to_parquet(file_path)

    with pytest.raises(ValueError, match="missing required schema columns"):
        validate_raw_data(data_path=file_path, min_rows=50)


def test_validate_raw_data_negative_distance(tmp_path: Path) -> None:
    """Test that negative trip distances trigger validation error."""
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2024-01-01 10:00:00"] * 100,
            "lpep_dropoff_datetime": ["2024-01-01 10:15:00"] * 100,
            "PULocationID": [100] * 100,
            "DOLocationID": [200] * 100,
            "trip_distance": [-1.0] + [2.0] * 99,
        }
    )
    file_path = tmp_path / "negative_distance.parquet"
    df.to_parquet(file_path)

    with pytest.raises(ValueError, match="negative trip distances detected"):
        validate_raw_data(data_path=file_path, min_rows=50)
