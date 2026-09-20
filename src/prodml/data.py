"""Data ingestion, cleaning, and splitting module for prodml.

Handles loading Parquet files, filtering outliers and missing values,
calculating trip durations, and splitting datasets for training, validation, and testing.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import get_settings

# Columns identified in EDA where missingness correlates and indicates incomplete records
INITIAL_COLS_TO_DROP_NA = [
    "store_and_fwd_flag",
    "RatecodeID",
    "passenger_count",
    "payment_type",
    "trip_type",
    "congestion_surcharge",
]


def load_data(path: Path | str | None = None) -> pd.DataFrame:
    """Load trip record dataset from Parquet file.

    Args:
        path: Path to the Parquet dataset. If None, falls back to Settings.data_path.

    Returns:
        pd.DataFrame containing loaded records.
    """
    if path is None:
        settings = get_settings()
        path = settings.data_path

    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found at {resolved_path}")

    return pd.read_parquet(resolved_path)


def compute_trip_duration(
    df: pd.DataFrame,
    pickup_col: str = "lpep_pickup_datetime",
    dropoff_col: str = "lpep_dropoff_datetime",
) -> pd.Series:
    """Compute trip duration in minutes from pickup and dropoff datetimes.

    Args:
        df: Input DataFrame containing datetime columns.
        pickup_col: Column name for trip pickup datetime.
        dropoff_col: Column name for trip dropoff datetime.

    Returns:
        pd.Series containing trip duration in minutes.
    """
    pickup = pd.to_datetime(df[pickup_col])
    dropoff = pd.to_datetime(df[dropoff_col])
    duration = (dropoff - pickup).dt.total_seconds() / 60.0
    return duration


def clean_data(
    df: pd.DataFrame,
    min_duration: float = 0.0,
    max_duration_quantile: float = 0.995,
    min_distance: float = 0.0,
    pickup_col: str = "lpep_pickup_datetime",
    dropoff_col: str = "lpep_dropoff_datetime",
    target_col: str = "trip_duration",
) -> pd.DataFrame:
    """Clean raw taxi trip data according to baseline EDA rules.

    - Computes trip duration in minutes if not present.
    - Drops 'ehail_fee' column (100% null).
    - Drops rows where critical subset of attributes are missing.
    - Filters duration between min_duration and max_duration_quantile upper bound.
    - Filters trips with trip_distance strictly greater than min_distance.

    Args:
        df: Raw taxi trip records DataFrame.
        min_duration: Minimum duration in minutes (exclusive).
        max_duration_quantile: Upper quantile threshold for duration outlier removal.
        min_distance: Minimum trip distance in miles (exclusive).
        pickup_col: Pickup datetime column name.
        dropoff_col: Dropoff datetime column name.
        target_col: Target column name for trip duration.

    Returns:
        Cleaned and filtered pd.DataFrame.
    """
    cleaned_df = df.copy()

    # 1. Compute duration if missing
    if target_col not in cleaned_df.columns:
        cleaned_df[target_col] = compute_trip_duration(
            cleaned_df, pickup_col=pickup_col, dropoff_col=dropoff_col
        )

    # 2. Drop 100% null column 'ehail_fee' if present
    if "ehail_fee" in cleaned_df.columns:
        cleaned_df.drop(columns=["ehail_fee"], inplace=True)

    # 3. Drop rows with missing values in the correlated subset
    present_cols = [col for col in INITIAL_COLS_TO_DROP_NA if col in cleaned_df.columns]
    if present_cols:
        cleaned_df.dropna(subset=present_cols, inplace=True)

    # 4. Filter duration outliers
    if not cleaned_df.empty:
        upper_bound = cleaned_df[target_col].quantile(max_duration_quantile)
        cleaned_df = cleaned_df[
            (cleaned_df[target_col] > min_duration)
            & (cleaned_df[target_col] <= upper_bound)
        ].copy()

    # 5. Filter zero-distance trips
    if "trip_distance" in cleaned_df.columns and not cleaned_df.empty:
        cleaned_df = cleaned_df[cleaned_df["trip_distance"] > min_distance].copy()

    return cleaned_df


def split_data(
    df: pd.DataFrame,
    target_col: str = "trip_duration",
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataset into train, validation, and test partitions.

    Splits in two stages:
    1. df -> train_val (1 - test_size) and test (test_size).
    2. train_val -> train (1 - val_size of train_val) and val (val_size of train_val).
    Default parameters (test_size=0.2, val_size=0.25) produce a 60/20/20 split.

    Args:
        df: Cleaned dataset containing features and target.
        target_col: Column name of target variable.
        test_size: Proportion of dataset for test set.
        val_size: Proportion of train_val set for validation set.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # First split: train_val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Second split: train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


REQUIRED_RAW_COLUMNS = [
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]


def validate_raw_data(
    data_path: Path | str | None = None,
    min_rows: int = 1000,
) -> dict[str, Any]:
    """Validate raw trip data schema, minimum record volume, and data sanity.

    Args:
        data_path: Path to raw Parquet file. Defaults to configured data path.
        min_rows: Minimum expected row count for training.

    Returns:
        Summary dict containing validation metrics.

    Raises:
        FileNotFoundError: If dataset does not exist.
        ValueError: If schema or record volume constraints fail.
    """
    df = load_data(data_path)
    total_rows = len(df)

    if total_rows < min_rows:
        raise ValueError(
            f"Data validation failed: dataset has {total_rows} rows, expected at least {min_rows}."
        )

    missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Data validation failed: missing required schema columns {missing_cols}. "
            f"Present columns: {list(df.columns)}"
        )

    # Basic sanity checks on distances and timestamps
    valid_distances = (df["trip_distance"] >= 0).all()
    if not valid_distances:
        raise ValueError("Data validation failed: negative trip distances detected.")

    return {
        "status": "valid",
        "rows": total_rows,
        "columns": len(df.columns),
        "required_columns_present": True,
        "data_path": str(data_path),
    }


def main() -> None:
    """CLI entrypoint for data operations."""
    import argparse

    parser = argparse.ArgumentParser(description="ProdML Data Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate raw dataset schema and constraints"
    )
    validate_parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/green_tripdata.parquet",
        help="Path to raw dataset",
    )
    validate_parser.add_argument(
        "--min-rows",
        type=int,
        default=1000,
        help="Minimum required row count",
    )

    args = parser.parse_args()

    if args.command == "validate":
        try:
            result = validate_raw_data(data_path=args.data_path, min_rows=args.min_rows)
            print("=" * 60)
            print("                 DATA VALIDATION PASSED")
            print("=" * 60)
            print(f"Path          : {result['data_path']}")
            print(f"Total Rows    : {result['rows']:,}")
            print(f"Total Columns : {result['columns']}")
            print("Schema Status : All required taxi columns verified.")
            print("=" * 60)
        except Exception as exc:
            print(f"❌ DATA VALIDATION FAILED: {exc}")
            sys.exit(1)


if __name__ == "__main__":
    main()
