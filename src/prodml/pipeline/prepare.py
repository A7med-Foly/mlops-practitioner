"""Pipeline stage 1: Data Preparation.

Loads raw taxi trip records, filters anomalies/outliers, splits deterministically
into train/validation/test partitions, and serializes to data/processed/.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from prodml.data import clean_data, split_data

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.pipeline.prepare")


def run_prepare(
    input_path: str | Path = "data/raw/green_tripdata.parquet",
    output_dir: str | Path = "data/processed",
    params_path: str | Path = "params.yaml",
) -> None:
    """Execute data cleaning and partitioning according to params.yaml."""
    logger.info("Loading parameters from %s", params_path)
    with open(params_path, encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f).get("prepare", {})

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_file}")

    logger.info("Loading raw dataset from %s", input_file)
    raw_df = pd.read_parquet(input_file)
    logger.info("Raw dataset shape: %s", raw_df.shape)

    logger.info("Cleaning data with params: %s", params)
    target_col = params.get("target_column", "trip_duration")
    cleaned_df = clean_data(
        raw_df,
        min_duration=float(params.get("min_duration", 0.0)),
        max_duration_quantile=float(params.get("max_duration_quantile", 0.995)),
        min_distance=float(params.get("min_distance", 0.0)),
        pickup_col=params.get("pickup_column", "lpep_pickup_datetime"),
        dropoff_col=params.get("dropoff_column", "lpep_dropoff_datetime"),
        target_col=target_col,
    )
    logger.info("Cleaned dataset shape: %s", cleaned_df.shape)

    logger.info("Splitting dataset into train, val, test partitions...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        cleaned_df,
        target_col=target_col,
        test_size=float(params.get("test_size", 0.2)),
        val_size=float(params.get("val_size", 0.25)),
        random_state=int(params.get("random_state", 42)),
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_file = out_path / "train.parquet"
    val_file = out_path / "val.parquet"
    test_file = out_path / "test.parquet"

    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)

    logger.info("Wrote %s rows to %s", len(train_df), train_file)
    logger.info("Wrote %s rows to %s", len(val_df), val_file)
    logger.info("Wrote %s rows to %s", len(test_df), test_file)


if __name__ == "__main__":
    run_prepare()
