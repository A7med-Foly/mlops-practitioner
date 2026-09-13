"""Feature engineering and transformation module for prodml.

Handles categorical encoding (store_and_fwd_flag), column filtering,
and dictionary formatting for DictVectorizer integration.
"""

from typing import Any

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

# Explicit list of feature names used by the baseline model
FEATURE_COLUMNS = [
    "cbd_congestion_fee",
    "store_and_fwd_flag_encoded",
    "improvement_surcharge",
    "trip_distance",
    "congestion_surcharge",
    "tolls_amount",
    "fare_amount",
    "tip_amount",
    "total_amount",
]

# Columns dropped or excluded during feature engineering in baseline notebook
EXCLUDED_COLUMNS = [
    "VendorID",
    "lpep_pickup_datetime",
    "lpep_dropoff_datetime",
    "RatecodeID",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "payment_type",
    "payment_type_name",
    "trip_type",
    "pickup_hour",
    "pickup_day_of_week",
    "total_amount_difference",
    "expected_total_amount",
    "RatecodeID_name",
    "trip_type_name",
    "tip_percentage",
    "extra",
    "mta_tax",
    "ehail_fee",
    "trip_duration",
]


def encode_store_and_fwd_flag(flag: Any) -> int:
    """Encode store_and_fwd_flag into binary integer (N -> 0, Y -> 1).

    Args:
        flag: Value of store_and_fwd_flag ('Y', 'N', 1, 0, or boolean).

    Returns:
        0 or 1.
    """
    if isinstance(flag, str):
        return 1 if flag.strip().upper() == "Y" else 0
    if isinstance(flag, (int, float, bool)):
        return 1 if bool(flag) else 0
    return 0


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations to a DataFrame.

    - Encodes 'store_and_fwd_flag' to 'store_and_fwd_flag_encoded'.
    - Drops excluded columns and target variable if present.

    Args:
        df: Input DataFrame.

    Returns:
        Transformed DataFrame with engineered features.
    """
    transformed = df.copy()

    # Encode store_and_fwd_flag
    if "store_and_fwd_flag" in transformed.columns:
        transformed["store_and_fwd_flag_encoded"] = transformed[
            "store_and_fwd_flag"
        ].apply(encode_store_and_fwd_flag)

    # Drop excluded columns
    cols_to_drop = [col for col in EXCLUDED_COLUMNS if col in transformed.columns]
    if cols_to_drop:
        transformed.drop(columns=cols_to_drop, inplace=True)

    if "store_and_fwd_flag" in transformed.columns:
        transformed.drop(columns=["store_and_fwd_flag"], inplace=True)

    return transformed


def dict_to_feature_dict(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single record dictionary into the format expected by DictVectorizer.

    Args:
        record: Raw feature dictionary (from JSON payload or caller).

    Returns:
        Normalized dictionary with encoded features.
    """
    output: dict[str, Any] = {}

    # Copy known numeric fields if present
    for key in [
        "cbd_congestion_fee",
        "improvement_surcharge",
        "trip_distance",
        "congestion_surcharge",
        "tolls_amount",
        "fare_amount",
        "tip_amount",
        "total_amount",
    ]:
        if key in record:
            output[key] = float(record[key])

    # Handle store_and_fwd_flag encoding
    if "store_and_fwd_flag_encoded" in record:
        output["store_and_fwd_flag_encoded"] = int(record["store_and_fwd_flag_encoded"])
    elif "store_and_fwd_flag" in record:
        output["store_and_fwd_flag_encoded"] = encode_store_and_fwd_flag(
            record["store_and_fwd_flag"]
        )
    else:
        output["store_and_fwd_flag_encoded"] = 0

    # Handle PU_DO interaction feature
    if "PU_DO" in record:
        output["PU_DO"] = str(record["PU_DO"])
    elif "PULocationID" in record and "DOLocationID" in record:
        output["PU_DO"] = f"{record['PULocationID']}_{record['DOLocationID']}"

    return output


def prepare_features(
    data: pd.DataFrame | list[dict[str, Any]] | dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert input data (DataFrame, dict, or list of dicts) into a list of feature dictionaries.

    Args:
        data: Tabular DataFrame, single record dict, or list of dicts.

    Returns:
        List of dictionaries ready for DictVectorizer.
    """
    if isinstance(data, pd.DataFrame):
        encoded_df = encode_features(data)
        return encoded_df.to_dict(orient="records")

    if isinstance(data, dict):
        return [dict_to_feature_dict(data)]

    if isinstance(data, list):
        return [dict_to_feature_dict(item) for item in data]

    raise TypeError(f"Unsupported data type for feature preparation: {type(data)}")


def create_vectorizer() -> DictVectorizer:
    """Instantiate a new DictVectorizer."""
    return DictVectorizer(sparse=True)


def build_pipeline(model: Any) -> Pipeline:
    """Create a scikit-learn Pipeline combining DictVectorizer and model estimator.

    Args:
        model: Scikit-learn compatible estimator (e.g. RandomForestRegressor).

    Returns:
        Pipeline instance.
    """
    return Pipeline([("vectorizer", create_vectorizer()), ("model", model)])
