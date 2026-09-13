"""Model training, evaluation, and persistence module for prodml.

Fits baseline regression models on green taxi trip records, evaluates metrics
(MAE, RMSE, R2), persists the fitted pipeline artifact, and exposes the CLI entry point.
"""

import logging
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from prodml.config import Settings, get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.features import build_pipeline, prepare_features

logger = logging.getLogger("prodml.train")


def create_model(model_type: str = "random_forest", **kwargs: Any) -> RegressorMixin:
    """Instantiate the regressor specified by model_type.

    Args:
        model_type: 'random_forest' or 'linear_regression'.
        **kwargs: Additional hyperparameters passed to model constructor.

    Returns:
        Instantiated scikit-learn regressor.
    """
    if model_type == "random_forest":
        n_estimators = kwargs.get("n_estimators", 100)
        n_jobs = kwargs.get("n_jobs", -1)
        random_state = kwargs.get("random_state", 42)
        return RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    elif model_type == "linear_regression":
        return LinearRegression()
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. Expected 'random_forest' or 'linear_regression'."
        )


def train_pipeline(
    X_train: list[dict[str, Any]],
    y_train: pd.Series | np.ndarray,
    model: RegressorMixin | None = None,
) -> Pipeline:
    """Build and fit a pipeline containing DictVectorizer and the given regressor.

    Args:
        X_train: Training feature records (list of dictionaries).
        y_train: Target values (trip duration in minutes).
        model: Regressor instance to fit. If None, defaults to RandomForestRegressor.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    if model is None:
        model = create_model("random_forest")

    pipeline = build_pipeline(model)
    logger.info("Fitting pipeline with model: %s", model.__class__.__name__)
    pipeline.fit(X_train, y_train)
    logger.info("Model fitting complete.")
    return pipeline


def evaluate_pipeline(
    pipeline: Pipeline,
    X_val: list[dict[str, Any]],
    y_val: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute evaluation metrics (MAE, RMSE, R2) on validation data.

    Args:
        pipeline: Fitted Pipeline.
        X_val: Validation feature records.
        y_val: Ground truth validation target values.

    Returns:
        Dictionary mapping metric names to computed scores.
    """
    y_pred = pipeline.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info("Validation Metrics - MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)
    return metrics


def save_artifact(artifact: Any, path: Path | str) -> None:
    """Serialize and save an artifact (e.g. Pipeline or dict) using pickle.

    Args:
        artifact: Python object to persist.
        path: Target file path.
    """
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved artifact to %s", dest_path.resolve())


def run_training(settings: Settings | None = None) -> tuple[Pipeline, dict[str, float]]:
    """Execute full training workflow: load, clean, split, fit, evaluate, and save.

    Args:
        settings: Application settings. If None, loaded from environment/defaults.

    Returns:
        Tuple of (fitted_pipeline, validation_metrics_dict).
    """
    if settings is None:
        settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("Loading dataset from: %s", settings.data_path)
    raw_df = load_data(settings.data_path)
    logger.info("Loaded %d raw rows.", len(raw_df))

    logger.info("Cleaning data...")
    cleaned_df = clean_data(
        raw_df,
        min_duration=settings.duration_min,
        max_duration_quantile=settings.duration_max_quantile,
        min_distance=settings.distance_min,
        pickup_col=settings.pickup_column,
        dropoff_col=settings.dropoff_column,
        target_col=settings.target_column,
    )
    logger.info("Cleaned dataset retained %d rows.", len(cleaned_df))

    logger.info("Splitting dataset...")
    X_train_df, X_val_df, _X_test_df, y_train, y_val, _y_test = split_data(
        cleaned_df,
        target_col=settings.target_column,
        test_size=settings.test_size,
        val_size=settings.val_size,
        random_state=settings.random_state,
    )

    X_train = prepare_features(X_train_df)
    X_val = prepare_features(X_val_df)

    model = create_model(
        model_type=settings.model_type,
        n_estimators=settings.n_estimators,
        n_jobs=settings.n_jobs,
        random_state=settings.random_state,
    )

    pipeline = train_pipeline(X_train, y_train, model=model)
    metrics = evaluate_pipeline(pipeline, X_val, y_val)

    save_artifact(pipeline, settings.model_path)
    return pipeline, metrics


def main() -> None:
    """CLI entry point for training (invoked via prodml-train)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train NYC green taxi trip duration regression model."
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="Path to input parquet dataset."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Destination path for serialized model artifact.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["random_forest", "linear_regression"],
        help="Type of regression model to train.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of trees for Random Forest.",
    )

    args = parser.parse_args()

    # Load base settings and apply CLI overrides if supplied
    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.data_path is not None:
        overrides["data_path"] = Path(args.data_path)
    if args.model_path is not None:
        overrides["models_dir"] = Path(args.model_path).parent
        overrides["model_name"] = Path(args.model_path).name
    if args.model_type is not None:
        overrides["model_type"] = args.model_type
    if args.n_estimators is not None:
        overrides["n_estimators"] = args.n_estimators

    if overrides:
        settings = settings.model_copy(update=overrides)

    run_training(settings=settings)


if __name__ == "__main__":
    main()
