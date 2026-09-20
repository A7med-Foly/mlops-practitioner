"""Model training, evaluation, experiment tracking, and persistence module for prodml.

Integrates MLflow tracking for multiple model families (Linear Regression, XGBoost, PyTorch MLP),
Optuna nested hyperparameter sweeps, artifact and plot generation, and CLI entrypoint.
"""

from collections.abc import Callable
import hashlib
import logging
from pathlib import Path
import pickle
import subprocess
import tempfile
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from prodml.config import Settings, get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.features import (
    ORDERED_FEATURE_NAMES,
    build_pipeline,
    features_to_matrix,
    prepare_features,
)

logger = logging.getLogger("prodml.train")


# =====================================================================
# Metadata and Plotting Helpers
# =====================================================================


def get_git_commit_hash() -> str:
    """Retrieve current git commit SHA-1 or return fallback."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted"


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA256 checksum of the dataset file for data version tracking."""
    if not file_path.exists():
        return "artifact-not-found"
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def dump_requirements_txt(output_path: Path) -> Path:
    """Write exact pinned requirements for runtime reproducibility."""
    content = (
        "fastapi>=0.141.1\n"
        "mlflow>=2.20.0\n"
        "numpy>=2.5.3\n"
        "onnxruntime>=1.30.0\n"
        "optuna>=4.0.0\n"
        "pandas>=3.0.5\n"
        "pyarrow>=25.0.1\n"
        "pydantic>=2.13.5\n"
        "scikit-learn>=1.9.1\n"
        "torch>=2.0.0\n"
        "xgboost>=3.0.0\n"
        "boto3>=1.35.0\n"
        "psycopg2-binary>=2.9.10\n"
        "matplotlib>=3.8.0\n"
    )
    output_path.write_text(content)
    return output_path


def create_residual_plot(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: Path
) -> Path:
    """Generate and save residual scatter and distribution plots."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter of predictions vs residuals
    ax1.scatter(y_pred, residuals, alpha=0.3, s=12, color="#008080")
    ax1.axhline(0, color="red", linestyle="--", lw=1.5)
    ax1.set_xlabel("Predicted Duration (min)")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    ax1.set_title("Residuals vs. Predictions")
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
    ax2.hist(residuals, bins=50, color="#008080", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual (min)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_feature_importance_plot(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: Path,
    title: str = "Feature Importance",
) -> Path:
    """Generate and save horizontal feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    ax.barh(
        range(len(sorted_names)),
        sorted_importances,
        color="#2E86C1",
        align="center",
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance Score")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# =====================================================================
# PyTorch MLP Architecture
# =====================================================================


class PyTorchMLP(nn.Module):
    """Small PyTorch Multi-Layer Perceptron regressor for trip duration prediction."""

    def __init__(
        self, input_dim: int = 9, hidden_dims: tuple[int, int] = (64, 32)
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# =====================================================================
# Legacy Sklearn Pipeline & Estimator Helpers (Backwards Compatible)
# =====================================================================


def create_model(model_type: str = "random_forest", **kwargs: Any) -> RegressorMixin:
    """Instantiate the regressor specified by model_type."""
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
    """Build and fit a pipeline containing DictVectorizer and the given regressor."""
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
    """Compute evaluation metrics (MAE, RMSE, R2) on validation data."""
    y_pred = pipeline.predict(X_val)
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    logger.info("Validation Metrics - MAE: %.4f, RMSE: %.4f, R2: %.4f", mae, rmse, r2)
    return metrics


def save_artifact(artifact: Any, path: Path | str) -> None:
    """Serialize and save an artifact (e.g. Pipeline or dict) using pickle."""
    dest_path = Path(path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        pickle.dump(artifact, f)
    logger.info("Saved artifact to %s", dest_path.resolve())


# =====================================================================
# Common Run Logging Routine
# =====================================================================


def log_run_metadata_and_artifacts(
    y_val: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    importances: np.ndarray,
    model_artifact_path: Path | None,
    log_model_fn: Callable[[], None] | None,
    train_duration: float,
    params: dict[str, Any],
    tags: dict[str, str],
    temp_dir: Path,
) -> dict[str, float]:
    """Compute evaluation metrics, generate plots, log params, metrics, artifacts, and tags."""
    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    model_size_mb = 0.0
    if model_artifact_path and model_artifact_path.exists():
        model_size_mb = model_artifact_path.stat().st_size / (1024 * 1024)

    # 1. Log Params
    mlflow.log_params(params)

    # 2. Log Metrics
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "train_duration_sec": round(train_duration, 4),
        "model_size_mb": round(model_size_mb, 4),
    }
    mlflow.log_metrics(metrics)

    # 3. Log Tags
    mlflow.set_tags(tags)

    # 4. Generate and Log Plots & Requirements Artifacts
    res_plot_path = create_residual_plot(y_val, y_pred, temp_dir / "residual_plot.png")
    feat_plot_path = create_feature_importance_plot(
        feature_names, importances, temp_dir / "feature_importance.png"
    )
    req_path = dump_requirements_txt(temp_dir / "requirements.txt")

    mlflow.log_artifact(str(res_plot_path))
    mlflow.log_artifact(str(feat_plot_path))
    mlflow.log_artifact(str(req_path))

    # 5. Log Model Artifact
    if log_model_fn is not None:
        try:
            log_model_fn()
        except Exception as exc:
            logger.warning("Failed to log native framework model artifact: %s", exc)

    logger.info(
        "Run logged successfully - Framework: %s, MAE: %.4f, RMSE: %.4f, R2: %.4f",
        tags.get("framework", "unknown"),
        mae,
        rmse,
        r2,
    )
    return metrics


# =====================================================================
# Model Family Trainers
# =====================================================================


def train_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
    git_commit: str,
    data_version: str,
    data_hash: str,
) -> dict[str, float]:
    """Train and log Linear Regression baseline model."""
    logger.info("Starting Linear Regression training run...")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        t0 = time.perf_counter()
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        duration = time.perf_counter() - t0

        y_pred = model.predict(X_val)

        # Model size
        model_file = tmp_dir / "model.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(model, f)

        # Feature importance from absolute coefficients
        importances = np.abs(model.coef_)

        params = {
            "model_family": "linear_regression",
            "fit_intercept": True,
            "split_seed": settings.random_state,
            "data_version_hash": data_hash,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": X_train.shape[1],
        }
        tags = {
            "framework": "scikit-learn",
            "git_commit": git_commit,
            "data_version": data_version,
            "author": "Ahmed Foly",
        }

        with mlflow.start_run(run_name="linear-regression-baseline") as run:
            metrics = log_run_metadata_and_artifacts(
                y_val=y_val,
                y_pred=y_pred,
                feature_names=ORDERED_FEATURE_NAMES,
                importances=importances,
                model_artifact_path=model_file,
                log_model_fn=lambda: mlflow.sklearn.log_model(model, "model"),
                train_duration=duration,
                params=params,
                tags=tags,
                temp_dir=tmp_dir,
            )
            logger.info("Linear Regression Run ID: %s", run.info.run_id)
            return metrics


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
    git_commit: str,
    data_version: str,
    data_hash: str,
    hyperparams: dict[str, Any] | None = None,
    run_name: str = "xgboost-baseline",
    nested: bool = False,
    enable_autolog: bool = True,
) -> tuple[dict[str, float], xgb.XGBRegressor]:
    """Train and log XGBoost regressor with autologging option."""
    if hyperparams is None:
        hyperparams = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": settings.random_state,
        }

    if enable_autolog:
        mlflow.xgboost.autolog(log_models=False, log_datasets=False)

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)
        t0 = time.perf_counter()
        model = xgb.XGBRegressor(**hyperparams)
        model.fit(X_train, y_train)
        duration = time.perf_counter() - t0

        y_pred = model.predict(X_val)

        # Model size
        model_file = tmp_dir / "model.json"
        model.save_model(str(model_file))

        importances = model.feature_importances_

        params = {
            "model_family": "xgboost",
            "split_seed": settings.random_state,
            "data_version_hash": data_hash,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": X_train.shape[1],
            **hyperparams,
        }
        tags = {
            "framework": "xgboost",
            "git_commit": git_commit,
            "data_version": data_version,
            "author": "Ahmed Foly",
        }

        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            metrics = log_run_metadata_and_artifacts(
                y_val=y_val,
                y_pred=y_pred,
                feature_names=ORDERED_FEATURE_NAMES,
                importances=importances,
                model_artifact_path=model_file,
                log_model_fn=lambda: mlflow.xgboost.log_model(model, "model"),
                train_duration=duration,
                params=params,
                tags=tags,
                temp_dir=tmp_dir,
            )
            logger.info("XGBoost Run ID: %s (%s)", run.info.run_id, run_name)
            return metrics, model


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
    git_commit: str,
    data_version: str,
    data_hash: str,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 0.005,
) -> dict[str, float]:
    """Train and log PyTorch Multi-Layer Perceptron regressor."""
    logger.info("Starting PyTorch MLP training run...")
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp_dir = Path(tmp_str)

        torch.manual_seed(settings.random_state)
        model = PyTorchMLP(input_dim=X_train.shape[1], hidden_dims=(64, 32))
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        t0 = time.perf_counter()
        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        duration = time.perf_counter() - t0

        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val.astype(np.float32))
            y_pred = model(X_val_tensor).numpy()

        # Model size
        model_file = tmp_dir / "model.pt"
        torch.save(model.state_dict(), model_file)

        # Feature importance derived from first layer weight magnitudes
        with torch.no_grad():
            importances = model.net[0].weight.abs().mean(dim=0).numpy()

        params = {
            "model_family": "pytorch_mlp",
            "hidden_dims": "(64, 32)",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "optimizer": "Adam",
            "split_seed": settings.random_state,
            "data_version_hash": data_hash,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": X_train.shape[1],
        }
        tags = {
            "framework": "pytorch",
            "git_commit": git_commit,
            "data_version": data_version,
            "author": "Ahmed Foly",
        }

        with mlflow.start_run(run_name="pytorch-mlp") as run:
            metrics = log_run_metadata_and_artifacts(
                y_val=y_val,
                y_pred=y_pred,
                feature_names=ORDERED_FEATURE_NAMES,
                importances=importances,
                model_artifact_path=model_file,
                log_model_fn=lambda: mlflow.pytorch.log_model(model, "model"),
                train_duration=duration,
                params=params,
                tags=tags,
                temp_dir=tmp_dir,
            )
            logger.info("PyTorch MLP Run ID: %s", run.info.run_id)
            return metrics


def run_xgboost_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
    git_commit: str,
    data_version: str,
    data_hash: str,
    n_trials: int = 10,
) -> dict[str, Any]:
    """Execute Optuna hyperparameter sweep on XGBoost with nested MLflow runs."""
    logger.info(
        "Starting Optuna hyperparameter sweep on XGBoost (%d trials)...", n_trials
    )
    # Turn off autologging during sweep to keep trial runs clean
    mlflow.xgboost.autolog(disable=True)

    parent_tags = {
        "framework": "xgboost",
        "git_commit": git_commit,
        "data_version": data_version,
        "author": "Ahmed Foly",
        "experiment_type": "hyperparameter_sweep",
    }

    with mlflow.start_run(
        run_name="xgboost-hyperparameter-sweep", tags=parent_tags
    ) as parent_run:
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimizer", "optuna")

        def objective(trial: optuna.Trial) -> float:
            hyperparams = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.03, 0.2, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 1.0, step=0.1
                ),
                "random_state": settings.random_state,
            }

            metrics, _ = train_xgboost(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                settings=settings,
                git_commit=git_commit,
                data_version=data_version,
                data_hash=data_hash,
                hyperparams=hyperparams,
                run_name=f"trial-{trial.number}",
                nested=True,
                enable_autolog=False,
            )
            return metrics["mae"]

        # Run Optuna study
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_value = study.best_value

        mlflow.log_metrics({"best_mae": best_value})
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

        logger.info(
            "Optuna sweep complete. Best Trial: #%d, Best MAE: %.4f",
            study.best_trial.number,
            best_value,
        )
        logger.info("Best Hyperparameters: %s", best_params)

        return {
            "parent_run_id": parent_run.info.run_id,
            "best_mae": best_value,
            "best_params": best_params,
        }


# =====================================================================
# Main Orchestrator
# =====================================================================


def run_training_pipeline(
    family: str = "all", settings: Settings | None = None
) -> dict[str, Any]:
    """Load data, split, configure MLflow, and execute chosen training workflows."""
    if settings is None:
        settings = get_settings()

    # Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    logger.info("MLflow Tracking URI: %s", settings.mlflow_tracking_uri)
    logger.info("MLflow Experiment: %s", settings.mlflow_experiment_name)

    # 1. Ingestion & Preprocessing
    logger.info("Loading dataset from: %s", settings.data_path)
    raw_df = load_data(settings.data_path)
    cleaned_df = clean_data(
        raw_df,
        min_duration=settings.duration_min,
        max_duration_quantile=settings.duration_max_quantile,
        min_distance=settings.distance_min,
        pickup_col=settings.pickup_column,
        dropoff_col=settings.dropoff_column,
        target_col=settings.target_column,
    )

    X_train_df, X_val_df, _X_test_df, y_train, y_val, _y_test = split_data(
        cleaned_df,
        target_col=settings.target_column,
        test_size=settings.test_size,
        val_size=settings.val_size,
        random_state=settings.random_state,
    )

    X_train = features_to_matrix(prepare_features(X_train_df), ORDERED_FEATURE_NAMES)
    X_val = features_to_matrix(prepare_features(X_val_df), ORDERED_FEATURE_NAMES)
    y_train_arr = y_train.to_numpy(dtype=np.float32)
    y_val_arr = y_val.to_numpy(dtype=np.float32)

    # Provenance tags
    git_commit = get_git_commit_hash()
    data_hash = compute_file_sha256(settings.data_path)
    data_version = data_hash[:8]

    logger.info("Provenance - Git Commit: %s, Data Hash: %s", git_commit, data_version)

    results: dict[str, Any] = {}

    if family in ("all", "linear"):
        results["linear_regression"] = train_linear_regression(
            X_train,
            y_train_arr,
            X_val,
            y_val_arr,
            settings,
            git_commit,
            data_version,
            data_hash,
        )

    if family in ("all", "pytorch"):
        results["pytorch_mlp"] = train_pytorch_mlp(
            X_train,
            y_train_arr,
            X_val,
            y_val_arr,
            settings,
            git_commit,
            data_version,
            data_hash,
        )

    if family in ("all", "xgboost"):
        results["xgboost_baseline"] = train_xgboost(
            X_train,
            y_train_arr,
            X_val,
            y_val_arr,
            settings,
            git_commit,
            data_version,
            data_hash,
            run_name="xgboost-baseline",
            enable_autolog=True,
        )[0]

    if family in ("all", "sweep"):
        results["xgboost_sweep"] = run_xgboost_sweep(
            X_train,
            y_train_arr,
            X_val,
            y_val_arr,
            settings,
            git_commit,
            data_version,
            data_hash,
            n_trials=10,
        )

    return results


def run_training(
    settings: Settings | None = None,
) -> tuple[Pipeline, dict[str, float]]:
    """Legacy interface for single baseline model fitting (backwards compatibility)."""
    if settings is None:
        settings = get_settings()

    raw_df = load_data(settings.data_path)
    cleaned_df = clean_data(raw_df)
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
        description="Train NYC green taxi trip duration regression models with MLflow tracking."
    )
    parser.add_argument(
        "--family",
        type=str,
        default="all",
        choices=["all", "linear", "xgboost", "pytorch", "sweep", "baseline"],
        help="Model family to train: 'all' (default), 'linear', 'xgboost', 'pytorch', 'sweep', or 'baseline'.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to input parquet dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Destination path for serialized model artifact.",
    )

    args = parser.parse_args()

    settings = get_settings()
    overrides: dict[str, Any] = {}
    if args.data_path is not None:
        overrides["data_path"] = Path(args.data_path)
    if args.model_path is not None:
        overrides["models_dir"] = Path(args.model_path).parent
        overrides["model_name"] = Path(args.model_path).name

    if overrides:
        settings = settings.model_copy(update=overrides)

    if args.family == "baseline":
        run_training(settings=settings)
    else:
        run_training_pipeline(family=args.family, settings=settings)


if __name__ == "__main__":
    main()
