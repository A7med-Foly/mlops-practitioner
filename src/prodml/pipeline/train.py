"""Pipeline stage 3: Model Training.

Loads engineered feature matrices, fits the regression model specified in params.yaml,
captures MLflow run metadata with DVC data hash lineage, and serializes the model artifact.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
import yaml

from prodml.config import get_settings
from prodml.pipeline.lineage import get_dvc_hash

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.pipeline.train")


def get_git_commit() -> str:
    """Retrieve current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown-commit"


def run_train(
    features_dir: str | Path = "data/features",
    models_dir: str | Path = "models",
    params_path: str | Path = "params.yaml",
) -> None:
    """Train model using feature matrices and log to MLflow with DVC data lineage."""
    logger.info("Loading parameters from %s", params_path)
    with open(params_path, encoding="utf-8") as f:
        params: dict[str, Any] = yaml.safe_load(f).get("train", {})

    feat_path = Path(features_dir)
    train_npz = feat_path / "train.npz"
    val_npz = feat_path / "val.npz"

    if not train_npz.exists() or not val_npz.exists():
        raise FileNotFoundError(f"Feature matrices missing: {train_npz}, {val_npz}")

    logger.info("Loading feature arrays from %s and %s", train_npz, val_npz)
    with np.load(train_npz) as d:
        X_train, y_train = d["X"], d["y"]
    with np.load(val_npz) as d:
        X_val = d["X"]

    xgb_params = {
        "n_estimators": int(params.get("n_estimators", 100)),
        "max_depth": int(params.get("max_depth", 6)),
        "learning_rate": float(params.get("learning_rate", 0.1)),
        "subsample": float(params.get("subsample", 0.8)),
        "colsample_bytree": float(params.get("colsample_bytree", 0.8)),
        "random_state": int(params.get("random_state", 42)),
        "n_jobs": -1,
    }

    logger.info("Initializing XGBoost regressor with params: %s", xgb_params)
    model = xgb.XGBRegressor(**xgb_params)

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_sec = time.perf_counter() - t0
    logger.info("Training completed in %.2f seconds", train_sec)

    # Save model artifact
    out_models = Path(models_dir)
    out_models.mkdir(parents=True, exist_ok=True)
    model_artifact = out_models / "dvc_model.joblib"
    joblib.dump(model, model_artifact)
    logger.info("Saved trained model artifact to %s", model_artifact)

    # Lineage tracking
    dvc_data_hash = get_dvc_hash("data/processed")
    if dvc_data_hash in ("unknown-dvc-hash", "file-not-found"):
        dvc_data_hash = get_dvc_hash("data/raw/green_tripdata.parquet")
    git_commit = get_git_commit()

    logger.info(
        "Lineage Info - DVC Data Hash: %s, Git Commit: %s", dvc_data_hash, git_commit
    )

    # MLflow tracking
    settings = get_settings()
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        run_name = f"dvc-pipeline-{params.get('model_family', 'xgboost')}"
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params(xgb_params)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("val_samples", len(X_val))
            mlflow.log_metric("train_duration_sec", train_sec)

            # Essential provenance tags
            mlflow.set_tag("dvc_data_hash", dvc_data_hash)
            mlflow.set_tag("git_commit", git_commit)
            mlflow.set_tag("pipeline", "dvc")
            mlflow.set_tag("framework", "xgboost")

            mlflow.xgboost.log_model(model, artifact_path="model")
            logger.info("Logged training run to MLflow (Run ID: %s)", run.info.run_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not log to MLflow server: %s. Continuing pipeline...", exc
        )


if __name__ == "__main__":
    run_train()
