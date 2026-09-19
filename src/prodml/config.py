"""Configuration module for prodml.

Manages application paths, hyperparameters, and runtime settings using
pydantic-settings. All settings can be overridden via environment variables
with the 'PRODML_' prefix or a local .env file.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and modeling configuration with sane defaults."""

    model_config = SettingsConfigDict(
        env_prefix="PRODML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # Storage & Artifact Paths (No hardcoded absolute paths)
    # ---------------------------------------------------------
    data_path: Path = Path("data/green_tripdata_2026-05.parquet")
    models_dir: Path = Path("models")
    model_name: str = "baseline.onnx"
    model_uri: str = Field(
        default="models:/ride-duration-predictor/Production",
        validation_alias=AliasChoices("PRODML_MODEL_URI", "MODEL_URI"),
    )

    @property
    def model_path(self) -> Path:
        """Full path to the model artifact file."""
        return self.models_dir / self.model_name

    # ---------------------------------------------------------
    # Data Cleaning & Splitting Hyperparameters
    # ---------------------------------------------------------
    pickup_column: str = "lpep_pickup_datetime"
    dropoff_column: str = "lpep_dropoff_datetime"
    target_column: str = "trip_duration"

    duration_min: float = 0.0
    duration_max_quantile: float = 0.995
    distance_min: float = 0.0

    test_size: float = 0.2
    val_size: float = 0.25
    random_state: int = 42

    # ---------------------------------------------------------
    # Model Hyperparameters
    # ---------------------------------------------------------
    model_type: Literal["random_forest", "linear_regression"] = "random_forest"
    n_estimators: int = 100
    n_jobs: int = -1

    # ---------------------------------------------------------
    # Service & Network Ports (Seam for Module 3 API & BentoML)
    # ---------------------------------------------------------
    server_host: str = "0.0.0.0"
    server_port: int = 8000
    log_level: str = "INFO"

    # ---------------------------------------------------------
    # Tracking Server & MLflow Configuration (Module 2)
    # ---------------------------------------------------------
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        validation_alias=AliasChoices(
            "PRODML_MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_URI"
        ),
    )
    mlflow_experiment_name: str = Field(
        default="nyc-taxi-duration",
        validation_alias=AliasChoices(
            "PRODML_MLFLOW_EXPERIMENT_NAME", "MLFLOW_EXPERIMENT_NAME"
        ),
    )
    mlflow_s3_endpoint_url: str = Field(
        default="http://localhost:9000",
        validation_alias=AliasChoices(
            "MLFLOW_S3_ENDPOINT_URL", "PRODML_MLFLOW_S3_ENDPOINT_URL"
        ),
    )
    aws_access_key_id: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices("AWS_ACCESS_KEY_ID", "PRODML_AWS_ACCESS_KEY_ID"),
    )
    aws_secret_access_key: str = Field(
        default="minioadmin",
        validation_alias=AliasChoices(
            "AWS_SECRET_ACCESS_KEY", "PRODML_AWS_SECRET_ACCESS_KEY"
        ),
    )
    aws_default_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices(
            "AWS_DEFAULT_REGION", "PRODML_AWS_DEFAULT_REGION"
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings and configure client env."""
    settings = Settings()

    # Ensure AWS and MLflow environment variables are set for boto3 / mlflow client
    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_DEFAULT_REGION", settings.aws_default_region)
    os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)

    return settings
