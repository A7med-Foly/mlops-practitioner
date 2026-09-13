"""Configuration module for prodml.

Manages application paths, hyperparameters, and runtime settings using
pydantic-settings. All settings can be overridden via environment variables
with the 'PRODML_' prefix or a local .env file.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    model_name: str = "baseline.pkl"

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


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
