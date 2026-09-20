"""Pytest fixtures for prodml test suite."""

import os
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app, compute_model_metadata
from prodml.config import get_settings
from prodml.predict import DurationPredictor
from prodml.train import create_model, train_pipeline

# Ensure any accidental network requests to MLflow fail immediately in test runs
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "2"
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"


@pytest.fixture
def sample_features() -> dict[str, float | str]:
    """Provide representative valid trip features."""
    return {
        "trip_distance": 3.5,
        "fare_amount": 15.0,
        "total_amount": 18.5,
        "store_and_fwd_flag": "N",
        "tip_amount": 2.0,
        "tolls_amount": 0.0,
        "improvement_surcharge": 1.0,
        "congestion_surcharge": 0.0,
        "cbd_congestion_fee": 0.0,
    }


@pytest.fixture(scope="session")
def trained_model() -> DurationPredictor:
    """Session-scoped trained DurationPredictor instance using synthetic data.

    Decouples unit testing from physical disk artifacts and prevents slow re-training.
    """
    n_samples = 30
    np.random.seed(42)

    X_records = [
        {
            "trip_distance": float(np.random.uniform(1.0, 15.0)),
            "fare_amount": float(np.random.uniform(5.0, 45.0)),
            "total_amount": float(np.random.uniform(8.0, 55.0)),
            "store_and_fwd_flag_encoded": int(np.random.choice([0, 1])),
            "tip_amount": float(np.random.uniform(0.0, 8.0)),
            "tolls_amount": 0.0,
            "improvement_surcharge": 1.0,
            "congestion_surcharge": 0.0,
            "cbd_congestion_fee": 0.0,
        }
        for _ in range(n_samples)
    ]
    # Sane durations: 5 to 60 minutes
    y_durations = pd.Series(
        [record["trip_distance"] * 3.0 + 5.0 for record in X_records]
    )

    model = create_model("random_forest", n_estimators=10, random_state=42)
    pipeline = train_pipeline(X_records, y_durations, model=model)
    return DurationPredictor(
        model=pipeline,
        model_uri="models:/ride-duration-predictor/Production",
    )


@pytest.fixture
def client(trained_model: DurationPredictor, monkeypatch: pytest.MonkeyPatch):
    """FastAPI TestClient fixture with lifespan execution and mocked model loading."""
    monkeypatch.setattr(
        DurationPredictor, "load", lambda *args, **kwargs: trained_model
    )
    with TestClient(app) as test_client:
        test_client.app.state.predictor = trained_model
        test_client.app.state.metadata = compute_model_metadata(
            get_settings(), trained_model
        )
        yield test_client
