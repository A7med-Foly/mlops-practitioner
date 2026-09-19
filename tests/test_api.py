"""Pytest suite for FastAPI endpoints with schema validation and mocking."""

import pytest
from fastapi.testclient import TestClient

from prodml.api.main import app
from prodml.api.schemas import (
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionResponse,
)


def test_health_returns_200(client: TestClient):
    """GET /health must return 200 and model_loaded: True."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    validated = HealthResponse.model_validate(data)
    assert validated.status == "healthy"
    assert validated.model_loaded is True
    assert "baseline.onnx" in validated.model_path or "models:/" in validated.model_path


def test_metadata_schema_matches(client: TestClient):
    """GET /metadata must return complete model metadata conforming to schema."""
    response = client.get("/metadata")
    assert response.status_code == 200

    data = response.json()
    validated = MetadataResponse.model_validate(data)
    assert validated.model_version in ("0.1.0", "registry-Production")
    assert len(validated.feature_names) > 0
    assert len(validated.artifact_hash) > 0
    assert "ONNX Runtime" in validated.framework or "MLflow" in validated.framework


def test_predict_happy_path(client: TestClient, sample_features: dict):
    """POST /predict happy path returns valid prediction and matching schema."""
    response = client.post(
        "/predict",
        json=sample_features,
        headers={"X-Request-ID": "test-trace-id"},
    )
    assert response.status_code == 200

    data = response.json()
    validated = PredictionResponse.model_validate(data)
    assert validated.prediction > 0.0
    assert validated.model_version == "0.1.0"
    assert validated.correlation_id == "test-trace-id"
    assert validated.latency_ms > 0.0
    assert response.headers.get("X-Request-ID") == "test-trace-id"


def test_predict_invalid_payload_returns_422(client: TestClient):
    """POST /predict with trip_distance: -5 must return clean 422 without stack trace."""
    invalid_payload = {
        "trip_distance": -5.0,
        "fare_amount": 10.0,
        "total_amount": 12.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

    data = response.json()
    assert data["error"] == "Validation Error"
    assert "trip_distance" in data["message"]
    assert "greater than 0" in data["message"]
    assert "Traceback" not in response.text


def test_predict_mocked_dependency(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, sample_features: dict
):
    """Use monkeypatch so API prediction does not depend on a real model training run."""
    # Mock predict_one to return a deterministic value
    monkeypatch.setattr(app.state.predictor, "predict_one", lambda x: 42.42)

    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200

    data = response.json()
    assert data["prediction"] == 42.42
    assert data["model_version"] == "0.1.0"


def test_predict_batch_happy_path(client: TestClient, sample_features: dict):
    """POST /predict/batch returns list of predictions conforming to BatchPredictionResponse schema."""
    batch_payload = {"trips": [sample_features, sample_features]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200

    data = response.json()
    validated = BatchPredictionResponse.model_validate(data)
    assert len(validated.predictions) == 2
    assert validated.count == 2
    assert validated.latency_ms > 0.0
