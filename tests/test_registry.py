"""Unit tests for MLflow Model Registry integration and promotion gate."""

from unittest.mock import MagicMock
import pytest

from prodml.registry import (
    get_client,
    get_production_model_version,
    is_metric_better,
    promote_if_better,
    transition_stage,
)


def test_is_metric_better():
    """Verify comparison logic for lower-is-better and higher-is-better metrics."""
    # MAE / RMSE / loss: lower is better
    assert is_metric_better("mae", 1.80, 2.50) is True
    assert is_metric_better("mae", 2.50, 1.80) is False
    assert is_metric_better("rmse", 5.2, 5.8) is True
    assert is_metric_better("loss", 0.05, 0.10) is True

    # R2 / accuracy / f1: higher is better
    assert is_metric_better("r2", 0.78, 0.65) is True
    assert is_metric_better("r2", 0.60, 0.75) is False
    assert is_metric_better("r2_score", 0.82, 0.70) is True
    assert is_metric_better("accuracy", 0.95, 0.90) is True


def test_get_production_model_version_found():
    """Test retrieving active production model when one exists."""
    client = MagicMock()
    v1 = MagicMock(version="1", current_stage="Staging")
    v2 = MagicMock(version="2", current_stage="Production")
    client.search_model_versions.return_value = [v1, v2]

    prod = get_production_model_version("ride-duration-predictor", client)
    assert prod is not None
    assert prod.version == "2"


def test_get_production_model_version_none():
    """Test retrieving active production model when none is in Production."""
    client = MagicMock()
    v1 = MagicMock(version="1", current_stage="Staging")
    v2 = MagicMock(version="2", current_stage="None")
    client.search_model_versions.return_value = [v1, v2]

    prod = get_production_model_version("ride-duration-predictor", client)
    assert prod is None


def test_promote_if_better_no_current_production():
    """When no production model exists, candidate must be promoted immediately."""
    client = MagicMock()
    client.search_model_versions.return_value = []

    candidate_run = MagicMock()
    candidate_run.data.metrics = {"mae": 2.15}
    client.get_run.return_value = candidate_run

    new_version = MagicMock(version="1", current_stage="None")
    promoted_version = MagicMock(version="1", current_stage="Production")
    client.create_model_version.return_value = new_version
    client.transition_model_version_stage.return_value = promoted_version

    result = promote_if_better("candidate-run-123", client=client)
    assert result["promoted"] is True
    assert result["candidate_version"] == "1"
    assert result["previous_version"] is None
    assert result["candidate_metric"] == 2.15


def test_promote_if_better_candidate_is_superior():
    """Candidate with lower MAE must beat the current Production model."""
    client = MagicMock()

    # Current Production model
    prod_version = MagicMock(
        version="1", current_stage="Production", run_id="prod-run-001"
    )
    client.search_model_versions.return_value = [prod_version]

    candidate_run = MagicMock()
    candidate_run.data.metrics = {"mae": 1.75}

    prod_run = MagicMock()
    prod_run.data.metrics = {"mae": 2.40}

    def get_run_side_effect(run_id):
        if run_id == "cand-run-002":
            return candidate_run
        elif run_id == "prod-run-001":
            return prod_run
        raise ValueError(f"Unknown run {run_id}")

    client.get_run.side_effect = get_run_side_effect

    new_version = MagicMock(version="2", current_stage="None")
    promoted_version = MagicMock(version="2", current_stage="Production")
    client.create_model_version.return_value = new_version
    client.transition_model_version_stage.return_value = promoted_version

    result = promote_if_better("cand-run-002", client=client)
    assert result["promoted"] is True
    assert result["candidate_version"] == "2"
    assert result["previous_version"] == "1"
    assert result["candidate_metric"] == 1.75
    assert result["production_metric"] == 2.40


def test_promote_if_better_candidate_is_inferior():
    """Candidate with higher MAE must be rejected and not promoted."""
    client = MagicMock()

    prod_version = MagicMock(
        version="1", current_stage="Production", run_id="prod-run-001"
    )
    client.search_model_versions.return_value = [prod_version]

    candidate_run = MagicMock()
    candidate_run.data.metrics = {"mae": 3.80}

    prod_run = MagicMock()
    prod_run.data.metrics = {"mae": 1.82}

    def get_run_side_effect(run_id):
        if run_id == "cand-run-worse":
            return candidate_run
        return prod_run

    client.get_run.side_effect = get_run_side_effect

    result = promote_if_better("cand-run-worse", client=client)
    assert result["promoted"] is False
    assert result["candidate_version"] is None
    assert result["current_production_version"] == "1"
    assert result["candidate_metric"] == 3.80
    assert result["production_metric"] == 1.82
    # Ensure transition was never called
    client.transition_model_version_stage.assert_not_called()


def test_promote_if_better_missing_metric_raises():
    """Candidate missing target metric must raise ValueError."""
    client = MagicMock()
    candidate_run = MagicMock()
    candidate_run.data.metrics = {"rmse": 5.0}  # No 'mae'
    client.get_run.return_value = candidate_run

    with pytest.raises(ValueError, match="Metric 'mae' not found"):
        promote_if_better("cand-run-no-mae", metric="mae", client=client)


def test_transition_stage_and_register():
    """Test transition_stage and register_candidate_model with client mock."""
    client = MagicMock()
    target_ver = MagicMock(version="3", current_stage="Staging")
    client.transition_model_version_stage.return_value = target_ver

    updated = transition_stage("test-model", version=3, stage="Staging", client=client)
    assert updated.current_stage == "Staging"
    client.transition_model_version_stage.assert_called_once()


def test_get_client_configuration():
    """Verify get_client populates environment credentials properly."""
    client = get_client()
    assert client is not None
