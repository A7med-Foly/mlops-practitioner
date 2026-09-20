"""Unit tests for Continuous Training (CT) promotion gate."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from prodml.ct_gate import evaluate_ct_gate, run_ct_gate


def test_evaluate_ct_gate_candidate_wins():
    """Test candidate strictly better than Production without margin."""
    passed, reason = evaluate_ct_gate(
        candidate_mae=1.75, production_mae=1.8271, margin=0.0
    )
    assert passed is True
    assert "beats Production MAE" in reason


def test_evaluate_ct_gate_candidate_loses():
    """Test candidate worse than Production."""
    passed, reason = evaluate_ct_gate(
        candidate_mae=1.95, production_mae=1.8271, margin=0.0
    )
    assert passed is False
    assert "did NOT beat Production MAE" in reason


def test_evaluate_ct_gate_with_margin():
    """Test margin enforcement: candidate must beat production by at least margin."""
    # Candidate is 1.80, Prod is 1.82. Margin is 0.05.
    # 1.80 is not < 1.82 - 0.05 (1.77) -> False
    passed, reason = evaluate_ct_gate(
        candidate_mae=1.80, production_mae=1.82, margin=0.05
    )
    assert passed is False

    # Candidate is 1.70, Prod is 1.82. Margin is 0.05.
    # 1.70 < 1.77 -> True
    passed, reason = evaluate_ct_gate(
        candidate_mae=1.70, production_mae=1.82, margin=0.05
    )
    assert passed is True


def test_run_ct_gate_passed(tmp_path: Path):
    """Test run_ct_gate when candidate passes challenge."""
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"mae": 1.70, "rmse": 5.0, "r2": 0.80}))

    with patch("prodml.ct_gate.get_production_model_version", return_value=None):
        result = run_ct_gate(
            candidate_metrics_path=str(metrics_file),
            margin=0.0,
            fallback_production_mae=1.8271,
        )

    assert result["promoted_to_staging"] is True
    assert result["candidate_mae"] == 1.70
    assert result["production_mae"] == 1.8271


def test_run_ct_gate_rejected(tmp_path: Path):
    """Test run_ct_gate when candidate fails challenge."""
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"mae": 2.10, "rmse": 6.0, "r2": 0.65}))

    with patch("prodml.ct_gate.get_production_model_version", return_value=None):
        result = run_ct_gate(
            candidate_metrics_path=str(metrics_file),
            margin=0.0,
            fallback_production_mae=1.8271,
        )

    assert result["promoted_to_staging"] is False
    assert result["candidate_mae"] == 2.10


def test_run_ct_gate_missing_file():
    """Test run_ct_gate raises FileNotFoundError when metrics file does not exist."""
    with pytest.raises(FileNotFoundError):
        run_ct_gate(candidate_metrics_path="non_existent_metrics.json")
