"""Tests for model quality gate module."""

import json
from pathlib import Path

import pytest

from prodml.quality_gate import (
    evaluate_quality_gate,
    get_git_baseline_metric,
    load_metric_from_json,
    run_quality_gate,
)


def test_evaluate_quality_gate_improved():
    # Better model (lower MAE)
    passed, delta = evaluate_quality_gate(candidate_mae=1.70, baseline_mae=1.8271)
    assert passed is True
    assert delta < 0


def test_evaluate_quality_gate_within_tolerance():
    # Slightly worse (+3%), within 5% limit
    passed, delta = evaluate_quality_gate(
        candidate_mae=1.85, baseline_mae=1.8271, max_regression_pct=0.05
    )
    assert passed is True
    assert 0 < delta <= 0.05


def test_evaluate_quality_gate_regressed():
    # Significantly worse (+10%), exceeding 5% limit
    passed, delta = evaluate_quality_gate(
        candidate_mae=2.05, baseline_mae=1.8271, max_regression_pct=0.05
    )
    assert passed is False
    assert delta > 0.05


def test_evaluate_quality_gate_invalid_baseline():
    with pytest.raises(ValueError, match="must be positive"):
        evaluate_quality_gate(candidate_mae=1.8, baseline_mae=0.0)


def test_load_metric_from_json(tmp_path: Path):
    metrics_file = tmp_path / "metrics.json"
    metrics_file.write_text(json.dumps({"mae": 1.8271, "rmse": 5.4446}))

    mae = load_metric_from_json(metrics_file, "mae")
    assert mae == 1.8271

    with pytest.raises(KeyError):
        load_metric_from_json(metrics_file, "non_existent_metric")

    with pytest.raises(FileNotFoundError):
        load_metric_from_json(tmp_path / "missing.json")


def test_run_quality_gate_with_files(tmp_path: Path):
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"

    candidate.write_text(json.dumps({"mae": 1.80}))
    baseline.write_text(json.dumps({"mae": 1.8271}))

    assert (
        run_quality_gate(
            candidate_metrics_path=candidate, baseline_metrics_path=baseline
        )
        is True
    )

    # Now make candidate worse
    candidate.write_text(json.dumps({"mae": 2.10}))
    assert (
        run_quality_gate(
            candidate_metrics_path=candidate, baseline_metrics_path=baseline
        )
        is False
    )


def test_get_git_baseline_metric_nonexistent():
    val = get_git_baseline_metric("nonexistent-ref:metrics.json")
    assert val is None
