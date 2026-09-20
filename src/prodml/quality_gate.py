"""Model quality gate module for Continuous Integration.

Compares candidate evaluation metrics against the baseline / Production model metrics.
Fails CI with an exit code of 1 if the target error metric (MAE) regresses by more
than the allowed threshold (default 5%).
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.quality_gate")

DEFAULT_CANONICAL_MAE: float = 1.8271


def load_metric_from_json(file_path: Path | str, metric_name: str = "mae") -> float:
    """Load metric value from a JSON file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    if metric_name not in data:
        raise KeyError(
            f"Metric '{metric_name}' not found in {path}. Available: {list(data.keys())}"
        )
    return float(data[metric_name])


def get_git_baseline_metric(
    git_ref: str = "origin/main:metrics.json", metric_name: str = "mae"
) -> float | None:
    """Attempt to load baseline metrics from a git revision (e.g. origin/main)."""
    try:
        res = subprocess.run(
            ["git", "show", git_ref],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(res.stdout)
        if metric_name in data:
            return float(data[metric_name])
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, OSError):
        pass
    return None


def evaluate_quality_gate(
    candidate_mae: float,
    baseline_mae: float,
    max_regression_pct: float = 0.05,
    metric_name: str = "MAE",
) -> tuple[bool, float]:
    """Compare candidate metric against baseline.

    Args:
        candidate_mae: Candidate model metric value (lower is better).
        baseline_mae: Baseline / Production model metric value.
        max_regression_pct: Maximum allowed degradation percentage (e.g. 0.05 = 5%).
        metric_name: Name of metric for reporting.

    Returns:
        Tuple of (passed: bool, regression_pct: float).
    """
    if baseline_mae <= 0:
        raise ValueError(f"Baseline {metric_name} must be positive, got {baseline_mae}")

    regression_pct = (candidate_mae - baseline_mae) / baseline_mae
    passed = regression_pct <= max_regression_pct

    return passed, regression_pct


def run_quality_gate(
    candidate_metrics_path: Path | str = "metrics.json",
    baseline_metrics_path: Path | str | None = None,
    candidate_mae: float | None = None,
    baseline_mae: float | None = None,
    max_regression_pct: float = 0.05,
    git_ref: str = "origin/main:metrics.json",
) -> bool:
    """Execute quality gate comparison and log formatted outcome."""
    # 1. Resolve candidate metric
    if candidate_mae is None:
        candidate_mae = load_metric_from_json(candidate_metrics_path, metric_name="mae")

    # 2. Resolve baseline metric
    if baseline_mae is None:
        if baseline_metrics_path and Path(baseline_metrics_path).exists():
            baseline_mae = load_metric_from_json(
                baseline_metrics_path, metric_name="mae"
            )
        else:
            # Try git origin/main:metrics.json
            git_val = get_git_baseline_metric(git_ref=git_ref, metric_name="mae")
            if git_val is not None:
                baseline_mae = git_val
                logger.info(
                    "Loaded baseline MAE from git ref '%s': %.4f", git_ref, baseline_mae
                )
            else:
                baseline_mae = DEFAULT_CANONICAL_MAE
                logger.info(
                    "Falling back to registered canonical baseline MAE: %.4f",
                    baseline_mae,
                )

    passed, regression_pct = evaluate_quality_gate(
        candidate_mae=candidate_mae,
        baseline_mae=baseline_mae,
        max_regression_pct=max_regression_pct,
    )

    pct_str = f"{regression_pct * 100:+.2f}%"
    threshold_str = f"{max_regression_pct * 100:.1f}%"

    print("\n" + "=" * 60)
    print("           MODEL QUALITY GATE EVALUATION")
    print("=" * 60)
    print(f"Candidate MAE  : {candidate_mae:.4f} min")
    print(f"Baseline MAE   : {baseline_mae:.4f} min")
    print(f"Observed Delta : {pct_str} (Allowed limit: +{threshold_str})")
    print("-" * 60)

    if passed:
        print(
            f"✅ QUALITY GATE PASSED: Model quality is acceptable ({pct_str} <= +{threshold_str})."
        )
        print("=" * 60 + "\n")
        return True
    else:
        print(
            f"❌ QUALITY GATE FAILED: Candidate MAE regressed by {pct_str}, exceeding the +{threshold_str} limit!"
        )
        print("   Merge blocked to prevent serving an inferior model in production.")
        print("=" * 60 + "\n")
        return False


def main() -> None:
    """CLI entry point for CI model quality gate."""
    parser = argparse.ArgumentParser(
        description="Enforce model quality gate against metric regressions in CI."
    )
    parser.add_argument(
        "--candidate-metrics",
        type=str,
        default="metrics.json",
        help="Path to candidate metrics JSON file (default: metrics.json)",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to baseline metrics JSON file (optional)",
    )
    parser.add_argument(
        "--candidate-mae",
        type=float,
        default=None,
        help="Override candidate MAE directly as a float",
    )
    parser.add_argument(
        "--baseline-mae",
        type=float,
        default=None,
        help="Override baseline MAE directly as a float",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Maximum allowed fractional regression (default: 0.05 for 5%%)",
    )
    parser.add_argument(
        "--git-ref",
        type=str,
        default="origin/main:metrics.json",
        help="Git ref for baseline metrics (default: origin/main:metrics.json)",
    )

    args = parser.parse_args()

    passed = run_quality_gate(
        candidate_metrics_path=args.candidate_metrics,
        baseline_metrics_path=args.baseline_metrics,
        candidate_mae=args.candidate_mae,
        baseline_mae=args.baseline_mae,
        max_regression_pct=args.max_regression,
        git_ref=args.git_ref,
    )

    if not passed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
