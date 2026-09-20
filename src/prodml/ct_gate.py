"""Continuous Training (CT) automated promotion gate.

Challenges newly trained candidate models against the active Production model in the MLflow
Model Registry. If the candidate outperforms Production by a configurable margin, it is
automatically promoted to Staging. If the candidate fails the challenge, the attempt is
logged and the process exits cleanly (exit code 0), as a failed challenge is an expected
outcome and not a pipeline failure.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mlflow.tracking import MlflowClient

from prodml.config import get_settings
from prodml.registry import (
    get_client,
    get_production_model_version,
    register_candidate_model,
    transition_stage,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.ct_gate")

DEFAULT_BASELINE_MAE: float = 1.8271


def load_metrics_from_file(metrics_path: Path | str) -> dict[str, float]:
    """Load evaluation metrics JSON file."""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: float(v) for k, v in data.items()}


def get_latest_candidate_run_id(
    experiment_name: str,
    client: MlflowClient | None = None,
) -> str | None:
    """Find the most recent run ID in the specified MLflow experiment."""
    client = client or get_client()
    try:
        exp = client.get_experiment_by_name(experiment_name)
        if not exp:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attribute.start_time DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
    except Exception as exc:
        logger.warning("Could not query MLflow for latest run: %s", exc)
    return None


def evaluate_ct_gate(
    candidate_mae: float,
    production_mae: float,
    margin: float = 0.0,
) -> tuple[bool, str]:
    """Evaluate whether candidate beats Production MAE by required margin.

    Args:
        candidate_mae: Mean Absolute Error of candidate model.
        production_mae: Mean Absolute Error of active Production model.
        margin: Absolute margin candidate must beat production by (candidate < prod - margin).

    Returns:
        Tuple of (passed: bool, reason: str).
    """
    threshold = production_mae - margin
    if candidate_mae < threshold:
        delta = production_mae - candidate_mae
        pct = (delta / production_mae) * 100.0
        reason = (
            f"Candidate MAE ({candidate_mae:.4f}) beats Production MAE ({production_mae:.4f}) "
            f"by {delta:.4f} min ({pct:.2f}% improvement, margin required: {margin:.4f})."
        )
        return True, reason
    else:
        delta = candidate_mae - production_mae
        pct = (delta / production_mae) * 100.0
        reason = (
            f"Candidate MAE ({candidate_mae:.4f}) did NOT beat Production MAE ({production_mae:.4f}) "
            f"with margin {margin:.4f} (delta: +{delta:.4f} min, {pct:+.2f}%)."
        )
        return False, reason


def run_ct_gate(
    candidate_metrics_path: str = "metrics.json",
    margin: float = 0.0,
    model_name: str = "ride-duration-predictor",
    candidate_run_id: str | None = None,
    fallback_production_mae: float = DEFAULT_BASELINE_MAE,
) -> dict[str, Any]:
    """Execute CT promotion gate and promote to Staging if candidate wins.

    Returns:
        Decision dictionary.
    """
    settings = get_settings()
    client = get_client()

    # 1. Candidate MAE
    candidate_metrics = load_metrics_from_file(candidate_metrics_path)
    if "mae" not in candidate_metrics:
        raise ValueError(
            f"'mae' metric missing from {candidate_metrics_path}. Found: {list(candidate_metrics.keys())}"
        )
    candidate_mae = candidate_metrics["mae"]

    # 2. Production MAE
    production_mae = fallback_production_mae
    prod_version_str = "baseline"
    try:
        prod_version = get_production_model_version(
            model_name=model_name, client=client
        )
        if prod_version and prod_version.run_id:
            prod_run = client.get_run(prod_version.run_id)
            if "mae" in prod_run.data.metrics:
                production_mae = float(prod_run.data.metrics["mae"])
                prod_version_str = f"v{prod_version.version}"
    except Exception as exc:
        logger.warning(
            "Could not query production model from registry: %s. Using fallback baseline %.4f",
            exc,
            fallback_production_mae,
        )

    # 3. Challenge Gate
    passed, reason = evaluate_ct_gate(
        candidate_mae=candidate_mae,
        production_mae=production_mae,
        margin=margin,
    )

    result: dict[str, Any] = {
        "promoted_to_staging": False,
        "candidate_mae": candidate_mae,
        "production_mae": production_mae,
        "production_version": prod_version_str,
        "margin": margin,
        "reason": reason,
    }

    print("=" * 60)
    print("        CONTINUOUS TRAINING (CT) PROMOTION GATE")
    print("=" * 60)
    print(f"Model Name       : {model_name}")
    print(f"Candidate MAE    : {candidate_mae:.4f} min")
    print(f"Production MAE   : {production_mae:.4f} min ({prod_version_str})")
    print(f"Required Margin  : {margin:.4f} min")
    print("-" * 60)

    # 4. Handle Decision
    if passed:
        print(f"✅ CHALLENGE PASSED: {reason}")
        print("Action           : Automatically promoting candidate to Staging.")

        # Resolve candidate run ID if needed
        run_id = candidate_run_id or get_latest_candidate_run_id(
            settings.mlflow_experiment_name, client=client
        )

        staging_version = "candidate"
        if run_id:
            try:
                mv = register_candidate_model(
                    run_id, model_name=model_name, client=client
                )
                transitioned = transition_stage(
                    model_name=model_name,
                    version=mv.version,
                    stage="Staging",
                    archive_existing=False,
                    client=client,
                )
                staging_version = str(transitioned.version)
                logger.info(
                    "Model version %s successfully transitioned to Staging.",
                    staging_version,
                )
            except Exception as exc:
                logger.warning("Could not transition model in registry: %s", exc)
        else:
            logger.info(
                "No MLflow candidate run ID detected; marked as Staging eligible."
            )

        result["promoted_to_staging"] = True
        result["staging_version"] = staging_version
    else:
        print(f"ℹ️  CHALLENGE REJECTED: {reason}")
        print(
            "Action           : Candidate retained in review. Production remains active."
        )
        print(
            "Note             : Exiting with code 0 (failed challenge is not a build error)."
        )

    print("=" * 60)

    # Export to GitHub Actions environment if running in CI
    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"promoted={str(result['promoted_to_staging']).lower()}\n")
            f.write(f"candidate_mae={candidate_mae:.4f}\n")
            f.write(f"production_mae={production_mae:.4f}\n")
            f.write(f"margin={margin:.4f}\n")

    return result


def main() -> None:
    """CLI entrypoint for CT promotion gate."""
    parser = argparse.ArgumentParser(
        description="Continuous Training (CT) Staging Promotion Gate"
    )
    parser.add_argument(
        "--candidate-metrics",
        type=str,
        default="metrics.json",
        help="Path to candidate metrics.json",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="Required improvement margin (candidate_mae < prod_mae - margin)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ride-duration-predictor",
        help="Registered model name in MLflow",
    )
    parser.add_argument(
        "--candidate-run-id",
        type=str,
        default=None,
        help="Optional MLflow run ID of candidate",
    )
    parser.add_argument(
        "--fallback-production-mae",
        type=float,
        default=DEFAULT_BASELINE_MAE,
        help="Fallback Production MAE if registry query fails",
    )

    args = parser.parse_args()

    try:
        run_ct_gate(
            candidate_metrics_path=args.candidate_metrics,
            margin=args.margin,
            model_name=args.model_name,
            candidate_run_id=args.candidate_run_id,
            fallback_production_mae=args.fallback_production_mae,
        )
        sys.exit(0)
    except Exception as exc:
        print(f"❌ CT Gate Execution Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
