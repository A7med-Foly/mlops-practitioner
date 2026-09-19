"""Model registry management and automated promotion lifecycle for prodml.

Provides programmatic utilities and CLI commands for:
- Registering candidate models into the MLflow Model Registry.
- Transitioning models across lifecycle stages (None -> Staging -> Production -> Archived).
- Automated promotion gating: `promote_if_better(candidate_run_id, metric="mae")`.
"""

import argparse
import logging
import os
import sys
from typing import Any

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.tracking import MlflowClient

from prodml.config import get_settings

logger = logging.getLogger("prodml.registry")

# Metric direction mapping: True if smaller is better, False if larger is better
LOWER_IS_BETTER_METRICS: dict[str, bool] = {
    "mae": True,
    "rmse": True,
    "mse": True,
    "loss": True,
    "val_loss": True,
    "train_loss": True,
    "r2": False,
    "r2_score": False,
    "accuracy": False,
    "precision": False,
    "recall": False,
    "f1": False,
}


def get_client(tracking_uri: str | None = None) -> MlflowClient:
    """Instantiate and return an MlflowClient with configured tracking URI and storage credentials."""
    settings = get_settings()
    uri = tracking_uri or settings.mlflow_tracking_uri

    os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.aws_access_key_id)
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", settings.aws_secret_access_key)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", settings.mlflow_s3_endpoint_url)
    os.environ.setdefault("AWS_DEFAULT_REGION", settings.aws_default_region)
    os.environ.setdefault("MLFLOW_TRACKING_URI", uri)

    mlflow.set_tracking_uri(uri)
    return MlflowClient(tracking_uri=uri)


def get_production_model_version(
    model_name: str = "ride-duration-predictor",
    client: MlflowClient | None = None,
) -> ModelVersion | None:
    """Retrieve the currently active Production model version, or None if none exists."""
    client = client or get_client()
    try:
        versions = client.search_model_versions(f"name = '{model_name}'")
        for version in versions:
            if getattr(version, "current_stage", "").lower() == "production":
                return version
        return None
    except Exception as exc:
        logger.warning("Could not query model registry for '%s': %s", model_name, exc)
        return None


def get_logged_model_uri_for_run(run_id: str, client: MlflowClient) -> str:
    """Find the logged model URI for a given run ID, supporting both MLflow 3.x and 2.x."""
    try:
        run = client.get_run(run_id)
        exp_id = run.info.experiment_id
        if hasattr(client, "search_logged_models"):
            logged = client.search_logged_models(experiment_ids=[exp_id])
            for lm in logged:
                if getattr(lm, "source_run_id", None) == run_id:
                    return lm.model_uri
    except Exception as exc:
        logger.debug("search_logged_models lookup error: %s", exc)

    return f"runs:/{run_id}/model"


def register_candidate_model(
    run_id: str,
    model_name: str = "ride-duration-predictor",
    artifact_path: str = "model",
    client: MlflowClient | None = None,
) -> ModelVersion:
    """Register a logged model artifact from a specific run into the MLflow Model Registry."""
    client = client or get_client()
    source_uri = get_logged_model_uri_for_run(run_id, client)
    logger.info("Registering model from '%s' as '%s'", source_uri, model_name)

    # Ensure registered model exists
    try:
        client.create_registered_model(model_name)
        logger.info("Created new registered model '%s'", model_name)
    except Exception:
        # Model already exists
        pass

    try:
        model_version = client.create_model_version(
            name=model_name,
            source=source_uri,
            run_id=run_id,
            description=f"Model registered from run {run_id}",
        )
    except Exception as err:
        logger.warning(
            "client.create_model_version failed with %s; attempting mlflow.register_model",
            err,
        )
        model_version = mlflow.register_model(source_uri, model_name)

    logger.info(
        "Successfully registered '%s' version %s (Stage: %s)",
        model_name,
        model_version.version,
        model_version.current_stage,
    )
    return model_version


def transition_stage(
    model_name: str,
    version: str | int,
    stage: str,
    archive_existing: bool = True,
    client: MlflowClient | None = None,
) -> ModelVersion:
    """Transition a registered model version to a new lifecycle stage.

    Args:
        model_name: Name of registered model.
        version: Model version number.
        stage: Target stage ('Staging', 'Production', 'Archived', 'None').
        archive_existing: If True and target is 'Production', existing versions are archived.
        client: Optional MlflowClient instance.

    Returns:
        Updated ModelVersion entity.
    """
    client = client or get_client()
    target_stage = stage.capitalize() if stage.lower() != "none" else "None"
    logger.info(
        "Transitioning '%s' version %s -> %s (archive_existing=%s)",
        model_name,
        version,
        target_stage,
        archive_existing,
    )

    updated = client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=target_stage,
        archive_existing_versions=archive_existing,
    )

    # In modern MLflow, also set alias champion when promoting to production
    if target_stage == "Production":
        try:
            client.set_registered_model_alias(
                name=model_name, alias="champion", version=str(version)
            )
        except Exception:
            pass

    return updated


def is_metric_better(
    metric_name: str, candidate_val: float, production_val: float
) -> bool:
    """Determine whether candidate_val is strictly superior to production_val."""
    normalized_name = metric_name.lower().replace("-", "_")
    lower_is_better = LOWER_IS_BETTER_METRICS.get(normalized_name, True)

    if lower_is_better:
        return candidate_val < production_val
    else:
        return candidate_val > production_val


def promote_if_better(
    candidate_run_id: str,
    model_name: str = "ride-duration-predictor",
    metric: str = "mae",
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Automated continuous delivery gate: compare candidate vs Production and promote if better.

    Args:
        candidate_run_id: MLflow Run ID of the newly trained candidate model.
        model_name: Registered model name.
        metric: Performance metric to evaluate (e.g. 'mae', 'rmse', 'r2').
        client: Optional MlflowClient instance.

    Returns:
        Dict with status, metrics, decision reason, and promoted version.
    """
    client = client or get_client()

    # 1. Fetch Candidate Run and Metric
    try:
        candidate_run = client.get_run(candidate_run_id)
    except Exception as err:
        raise ValueError(
            f"Candidate run '{candidate_run_id}' not found: {err}"
        ) from err

    candidate_metrics = candidate_run.data.metrics
    if metric not in candidate_metrics:
        raise ValueError(
            f"Metric '{metric}' not found in candidate run '{candidate_run_id}'. "
            f"Available metrics: {list(candidate_metrics.keys())}"
        )
    candidate_metric_val = float(candidate_metrics[metric])

    # 2. Check Current Production Model
    prod_version = get_production_model_version(model_name, client)

    # If no existing Production model, promote candidate immediately
    if prod_version is None:
        logger.info(
            "No active Production model found for '%s'. Promoting candidate run %s as initial baseline.",
            model_name,
            candidate_run_id,
        )
        new_version = register_candidate_model(
            candidate_run_id, model_name=model_name, client=client
        )
        promoted = transition_stage(
            model_name,
            new_version.version,
            stage="Production",
            archive_existing=True,
            client=client,
        )
        return {
            "promoted": True,
            "reason": f"No existing Production model; candidate {candidate_run_id} promoted as baseline.",
            "candidate_run_id": candidate_run_id,
            "candidate_version": promoted.version,
            "previous_version": None,
            "metric": metric,
            "candidate_metric": candidate_metric_val,
            "production_metric": None,
        }

    # 3. Retrieve Current Production Run Metric
    prod_run_id = prod_version.run_id
    try:
        prod_run = client.get_run(prod_run_id)
        prod_metric_val = float(prod_run.data.metrics.get(metric, float("inf")))
    except Exception as err:
        logger.warning(
            "Could not fetch production run %s metrics: %s. Assuming inf.",
            prod_run_id,
            err,
        )
        prod_metric_val = float("inf")

    # 4. Compare Performance
    better = is_metric_better(metric, candidate_metric_val, prod_metric_val)

    if better:
        logger.info(
            "Promotion Gate PASSED: Candidate %s (%s=%.4f) beat Production v%s (%s=%.4f)",
            candidate_run_id,
            metric,
            candidate_metric_val,
            prod_version.version,
            metric,
            prod_metric_val,
        )
        new_version = register_candidate_model(
            candidate_run_id, model_name=model_name, client=client
        )
        promoted = transition_stage(
            model_name,
            new_version.version,
            stage="Production",
            archive_existing=True,
            client=client,
        )
        return {
            "promoted": True,
            "reason": (
                f"Candidate {candidate_run_id} ({metric}={candidate_metric_val:.4f}) "
                f"outperformed Production v{prod_version.version} ({metric}={prod_metric_val:.4f})."
            ),
            "candidate_run_id": candidate_run_id,
            "candidate_version": promoted.version,
            "previous_version": prod_version.version,
            "metric": metric,
            "candidate_metric": candidate_metric_val,
            "production_metric": prod_metric_val,
        }
    else:
        logger.info(
            "Promotion Gate REJECTED: Candidate %s (%s=%.4f) did not beat Production v%s (%s=%.4f)",
            candidate_run_id,
            metric,
            candidate_metric_val,
            prod_version.version,
            metric,
            prod_metric_val,
        )
        return {
            "promoted": False,
            "reason": (
                f"Candidate {candidate_run_id} ({metric}={candidate_metric_val:.4f}) "
                f"did not beat Production v{prod_version.version} ({metric}={prod_metric_val:.4f})."
            ),
            "candidate_run_id": candidate_run_id,
            "candidate_version": None,
            "current_production_version": prod_version.version,
            "metric": metric,
            "candidate_metric": candidate_metric_val,
            "production_metric": prod_metric_val,
        }


def main() -> None:
    """CLI entrypoint for model registry operations."""
    parser = argparse.ArgumentParser(
        description="MLflow Model Registry Management and Continuous Delivery Gate"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status
    status_parser = subparsers.add_parser(
        "status", help="Check active Production model"
    )
    status_parser.add_argument(
        "--model-name", default="ride-duration-predictor", help="Model name"
    )

    # promote
    promote_parser = subparsers.add_parser(
        "promote", help="Evaluate and promote candidate run if better"
    )
    promote_parser.add_argument(
        "--run-id", required=True, help="Candidate MLflow run ID"
    )
    promote_parser.add_argument(
        "--model-name", default="ride-duration-predictor", help="Model name"
    )
    promote_parser.add_argument(
        "--metric", default="mae", help="Comparison metric (e.g. mae, rmse, r2)"
    )

    # transition
    trans_parser = subparsers.add_parser(
        "transition", help="Manually transition a model version stage"
    )
    trans_parser.add_argument(
        "--model-name", default="ride-duration-predictor", help="Model name"
    )
    trans_parser.add_argument("--version", required=True, help="Model version")
    trans_parser.add_argument(
        "--stage",
        required=True,
        choices=["None", "Staging", "Production", "Archived"],
        help="Target lifecycle stage",
    )

    args = parser.parse_args()

    client = get_client()

    if args.command == "status":
        prod = get_production_model_version(args.model_name, client)
        if prod:
            print(
                f"Model: {args.model_name} | Production Version: {prod.version} | Run ID: {prod.run_id}"
            )
        else:
            print(f"Model: {args.model_name} | No model currently in Production stage.")

    elif args.command == "promote":
        result = promote_if_better(
            candidate_run_id=args.run_id,
            model_name=args.model_name,
            metric=args.metric,
            client=client,
        )
        print(f"Result: {result}")
        if not result["promoted"]:
            sys.exit(1)

    elif args.command == "transition":
        updated = transition_stage(
            model_name=args.model_name,
            version=args.version,
            stage=args.stage,
            archive_existing=True,
            client=client,
        )
        print(
            f"Model: {args.model_name} v{updated.version} -> Stage: {updated.current_stage}"
        )


if __name__ == "__main__":
    main()
