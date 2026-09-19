"""Pipeline stage 4: Model Evaluation.

Loads trained model artifact and validation features, computes regression metrics
(MAE, RMSE, R2), generates evaluation plots, and writes metrics.json for DVC tracking.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from prodml.features import ORDERED_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.pipeline.evaluate")


def generate_residual_plot(
    y_true: np.ndarray, y_pred: np.ndarray, out_file: Path
) -> None:
    """Generate residuals scatter and histogram."""
    residuals = y_true - y_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_pred, residuals, alpha=0.3, s=12, color="#008080")
    ax1.axhline(0, color="red", linestyle="--", lw=1.5)
    ax1.set_xlabel("Predicted Duration (min)")
    ax1.set_ylabel("Residual (Actual - Predicted)")
    ax1.set_title("Residuals vs. Predictions")
    ax1.grid(True, alpha=0.3)

    ax2.hist(residuals, bins=50, color="#008080", edgecolor="black", alpha=0.7)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5)
    ax2.set_xlabel("Residual (min)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Residual Distribution")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def generate_feature_importance_plot(
    feature_names: list[str], importances: np.ndarray, out_file: Path
) -> None:
    """Generate feature importance horizontal bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    indices = np.argsort(importances)
    sorted_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    ax.barh(
        range(len(sorted_names)), sorted_importances, color="#2E86C1", align="center"
    )
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def run_evaluate(
    model_path: str | Path = "models/dvc_model.joblib",
    features_path: str | Path = "data/features/val.npz",
    metrics_path: str | Path = "metrics.json",
    plots_dir: str | Path = "plots",
    params_path: str | Path = "params.yaml",
) -> dict[str, float]:
    """Evaluate model and save metrics and diagnostic plots."""
    logger.info("Loading parameters from %s", params_path)
    with open(params_path, encoding="utf-8") as f:
        _params: dict[str, Any] = yaml.safe_load(f).get("evaluate", {})

    model_file = Path(model_path)
    val_file = Path(features_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation features not found: {val_file}")

    logger.info("Loading model from %s", model_file)
    model = joblib.load(model_file)

    logger.info("Loading validation features from %s", val_file)
    with np.load(val_file) as d:
        X_val, y_val = d["X"], d["y"]

    y_pred = model.predict(X_val)

    mae = float(mean_absolute_error(y_val, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    r2 = float(r2_score(y_val, y_pred))

    metrics = {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }

    logger.info("Computed Metrics: %s", metrics)

    out_metrics = Path(metrics_path)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote metrics to %s", out_metrics)

    out_plots = Path(plots_dir)
    out_plots.mkdir(parents=True, exist_ok=True)

    residual_file = out_plots / "residuals.png"
    generate_residual_plot(y_val, y_pred, residual_file)
    logger.info("Saved residual plot to %s", residual_file)

    if hasattr(model, "feature_importances_"):
        fi_file = out_plots / "feature_importance.png"
        generate_feature_importance_plot(
            ORDERED_FEATURE_NAMES, model.feature_importances_, fi_file
        )
        logger.info("Saved feature importance plot to %s", fi_file)

    return metrics


if __name__ == "__main__":
    run_evaluate()
