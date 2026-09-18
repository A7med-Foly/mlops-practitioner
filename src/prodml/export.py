"""Model export and serialization module for prodml.

Handles converting scikit-learn models/pipelines to ONNX format
with dynamic axes on the batch dimension for cross-platform, high-performance serving.
"""

import logging
from pathlib import Path
import pickle
from typing import Any

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import get_settings

logger = logging.getLogger("prodml.export")


def export_model_to_onnx(
    model_or_pipeline: Any,
    output_path: Path | str,
    n_features: int = 9,
    input_name: str = "float_input",
    target_opset: int = 17,
) -> Path:
    """Convert a scikit-learn regressor or pipeline model to ONNX format.

    Configures dynamic axes on the batch dimension: [None, n_features] -> [None, 1].

    Args:
        model_or_pipeline: Trained scikit-learn model, or Pipeline containing ('model', estimator).
        output_path: Destination path for the .onnx file.
        n_features: Number of numerical input features (default 9).
        input_name: Name of the input tensor in the ONNX graph.
        target_opset: ONNX operator set version (default 17).

    Returns:
        Path to the written .onnx file.
    """
    # Extract the underlying estimator if a Pipeline is passed
    if (
        hasattr(model_or_pipeline, "named_steps")
        and "model" in model_or_pipeline.named_steps
    ):
        estimator = model_or_pipeline.named_steps["model"]
    elif isinstance(model_or_pipeline, dict) and "model" in model_or_pipeline:
        estimator = model_or_pipeline["model"]
    else:
        estimator = model_or_pipeline

    # Configure dynamic batch dimension [None, n_features]
    initial_types = [(input_name, FloatTensorType([None, n_features]))]

    logger.info(
        "Exporting %s to ONNX with dynamic batch dimension [None, %d]...",
        estimator.__class__.__name__,
        n_features,
    )

    onx_proto = convert_sklearn(
        estimator,
        initial_types=initial_types,
        target_opset=target_opset,
    )

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(onx_proto.SerializeToString())

    logger.info("Successfully exported ONNX model to: %s", dest.resolve())
    return dest


def export_baseline(
    model_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Load baseline pickle artifact and export it to an ONNX file.

    Args:
        model_path: Path to .pkl model artifact. Defaults to Settings.model_path.
        output_path: Path to output .onnx file. Defaults to models/baseline.onnx.

    Returns:
        Path to the exported .onnx artifact.
    """
    settings = get_settings()
    src_path = (
        Path(model_path) if model_path else (settings.models_dir / "baseline.pkl")
    )

    if output_path is None:
        dest_path = settings.models_dir / "baseline.onnx"
    else:
        dest_path = Path(output_path)

    if not src_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {src_path.resolve()}")

    with open(src_path, "rb") as f:
        artifact = pickle.load(f)

    # Determine number of features if vectorizer is present
    n_features = 9
    if hasattr(artifact, "named_steps") and "vectorizer" in artifact.named_steps:
        dv = artifact.named_steps["vectorizer"]
        if hasattr(dv, "get_feature_names_out"):
            n_features = len(dv.get_feature_names_out())

    return export_model_to_onnx(artifact, output_path=dest_path, n_features=n_features)
