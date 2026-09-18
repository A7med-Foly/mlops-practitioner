"""Model inference and prediction interface for prodml.

Provides DurationPredictor class for single and batch predictions,
featuring execution timing instrumentation, structured logging, and flexible artifact loading.
"""

import logging
from pathlib import Path
import pickle
import time
from typing import Any

import onnxruntime as ort
import pandas as pd

from prodml.config import get_settings
from prodml.decorators import timed
from prodml.features import (
    ORDERED_FEATURE_NAMES,
    features_to_matrix,
    prepare_features,
)

logger = logging.getLogger("prodml.predict")


class DurationPredictor:
    """Interface for NYC green taxi trip duration predictions.

    Acts as the seam decoupling model serving from training and feature serialization details.
    Supports both ONNX Runtime (sub-millisecond online serving) and scikit-learn Pipelines.
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize predictor with a fitted model, Pipeline, or ONNX InferenceSession.

        Args:
            model: Fitted Pipeline or ONNX InferenceSession.
            feature_names: Optional explicit list of feature names in column order.
        """
        self.model = model
        self._is_onnx = isinstance(model, ort.InferenceSession) or (
            hasattr(model, "run") and hasattr(model, "get_inputs")
        )

        if self._is_onnx:
            self.input_name: str | None = self.model.get_inputs()[0].name
            self.output_name: str | None = self.model.get_outputs()[0].name
            self.feature_names = (
                list(feature_names)
                if feature_names is not None
                else list(ORDERED_FEATURE_NAMES)
            )
        else:
            self.input_name = None
            self.output_name = None
            if (
                feature_names is None
                and hasattr(model, "named_steps")
                and "vectorizer" in model.named_steps
            ):
                dv = model.named_steps["vectorizer"]
                if hasattr(dv, "get_feature_names_out"):
                    self.feature_names = list(dv.get_feature_names_out())
                else:
                    self.feature_names = list(ORDERED_FEATURE_NAMES)
            else:
                self.feature_names = (
                    list(feature_names)
                    if feature_names is not None
                    else list(ORDERED_FEATURE_NAMES)
                )

    @property
    def is_onnx(self) -> bool:
        """Whether the underlying engine is ONNX Runtime."""
        return self._is_onnx

    @property
    def is_ready(self) -> bool:
        """Whether the model/session is loaded in memory."""
        return self.model is not None

    @classmethod
    def load(
        cls,
        model_path: Path | str | None = None,
        feature_names: list[str] | None = None,
    ) -> "DurationPredictor":
        """Load a persisted model artifact (.onnx or .pkl) from disk.

        Args:
            model_path: Path to serialized artifact. If None, loaded from Settings.model_path.
            feature_names: Optional feature names order.

        Returns:
            Instantiated DurationPredictor.
        """
        if model_path is None:
            settings = get_settings()
            model_path = settings.model_path

        resolved_path = Path(model_path)
        if not resolved_path.exists():
            msg = f"Model artifact not found at {resolved_path.resolve()}"
            logger.error("Model load failure: %s", msg)
            raise FileNotFoundError(msg)

        suffix = resolved_path.suffix.lower()
        try:
            if suffix == ".onnx":
                logger.info(
                    "Loading ONNX model session from %s", resolved_path.resolve()
                )
                session = ort.InferenceSession(str(resolved_path))
                return cls(model=session, feature_names=feature_names)
            else:
                logger.info("Loading Pickle model from %s", resolved_path.resolve())
                with open(resolved_path, "rb") as f:
                    artifact = pickle.load(f)

                # Support both a direct Pipeline/model or a dictionary container
                if isinstance(artifact, dict) and "pipeline" in artifact:
                    model = artifact["pipeline"]
                elif isinstance(artifact, dict) and "model" in artifact:
                    model = artifact["model"]
                else:
                    model = artifact

                return cls(model=model, feature_names=feature_names)
        except Exception as err:
            logger.error("Model load failure from %s: %s", resolved_path.resolve(), err)
            raise

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Predict trip duration in minutes for a single feature record.

        Args:
            features: Dictionary containing trip features (e.g. trip_distance, fare_amount).

        Returns:
            Predicted duration in minutes as a float.
        """
        if not isinstance(features, dict):
            msg = f"Expected dictionary of features, got {type(features).__name__}"
            logger.error("Validation rejection: %s", msg)
            raise ValueError(msg)

        # WARNING: input outside the training range (trip_distance > 100)
        try:
            trip_distance = float(features.get("trip_distance", 0.0))
            if trip_distance > 100:
                logger.warning(
                    "Input outside the training range: trip_distance=%.2f > 100",
                    trip_distance,
                )
        except (ValueError, TypeError) as err:
            msg = f"Invalid trip_distance format: {err}"
            logger.error("Validation rejection: %s", msg)
            raise ValueError(msg) from err

        prepared = prepare_features(features)

        # DEBUG: feature vector
        logger.debug("Feature vector: %s", prepared[0] if prepared else {})

        start_time = time.perf_counter()
        if self._is_onnx:
            matrix = features_to_matrix(prepared, self.feature_names)
            outputs = self.model.run([self.output_name], {self.input_name: matrix})
            result = float(outputs[0].ravel()[0])
        else:
            prediction = self.model.predict(prepared)
            result = float(prediction[0])
        latency_ms = (time.perf_counter() - start_time) * 1000

        # INFO: prediction served with latency
        logger.info(
            "Prediction served: %.2f minutes with latency %.2f ms",
            result,
            latency_ms,
        )

        return result

    def predict(self, features: dict[str, Any]) -> float:
        """Alias for predict_one providing explicit type signature.

        Args:
            features: Dictionary containing trip features.

        Returns:
            Predicted duration in minutes as a float.
        """
        return self.predict_one(features)

    def predict_batch(
        self, features: list[dict[str, Any]] | pd.DataFrame
    ) -> list[float]:
        """Predict trip duration in minutes for a batch of feature records.

        Args:
            features: List of feature dictionaries or a pandas DataFrame.

        Returns:
            List of predicted durations in minutes.
        """
        prepared = prepare_features(features)
        if self._is_onnx:
            matrix = features_to_matrix(prepared, self.feature_names)
            outputs = self.model.run([self.output_name], {self.input_name: matrix})
            predictions = outputs[0].ravel()
            return [float(val) for val in predictions]
        else:
            predictions = self.model.predict(prepared)
            return [float(val) for val in predictions]
