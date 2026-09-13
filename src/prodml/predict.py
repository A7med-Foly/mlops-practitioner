"""Model inference and prediction interface for prodml.

Provides DurationPredictor class for single and batch predictions,
featuring execution timing instrumentation, structured logging, and flexible artifact loading.
"""

import logging
from pathlib import Path
import pickle
import time
from typing import Any

import pandas as pd

from prodml.config import get_settings
from prodml.decorators import timed
from prodml.features import prepare_features

logger = logging.getLogger("prodml.predict")


class DurationPredictor:
    """Interface for NYC green taxi trip duration predictions.

    Acts as the seam decoupling model serving from training and feature serialization details.
    """

    def __init__(self, model: Any) -> None:
        """Initialize predictor with a fitted model or scikit-learn Pipeline.

        Args:
            model: Fitted Pipeline or model capable of scoring feature dictionaries.
        """
        self.model = model

    @classmethod
    def load(cls, model_path: Path | str | None = None) -> "DurationPredictor":
        """Load a persisted model artifact from disk.

        Args:
            model_path: Path to serialized artifact. If None, loaded from Settings.model_path.

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

        try:
            with open(resolved_path, "rb") as f:
                artifact = pickle.load(f)
        except Exception as err:
            logger.error("Model load failure from %s: %s", resolved_path.resolve(), err)
            raise

        # Support both a direct Pipeline/model or a dictionary container
        if isinstance(artifact, dict) and "pipeline" in artifact:
            model = artifact["pipeline"]
        elif isinstance(artifact, dict) and "model" in artifact:
            model = artifact["model"]
        else:
            model = artifact

        return cls(model=model)

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
        prediction = self.model.predict(prepared)
        latency_ms = (time.perf_counter() - start_time) * 1000
        result = float(prediction[0])

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
        predictions = self.model.predict(prepared)
        return [float(val) for val in predictions]
