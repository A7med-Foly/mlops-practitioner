"""Model inference and prediction interface for prodml.

Provides DurationPredictor class for single and batch predictions,
featuring execution timing instrumentation and flexible artifact loading.
"""

from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from prodml.config import get_settings
from prodml.decorators import timed
from prodml.features import prepare_features


class DurationPredictor:
    """Interface for NYC green taxi trip duration predictions.

    Acts as the seam decoupling model serving (FastAPI, BentoML, ONNX)
    from training and feature serialization details.
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
            raise FileNotFoundError(
                f"Model artifact not found at {resolved_path.resolve()}"
            )

        with open(resolved_path, "rb") as f:
            artifact = pickle.load(f)

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
        prepared = prepare_features(features)
        prediction = self.model.predict(prepared)
        return float(prediction[0])

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
