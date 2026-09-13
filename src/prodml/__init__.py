"""prodml: Production Machine Learning package for trip duration prediction."""

from prodml.config import Settings, get_settings
from prodml.data import clean_data, compute_trip_duration, load_data, split_data
from prodml.decorators import timed
from prodml.export import export_baseline, export_model_to_onnx
from prodml.features import build_pipeline, encode_features, prepare_features
from prodml.logging_conf import (
    JSONFormatter,
    get_correlation_id,
    set_correlation_id,
    setup_logging,
)
from prodml.predict import DurationPredictor
from prodml.train import evaluate_pipeline, run_training, train_pipeline

__all__ = [
    "DurationPredictor",
    "JSONFormatter",
    "Settings",
    "build_pipeline",
    "clean_data",
    "compute_trip_duration",
    "encode_features",
    "evaluate_pipeline",
    "export_baseline",
    "export_model_to_onnx",
    "get_correlation_id",
    "get_settings",
    "load_data",
    "prepare_features",
    "run_training",
    "set_correlation_id",
    "setup_logging",
    "split_data",
    "timed",
    "train_pipeline",
]
