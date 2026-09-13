"""prodml: Production Machine Learning package for trip duration prediction."""

from prodml.config import Settings, get_settings
from prodml.data import clean_data, compute_trip_duration, load_data, split_data
from prodml.decorators import timed
from prodml.features import build_pipeline, encode_features, prepare_features
from prodml.predict import DurationPredictor
from prodml.train import evaluate_pipeline, run_training, train_pipeline

__all__ = [
    "DurationPredictor",
    "Settings",
    "build_pipeline",
    "clean_data",
    "compute_trip_duration",
    "encode_features",
    "evaluate_pipeline",
    "get_settings",
    "load_data",
    "prepare_features",
    "run_training",
    "split_data",
    "timed",
    "train_pipeline",
]
