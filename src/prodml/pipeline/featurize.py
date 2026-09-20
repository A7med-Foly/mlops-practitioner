"""Pipeline stage 2: Feature Engineering & Extraction.

Reads processed train/val partitions, encodes categorical features and interactions,
converts to numerical matrices, fits DictVectorizer, and outputs to data/features/.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer

from prodml.features import (
    ORDERED_FEATURE_NAMES,
    features_to_matrix,
    prepare_features,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("prodml.pipeline.featurize")


def run_featurize(
    train_path: str | Path = "data/processed/train.parquet",
    val_path: str | Path = "data/processed/val.parquet",
    output_dir: str | Path = "data/features",
    params_path: str | Path = "params.yaml",
) -> None:
    """Transform partitioned data into feature matrices and persist artifacts."""
    logger.info("Loading parameters from %s", params_path)
    with open(params_path, encoding="utf-8") as f:
        _params: dict[str, Any] = yaml.safe_load(f).get("featurize", {})

    train_file = Path(train_path)
    val_file = Path(val_path)
    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Processed datasets missing: {train_file}, {val_file}")

    logger.info("Reading processed partitions: %s and %s", train_file, val_file)
    train_df = pd.read_parquet(train_file)
    val_df = pd.read_parquet(val_file)

    target_col = "trip_duration"
    y_train = train_df[target_col].to_numpy(dtype=np.float32)
    y_val = val_df[target_col].to_numpy(dtype=np.float32)

    X_train_df = train_df.drop(columns=[target_col])
    X_val_df = val_df.drop(columns=[target_col])

    logger.info("Preparing feature dictionaries...")
    X_train_dicts = prepare_features(X_train_df)
    X_val_dicts = prepare_features(X_val_df)

    logger.info("Fitting DictVectorizer...")
    dv = DictVectorizer(sparse=False)
    dv.fit(X_train_dicts)

    logger.info("Converting features to ordered NumPy matrices...")
    X_train_mat = features_to_matrix(X_train_dicts, feature_names=ORDERED_FEATURE_NAMES)
    X_val_mat = features_to_matrix(X_val_dicts, feature_names=ORDERED_FEATURE_NAMES)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_npz = out_path / "train.npz"
    val_npz = out_path / "val.npz"
    dv_file = out_path / "dv.pkl"

    np.savez_compressed(train_npz, X=X_train_mat, y=y_train)
    np.savez_compressed(val_npz, X=X_val_mat, y=y_val)

    with open(dv_file, "wb") as f:
        pickle.dump(dv, f)

    logger.info(
        "Saved feature matrices: %s (shape %s) and %s (shape %s)",
        train_npz,
        X_train_mat.shape,
        val_npz,
        X_val_mat.shape,
    )
    logger.info("Saved vectorizer to %s", dv_file)


if __name__ == "__main__":
    run_featurize()
