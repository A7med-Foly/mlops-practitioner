"""Tests for DVC pipeline stages and lineage tracking."""

from pathlib import Path

import pandas as pd
import yaml

from prodml.pipeline.evaluate import run_evaluate
from prodml.pipeline.featurize import run_featurize
from prodml.pipeline.lineage import compute_md5, get_dvc_hash
from prodml.pipeline.prepare import run_prepare
from prodml.pipeline.train import get_git_commit, run_train


def test_compute_md5_file(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("hello dvc")
    h1 = compute_md5(test_file)
    assert isinstance(h1, str)
    assert len(h1) == 32


def test_compute_md5_dir(tmp_path: Path):
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "a.txt").write_text("aaa")
    (sub / "b.txt").write_text("bbb")
    h = compute_md5(sub)
    assert h.endswith(".dir")


def test_get_dvc_hash_with_dvc_file(tmp_path: Path):
    target = tmp_path / "data.parquet"
    target.touch()
    dvc_file = tmp_path / "data.parquet.dvc"
    dvc_file.write_text(
        "outs:\n- md5: 1234567890abcdef1234567890abcdef\n  path: data.parquet\n"
    )

    h = get_dvc_hash("data.parquet", repo_root=tmp_path)
    assert h == "1234567890abcdef1234567890abcdef"


def test_get_dvc_hash_with_lock_file(tmp_path: Path):
    target = tmp_path / "data/processed/train.parquet"
    target.parent.mkdir(parents=True)
    target.touch()

    lock_file = tmp_path / "dvc.lock"
    lock_content = {
        "stages": {
            "prepare": {
                "outs": [
                    {
                        "path": "data/processed/train.parquet",
                        "md5": "deadbeefcafebabe0123456789abcdef",
                    }
                ]
            }
        }
    }
    with open(lock_file, "w") as f:
        yaml.dump(lock_content, f)

    h = get_dvc_hash("data/processed/train.parquet", repo_root=tmp_path)
    assert h == "deadbeefcafebabe0123456789abcdef"


def test_get_git_commit():
    commit = get_git_commit()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_end_to_end_pipeline_flow(tmp_path: Path):
    # 1. Setup raw mock data
    raw_path = tmp_path / "raw.parquet"
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": ["2026-05-01 10:00:00", "2026-05-01 11:00:00"] * 30,
            "lpep_dropoff_datetime": ["2026-05-01 10:15:00", "2026-05-01 11:20:00"]
            * 30,
            "trip_distance": [2.5, 4.0] * 30,
            "fare_amount": [12.0, 18.0] * 30,
            "total_amount": [15.0, 22.0] * 30,
            "PULocationID": [10, 20] * 30,
            "DOLocationID": [15, 25] * 30,
            "RatecodeID": [1, 1] * 30,
            "payment_type": [1, 2] * 30,
            "trip_type": [1, 1] * 30,
            "congestion_surcharge": [0.0, 0.0] * 30,
            "store_and_fwd_flag": ["N", "N"] * 30,
            "passenger_count": [1, 1] * 30,
        }
    )
    df.to_parquet(raw_path)

    params_file = tmp_path / "params.yaml"
    params_data = {
        "prepare": {
            "min_duration": 0.0,
            "max_duration_quantile": 1.0,
            "min_distance": 0.0,
            "test_size": 0.2,
            "val_size": 0.25,
            "random_state": 42,
            "target_column": "trip_duration",
            "pickup_column": "lpep_pickup_datetime",
            "dropoff_column": "lpep_dropoff_datetime",
        },
        "featurize": {
            "numerical_features": ["trip_distance", "fare_amount", "total_amount"],
            "categorical_features": ["PULocationID", "DOLocationID"],
        },
        "train": {
            "model_family": "xgboost",
            "n_estimators": 5,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_state": 42,
        },
        "evaluate": {"metrics": ["mae", "rmse", "r2"]},
    }
    with open(params_file, "w") as f:
        yaml.dump(params_data, f)

    processed_dir = tmp_path / "processed"
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    plots_dir = tmp_path / "plots"
    metrics_file = tmp_path / "metrics.json"

    # Step 1: Prepare
    run_prepare(input_path=raw_path, output_dir=processed_dir, params_path=params_file)
    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "val.parquet").exists()
    assert (processed_dir / "test.parquet").exists()

    # Step 2: Featurize
    run_featurize(
        train_path=processed_dir / "train.parquet",
        val_path=processed_dir / "val.parquet",
        output_dir=features_dir,
        params_path=params_file,
    )
    assert (features_dir / "train.npz").exists()
    assert (features_dir / "val.npz").exists()
    assert (features_dir / "dv.pkl").exists()

    # Step 3: Train
    run_train(features_dir=features_dir, models_dir=models_dir, params_path=params_file)
    assert (models_dir / "dvc_model.joblib").exists()

    # Step 4: Evaluate
    metrics = run_evaluate(
        model_path=models_dir / "dvc_model.joblib",
        features_path=features_dir / "val.npz",
        metrics_path=metrics_file,
        plots_dir=plots_dir,
        params_path=params_file,
    )
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics_file.exists()
    assert (plots_dir / "residuals.png").exists()
