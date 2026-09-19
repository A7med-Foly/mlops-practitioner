"""Unit tests for prodml.train: metadata helpers, plots, and model training."""

from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from prodml.config import Settings
from prodml.train import (
    PyTorchMLP,
    compute_file_sha256,
    create_feature_importance_plot,
    create_residual_plot,
    dump_requirements_txt,
    get_git_commit_hash,
    train_linear_regression,
    train_pytorch_mlp,
    train_xgboost,
)


def test_git_commit_hash_returns_str():
    commit = get_git_commit_hash()
    assert isinstance(commit, str)
    assert len(commit) > 0


def test_compute_file_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("mlops practitioner")
    digest = compute_file_sha256(test_file)
    assert isinstance(digest, str)
    assert len(digest) == 64

    missing = compute_file_sha256(tmp_path / "non_existent.txt")
    assert missing == "artifact-not-found"


def test_dump_requirements_txt(tmp_path: Path):
    out = tmp_path / "requirements.txt"
    dump_requirements_txt(out)
    assert out.exists()
    content = out.read_text()
    assert "mlflow" in content
    assert "xgboost" in content
    assert "torch" in content


def test_residual_plot(tmp_path: Path):
    y_true = np.array([10.0, 15.0, 20.0], dtype=np.float32)
    y_pred = np.array([9.5, 15.2, 19.0], dtype=np.float32)
    plot_path = tmp_path / "residual_plot.png"
    create_residual_plot(y_true, y_pred, plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_feature_importance_plot(tmp_path: Path):
    names = ["f1", "f2", "f3"]
    importances = np.array([0.2, 0.5, 0.3])
    plot_path = tmp_path / "feature_importance.png"
    create_feature_importance_plot(names, importances, plot_path)
    assert plot_path.exists()
    assert plot_path.stat().st_size > 0


def test_pytorch_mlp_forward():
    model = PyTorchMLP(input_dim=9, hidden_dims=(16, 8))
    x = torch.randn(4, 9)
    out = model(x)
    assert out.shape == (4,)


@pytest.fixture
def synthetic_data():
    np.random.seed(42)
    X_train = np.random.randn(50, 9).astype(np.float32)
    y_train = np.random.uniform(5.0, 30.0, size=(50,)).astype(np.float32)
    X_val = np.random.randn(20, 9).astype(np.float32)
    y_val = np.random.uniform(5.0, 30.0, size=(20,)).astype(np.float32)
    return X_train, y_train, X_val, y_val


def test_train_linear_regression(synthetic_data, tmp_path: Path, monkeypatch):
    X_train, y_train, X_val, y_val = synthetic_data
    settings = Settings()
    # Use local file store for testing mlflow logging
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: None)

    metrics = train_linear_regression(
        X_train,
        y_train,
        X_val,
        y_val,
        settings,
        git_commit="test-commit",
        data_version="test-ver",
        data_hash="test-hash",
    )
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert metrics["mae"] > 0


def test_train_xgboost(synthetic_data, tmp_path: Path):
    X_train, y_train, X_val, y_val = synthetic_data
    settings = Settings()
    hyperparams = {
        "n_estimators": 5,
        "max_depth": 2,
        "learning_rate": 0.1,
        "random_state": 42,
    }
    metrics, model = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        settings,
        git_commit="test-commit",
        data_version="test-ver",
        data_hash="test-hash",
        hyperparams=hyperparams,
        enable_autolog=False,
    )
    assert "mae" in metrics
    assert model is not None


def test_train_pytorch_mlp(synthetic_data):
    X_train, y_train, X_val, y_val = synthetic_data
    settings = Settings()
    metrics = train_pytorch_mlp(
        X_train,
        y_train,
        X_val,
        y_val,
        settings,
        git_commit="test-commit",
        data_version="test-ver",
        data_hash="test-hash",
        epochs=2,
        batch_size=16,
    )
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
