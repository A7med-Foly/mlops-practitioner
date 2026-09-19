"""Tests for prediction interface: return types, sane ranges, and determinism."""

from prodml.predict import DurationPredictor


def test_prediction_returns_float(
    trained_model: DurationPredictor, sample_features: dict
):
    """Prediction must return a Python float."""
    pred = trained_model.predict_one(sample_features)
    assert isinstance(pred, float)

    alias_pred = trained_model.predict(sample_features)
    assert isinstance(alias_pred, float)
    assert alias_pred == pred


def test_prediction_in_sane_range(
    trained_model: DurationPredictor, sample_features: dict
):
    """Predicted trip duration must fall within plausible bounds (0 to 300 minutes)."""
    pred = trained_model.predict_one(sample_features)
    assert 0.0 < pred < 300.0


def test_prediction_is_deterministic(
    trained_model: DurationPredictor, sample_features: dict
):
    """Calling prediction twice on identical feature input must return identical values."""
    pred_first = trained_model.predict_one(sample_features)
    pred_second = trained_model.predict_one(sample_features)
    assert pred_first == pred_second


def test_prediction_batch(trained_model: DurationPredictor, sample_features: dict):
    """Batch prediction must return a list of floats matching input record count."""
    batch_input = [sample_features, sample_features, sample_features]
    preds = trained_model.predict_batch(batch_input)

    assert isinstance(preds, list)
    assert len(preds) == 3
    for p in preds:
        assert isinstance(p, float)
        assert 0.0 < p < 300.0


def test_onnx_predictor_loading_and_inference(sample_features: dict):
    """DurationPredictor.load with .onnx model must instantiate an ONNX session and predict correctly."""
    from prodml.config import get_settings

    settings = get_settings()
    onnx_path = settings.models_dir / "baseline.onnx"
    assert onnx_path.exists()

    predictor = DurationPredictor.load(onnx_path)
    assert predictor.is_onnx is True
    assert predictor.is_ready is True
    assert predictor.input_name == "float_input"
    assert predictor.output_name == "variable"

    pred = predictor.predict_one(sample_features)
    assert isinstance(pred, float)
    assert 0.0 < pred < 300.0

    batch_preds = predictor.predict_batch([sample_features, sample_features])
    assert len(batch_preds) == 2
    for p in batch_preds:
        assert isinstance(p, float)
        assert 0.0 < p < 300.0


def test_pyfunc_predictor_mock_inference(sample_features: dict):
    """Test DurationPredictor with a mock PyFuncModel."""
    from unittest.mock import MagicMock
    import numpy as np

    mock_pyfunc = MagicMock()
    mock_pyfunc.predict.return_value = np.array([18.5])
    mock_pyfunc.metadata = MagicMock()

    predictor = DurationPredictor(
        model=mock_pyfunc, model_uri="models:/ride-duration-predictor/Production"
    )
    assert predictor.is_pyfunc is True
    assert predictor.is_ready is True
    assert predictor.model_uri == "models:/ride-duration-predictor/Production"

    res = predictor.predict_one(sample_features)
    assert res == 18.5

    mock_pyfunc.predict.return_value = np.array([18.5, 18.5])
    batch_res = predictor.predict_batch([sample_features, sample_features])
    assert len(batch_res) == 2
    assert batch_res == [18.5, 18.5]
