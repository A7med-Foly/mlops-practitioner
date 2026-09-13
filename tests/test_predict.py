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
