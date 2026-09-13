"""Parity assertion test for Pickle and ONNX serialization formats."""

import pickle
import numpy as np
import onnxruntime as ort

from prodml.config import get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.export import export_baseline
from prodml.features import prepare_features


def test_pickle_onnx_numerical_parity():
    """Assert numerical parity between Pickle and ONNX models on 500 validation rows."""
    settings = get_settings()
    pkl_path = settings.model_path
    onnx_path = pkl_path.with_suffix(".onnx")

    # Ensure ONNX model exists
    if not onnx_path.exists():
        export_baseline(pkl_path, onnx_path)

    # 1. Load Pickle pipeline
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    dv = pipeline.named_steps["vectorizer"]

    # 2. Load ONNX Session
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 3. Load 500 validation rows
    raw_df = load_data(settings.data_path)
    cleaned_df = clean_data(raw_df)
    _, X_val_df, _, _, _, _ = split_data(cleaned_df)

    val_500 = X_val_df.iloc[:500]
    prepared = prepare_features(val_500)
    X_matrix = dv.transform(prepared).toarray().astype(np.float32)

    # 4. Predict with both engines
    pred_pkl = pipeline.predict(prepared)
    pred_onnx = session.run([output_name], {input_name: X_matrix})[0].ravel()

    # 5. Assert parity
    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)
