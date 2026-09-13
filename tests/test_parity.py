"""Parity and benchmark tests: Pickle vs ONNX runtime."""

import pickle
import time
import unittest

import numpy as np
import onnxruntime as ort

from prodml.config import get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.export import export_baseline
from prodml.features import prepare_features


class TestOnnxParity(unittest.TestCase):
    """Verify numerical parity between Scikit-Learn (Pickle) and ONNX Runtime."""

    @classmethod
    def setUpClass(cls):
        settings = get_settings()
        cls.pkl_path = settings.model_path
        cls.onnx_path = cls.pkl_path.with_suffix(".onnx")

        # Ensure ONNX model exists
        if not cls.onnx_path.exists():
            export_baseline(cls.pkl_path, cls.onnx_path)

        # Load Pickle pipeline
        with open(cls.pkl_path, "rb") as f:
            cls.pipeline = pickle.load(f)

        cls.model = cls.pipeline.named_steps["model"]
        cls.dv = cls.pipeline.named_steps["vectorizer"]

        # Load ONNX session
        cls.session = ort.InferenceSession(str(cls.onnx_path))
        cls.input_name = cls.session.get_inputs()[0].name
        cls.output_name = cls.session.get_outputs()[0].name

        # Load dataset & extract 500 validation rows
        raw_df = load_data(settings.data_path)
        cleaned_df = clean_data(raw_df)
        _, X_val_df, _, _, _, _ = split_data(cleaned_df)

        cls.val_500_df = X_val_df.iloc[:500]
        cls.prepared = prepare_features(cls.val_500_df)
        cls.X_matrix = cls.dv.transform(cls.prepared).toarray().astype(np.float32)

    def test_numerical_parity_500_rows(self):
        """Run 500 validation rows through both models and assert numerical equivalence."""
        # 1. Prediction via Pickle model
        pred_pkl = self.pipeline.predict(self.prepared)

        # 2. Prediction via ONNX Runtime
        pred_onnx = self.session.run(
            [self.output_name],
            {self.input_name: self.X_matrix},
        )[0].ravel()

        # 3. Assert parity within atol=1e-4
        max_diff = float(np.max(np.abs(pred_pkl - pred_onnx)))
        parity = np.allclose(pred_pkl, pred_onnx, atol=1e-4)

        self.assertTrue(
            parity,
            f"Parity check failed: max absolute difference is {max_diff:.8f} > 1e-4",
        )

    def test_benchmark_latency(self):
        """Measure mean and p95 latency for both Pickle and ONNX on 500 single-row requests."""
        # Warmup
        for _ in range(5):
            _ = self.model.predict(self.X_matrix[:5])
            _ = self.session.run(
                [self.output_name], {self.input_name: self.X_matrix[:5]}
            )

        pkl_latencies = []
        for i in range(len(self.X_matrix)):
            row = self.X_matrix[i : i + 1]
            t0 = time.perf_counter()
            _ = self.model.predict(row)
            pkl_latencies.append((time.perf_counter() - t0) * 1000)

        onnx_latencies = []
        for i in range(len(self.X_matrix)):
            row = self.X_matrix[i : i + 1]
            t0 = time.perf_counter()
            _ = self.session.run([self.output_name], {self.input_name: row})
            onnx_latencies.append((time.perf_counter() - t0) * 1000)

        pkl_mean = float(np.mean(pkl_latencies))
        pkl_p95 = float(np.percentile(pkl_latencies, 95))
        onnx_mean = float(np.mean(onnx_latencies))
        onnx_p95 = float(np.percentile(onnx_latencies, 95))

        self.assertGreater(pkl_mean, 0.0)
        self.assertGreater(pkl_p95, 0.0)
        self.assertGreater(onnx_mean, 0.0)
        self.assertGreater(onnx_p95, 0.0)
        # ONNX single-row latency is expected to be substantially faster than Python scikit-learn
        self.assertLess(onnx_mean, pkl_mean)


if __name__ == "__main__":
    unittest.main()
