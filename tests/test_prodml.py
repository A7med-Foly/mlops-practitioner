"""Unit and integration test suite for prodml package."""

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from prodml.config import Settings
from prodml.data import clean_data, compute_trip_duration, split_data
from prodml.decorators import timed
from prodml.features import (
    dict_to_feature_dict,
    encode_store_and_fwd_flag,
    prepare_features,
)
from prodml.predict import DurationPredictor
from prodml.train import create_model, evaluate_pipeline, save_artifact, train_pipeline


class TestConfig(unittest.TestCase):
    """Test configuration behavior and environment overrides."""

    def test_default_settings(self):
        settings = Settings()
        self.assertEqual(settings.server_port, 8000)
        self.assertEqual(settings.model_type, "random_forest")
        self.assertEqual(settings.model_path, settings.models_dir / settings.model_name)
        self.assertFalse(settings.data_path.is_absolute())

    def test_env_override(self):
        os.environ["PRODML_SERVER_PORT"] = "9090"
        os.environ["PRODML_N_ESTIMATORS"] = "50"
        try:
            settings = Settings()
            self.assertEqual(settings.server_port, 9090)
            self.assertEqual(settings.n_estimators, 50)
        finally:
            del os.environ["PRODML_SERVER_PORT"]
            del os.environ["PRODML_N_ESTIMATORS"]


class TestDecorators(unittest.TestCase):
    """Test custom decorators."""

    def test_timed_decorator(self):
        with self.assertLogs("prodml.timed", level="INFO") as cm:

            @timed
            def sample_func(x: int, y: int) -> int:
                return x + y

            result = sample_func(3, 7)
            self.assertEqual(result, 10)
            self.assertTrue(any("sample_func executed in" in log for log in cm.output))


class TestData(unittest.TestCase):
    """Test data loading, cleaning, duration computation, and splitting."""

    def setUp(self):
        # Create a synthetic DataFrame resembling NYC green taxi data
        self.raw_df = pd.DataFrame(
            {
                "lpep_pickup_datetime": pd.date_range(
                    "2026-05-01 10:00:00", periods=10, freq="15min"
                ),
                "lpep_dropoff_datetime": pd.date_range(
                    "2026-05-01 10:20:00", periods=10, freq="15min"
                ),
                "store_and_fwd_flag": [
                    "N",
                    "Y",
                    "N",
                    "N",
                    "Y",
                    "N",
                    "Y",
                    "N",
                    "N",
                    "Y",
                ],
                "RatecodeID": [1.0] * 10,
                "passenger_count": [1.0] * 10,
                "payment_type": [1.0] * 10,
                "trip_type": [1.0] * 10,
                "congestion_surcharge": [0.0] * 10,
                "trip_distance": [
                    2.5,
                    3.0,
                    1.2,
                    0.0,
                    4.5,
                    2.0,
                    3.5,
                    1.8,
                    5.0,
                    2.2,
                ],  # row 3 has distance 0.0
                "fare_amount": [12.0] * 10,
                "tip_amount": [2.0] * 10,
                "tolls_amount": [0.0] * 10,
                "improvement_surcharge": [1.0] * 10,
                "total_amount": [15.0] * 10,
                "cbd_congestion_fee": [0.75] * 10,
                "ehail_fee": [np.nan] * 10,
            }
        )

    def test_compute_trip_duration(self):
        duration = compute_trip_duration(self.raw_df)
        self.assertEqual(len(duration), 10)
        self.assertTrue((duration == 20.0).all())

    def test_clean_data(self):
        cleaned = clean_data(self.raw_df)
        # ehail_fee dropped
        self.assertNotIn("ehail_fee", cleaned.columns)
        # trip_duration computed
        self.assertIn("trip_duration", cleaned.columns)
        # distance == 0 filtered out (1 row excluded)
        self.assertEqual(len(cleaned), 9)

    def test_split_data(self):
        # Create 20 samples to test 60/20/20 split
        extended_df = pd.concat([self.raw_df] * 2, ignore_index=True)
        extended_df = clean_data(extended_df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            extended_df, test_size=0.2, val_size=0.25, random_state=42
        )
        total = len(extended_df)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), total)
        self.assertEqual(len(y_train) + len(y_val) + len(y_test), total)
        self.assertAlmostEqual(len(X_test) / total, 0.2, delta=0.1)


class TestFeatures(unittest.TestCase):
    """Test feature encoding and dictionary conversion."""

    def test_encode_store_and_fwd_flag(self):
        self.assertEqual(encode_store_and_fwd_flag("Y"), 1)
        self.assertEqual(encode_store_and_fwd_flag("N"), 0)
        self.assertEqual(encode_store_and_fwd_flag("y"), 1)
        self.assertEqual(encode_store_and_fwd_flag(1), 1)
        self.assertEqual(encode_store_and_fwd_flag(0), 0)

    def test_dict_to_feature_dict(self):
        raw_dict = {
            "store_and_fwd_flag": "Y",
            "trip_distance": "3.5",
            "fare_amount": 14.5,
            "extra_field_not_used": 999,
        }
        res = dict_to_feature_dict(raw_dict)
        self.assertEqual(res["store_and_fwd_flag_encoded"], 1)
        self.assertEqual(res["trip_distance"], 3.5)
        self.assertEqual(res["fare_amount"], 14.5)
        self.assertNotIn("extra_field_not_used", res)

    def test_prepare_features(self):
        sample = [{"trip_distance": 2.0, "store_and_fwd_flag": "N"}]
        prepared = prepare_features(sample)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["store_and_fwd_flag_encoded"], 0)


class TestTrainAndPredict(unittest.TestCase):
    """Test end-to-end model training, evaluation, persistence, and inference."""

    def test_training_and_inference_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.pkl"

            # 1. Prepare sample training data
            X_train = [
                {
                    "trip_distance": 1.0,
                    "fare_amount": 7.0,
                    "store_and_fwd_flag_encoded": 0,
                },
                {
                    "trip_distance": 2.0,
                    "fare_amount": 10.0,
                    "store_and_fwd_flag_encoded": 0,
                },
                {
                    "trip_distance": 3.0,
                    "fare_amount": 14.0,
                    "store_and_fwd_flag_encoded": 1,
                },
                {
                    "trip_distance": 4.0,
                    "fare_amount": 18.0,
                    "store_and_fwd_flag_encoded": 0,
                },
                {
                    "trip_distance": 5.0,
                    "fare_amount": 22.0,
                    "store_and_fwd_flag_encoded": 1,
                },
            ] * 4
            y_train = pd.Series([8.0, 12.0, 16.0, 20.0, 24.0] * 4)

            X_val = [
                {
                    "trip_distance": 2.5,
                    "fare_amount": 11.0,
                    "store_and_fwd_flag_encoded": 0,
                },
                {
                    "trip_distance": 3.5,
                    "fare_amount": 15.0,
                    "store_and_fwd_flag_encoded": 1,
                },
            ]
            y_val = pd.Series([13.0, 17.0])

            # 2. Fit pipeline
            model = create_model("random_forest", n_estimators=5, random_state=42)
            pipeline = train_pipeline(X_train, y_train, model=model)

            # 3. Evaluate
            metrics = evaluate_pipeline(pipeline, X_val, y_val)
            self.assertIn("mae", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("r2", metrics)

            # 4. Save artifact
            save_artifact(pipeline, model_path)
            self.assertTrue(model_path.exists())

            # 5. Load via DurationPredictor
            predictor = DurationPredictor.load(model_path)

            # 6. Single predict with @timed
            test_record = {
                "trip_distance": 2.5,
                "fare_amount": 11.0,
                "store_and_fwd_flag": "N",
            }
            pred_one = predictor.predict_one(test_record)
            self.assertIsInstance(pred_one, float)
            self.assertGreater(pred_one, 0.0)

            # Alias predict
            pred_alias = predictor.predict(test_record)
            self.assertEqual(pred_one, pred_alias)

            # 7. Batch predict
            batch_records = [
                {"trip_distance": 2.0, "fare_amount": 10.0},
                {"trip_distance": 4.0, "fare_amount": 18.0},
            ]
            batch_preds = predictor.predict_batch(batch_records)
            self.assertEqual(len(batch_preds), 2)
            for p in batch_preds:
                self.assertIsInstance(p, float)


if __name__ == "__main__":
    unittest.main()
