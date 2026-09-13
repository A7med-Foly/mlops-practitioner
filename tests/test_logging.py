"""Unit and integration tests for structured JSON logging and log levels."""

import json
import logging
from pathlib import Path
import unittest
import uuid

import pandas as pd

from prodml.logging_conf import (
    JSONFormatter,
    get_correlation_id,
    set_correlation_id,
)
from prodml.predict import DurationPredictor
from prodml.train import create_model, train_pipeline


class TestStructuredLogging(unittest.TestCase):
    """Test JSONFormatter and correlation ID propagation via contextvars."""

    def setUp(self):
        self.logger = logging.getLogger("test.structured.logger")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # Capture log output in a custom handler
        self.log_records: list[dict] = []

        class ListHandler(logging.Handler):
            def __init__(handler_self):
                super().__init__()
                handler_self.setFormatter(JSONFormatter())

            def emit(handler_self, record):
                formatted = handler_self.format(record)
                handler_self.log_records.append(json.loads(formatted))

        self.handler = ListHandler()
        self.handler.log_records = self.log_records
        self.logger.addHandler(self.handler)

    def tearDown(self):
        set_correlation_id(None)
        self.logger.handlers.clear()

    def test_json_formatter_keys(self):
        """Every log line must carry timestamp, level, logger, message, correlation_id."""
        self.logger.info("Test message without correlation ID")
        self.assertEqual(len(self.log_records), 1)

        entry = self.log_records[0]
        self.assertIn("timestamp", entry)
        self.assertIn("level", entry)
        self.assertIn("logger", entry)
        self.assertIn("message", entry)
        self.assertIn("correlation_id", entry)

        self.assertEqual(entry["level"], "INFO")
        self.assertEqual(entry["message"], "Test message without correlation ID")
        self.assertEqual(entry["logger"], "test.structured.logger")
        self.assertIsNone(entry["correlation_id"])

    def test_correlation_id_contextvars(self):
        """Correlation ID set via contextvars must appear in the JSON log line."""
        corr_id = str(uuid.uuid4())
        set_correlation_id(corr_id)

        self.assertEqual(get_correlation_id(), corr_id)
        self.logger.info("Message with correlation ID")

        self.assertEqual(len(self.log_records), 1)
        entry = self.log_records[0]
        self.assertEqual(entry["correlation_id"], corr_id)

    def test_log_levels_in_predictor(self):
        """Test proper log level usage: DEBUG, INFO, WARNING, ERROR."""
        # 1. Train a miniature model for testing predictor logs
        X_dummy = [
            {"trip_distance": 2.0, "fare_amount": 10.0, "store_and_fwd_flag_encoded": 0}
        ] * 3
        y_dummy = pd.Series([10.0, 10.0, 10.0])
        model = create_model("linear_regression")
        pipeline = train_pipeline(X_dummy, y_dummy, model=model)

        predictor = DurationPredictor(model=pipeline)

        # Attach handler to prodml.predict
        predict_logger = logging.getLogger("prodml.predict")
        predict_logger.setLevel(logging.DEBUG)
        predict_logger.addHandler(self.handler)

        corr_id = str(uuid.uuid4())
        set_correlation_id(corr_id)

        # 2. WARNING: input outside the training range (trip_distance > 100)
        # 3. DEBUG: feature vector
        # 4. INFO: prediction served with latency
        valid_feature = {
            "trip_distance": 150.0,  # > 100 triggers warning
            "fare_amount": 25.0,
            "store_and_fwd_flag": "N",
        }
        pred = predictor.predict_one(valid_feature)
        self.assertIsInstance(pred, float)

        levels_logged = [rec["level"] for rec in self.log_records]
        messages_logged = [rec["message"] for rec in self.log_records]

        # Check WARNING
        self.assertIn("WARNING", levels_logged)
        self.assertTrue(
            any("Input outside the training range" in m for m in messages_logged)
        )

        # Check DEBUG
        self.assertIn("DEBUG", levels_logged)
        self.assertTrue(any("Feature vector:" in m for m in messages_logged))

        # Check INFO
        self.assertIn("INFO", levels_logged)
        self.assertTrue(any("Prediction served:" in m for m in messages_logged))

        # All records should share the same correlation ID
        for rec in self.log_records:
            self.assertEqual(rec["correlation_id"], corr_id)

        # 5. ERROR: validation rejection
        with self.assertRaises(ValueError):
            predictor.predict_one("not-a-dict")  # type: ignore

        self.assertEqual(self.log_records[-1]["level"], "ERROR")
        self.assertIn("Validation rejection", self.log_records[-1]["message"])

        # 6. ERROR: model load failure
        with self.assertRaises(FileNotFoundError):
            DurationPredictor.load(Path("non_existent_model_path.pkl"))

        self.assertEqual(self.log_records[-1]["level"], "ERROR")
        self.assertIn("Model load failure", self.log_records[-1]["message"])


if __name__ == "__main__":
    unittest.main()
