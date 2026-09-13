"""Unit and integration tests for FastAPI prediction endpoints."""

import unittest
from fastapi.testclient import TestClient

from prodml.api.main import app


class TestFastAPIServing(unittest.TestCase):
    """Test API endpoints: /health, /metadata, /predict, /predict/batch, and error handlers."""

    @classmethod
    def setUpClass(cls):
        # Using TestClient as context manager triggers the lifespan startup/shutdown
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_health_endpoint_success(self):
        """GET /health must return 200 only if model is loaded in memory."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("model_path", data)

    def test_metadata_endpoint(self):
        """GET /metadata must return version, date, feature names, framework, artifact hash."""
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["model_version"], "0.1.0")
        self.assertIn("training_date", data)
        self.assertIsInstance(data["feature_names"], list)
        self.assertGreater(len(data["feature_names"]), 0)
        self.assertIn("scikit-learn", data["framework"])
        self.assertIsInstance(data["artifact_hash"], str)
        self.assertGreater(len(data["artifact_hash"]), 10)

    def test_predict_single_success(self):
        """POST /predict must return single prediction, model_version, correlation_id, latency."""
        payload = {
            "trip_distance": 3.5,
            "fare_amount": 15.0,
            "total_amount": 18.5,
            "store_and_fwd_flag": "N",
            "tip_amount": 2.0,
            "tolls_amount": 0.0,
            "improvement_surcharge": 1.0,
            "congestion_surcharge": 0.0,
            "cbd_congestion_fee": 0.0,
        }
        custom_request_id = "test-request-id-12345"
        response = self.client.post(
            "/predict",
            json=payload,
            headers={"X-Request-ID": custom_request_id},
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("prediction", data)
        self.assertIsInstance(data["prediction"], float)
        self.assertGreater(data["prediction"], 0.0)
        self.assertEqual(data["model_version"], "0.1.0")
        self.assertEqual(data["correlation_id"], custom_request_id)
        self.assertEqual(response.headers.get("X-Request-ID"), custom_request_id)
        self.assertIn("latency_ms", data)
        self.assertGreater(data["latency_ms"], 0.0)

    def test_predict_batch_success(self):
        """POST /predict/batch must return list in, list out with count and latency."""
        payload = {
            "trips": [
                {
                    "trip_distance": 2.0,
                    "fare_amount": 10.0,
                    "total_amount": 12.0,
                    "store_and_fwd_flag": "N",
                },
                {
                    "trip_distance": 4.5,
                    "fare_amount": 18.0,
                    "total_amount": 21.0,
                    "store_and_fwd_flag": "Y",
                },
            ]
        }
        response = self.client.post("/predict/batch", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("predictions", data)
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["count"], 2)
        self.assertIn("latency_ms", data)

    def test_validation_error_handling_trip_distance_negative(self):
        """Sending trip_distance: -5 must return clean 422 with readable message, not a stack trace."""
        payload = {
            "trip_distance": -5.0,
            "fare_amount": 15.0,
            "total_amount": 18.5,
            "store_and_fwd_flag": "N",
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)

        data = response.json()
        self.assertEqual(data.get("error"), "Validation Error")
        self.assertIn("trip_distance", data.get("message", ""))
        self.assertIn("greater than 0", data.get("message", ""))
        # Verify no stack trace leaked
        self.assertNotIn("Traceback", response.text)
        self.assertNotIn('File "', response.text)
        self.assertIn("details", data)

    def test_validation_error_handling_trip_distance_too_large(self):
        """Sending trip_distance: 250 must return 422 since lt=200 constraint is enforced."""
        payload = {
            "trip_distance": 250.0,
            "fare_amount": 15.0,
            "total_amount": 18.5,
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("trip_distance", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
