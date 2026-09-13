# NYC Taxi Trip Duration Prediction Service (`prodml`)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker Image](https://img.shields.io/badge/Docker-ahmedfoly%2Fprodml--api-2496ED.svg)](https://hub.docker.com/r/ahmedfoly/prodml-api)
[![Coverage >= 70%](https://img.shields.io/badge/Coverage-80.19%25-brightgreen.svg)](https://pytest.org/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black%20%26%20Ruff-black.svg)](https://github.com/psf/black)

A production-grade machine learning service that predicts NYC taxi trip durations using a Random Forest regressor with scikit-learn and high-performance ONNX Runtime inference serving. Decomposed from an exploratory notebook into a clean, modular Python package, the service features structured JSON logging with UUID correlation tracking, strict Pydantic request validation, a multi-stage non-root Docker container, and an automated test suite enforcing a 70% coverage gate.

---

## ⚡ Quickstart: Zero to Prediction in 3 Commands

You can run and query the production service directly without cloning the repository or installing Python:

```bash
# 1. Run the container directly from Docker Hub (non-root appuser on port 8000)
docker run -d --name prodml-api -p 8000:8000 ahmedfoly/prodml-api:0.1.0

# 2. Confirm service health and in-memory model readiness
curl -s http://localhost:8000/health

# 3. Request a real-time trip duration prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 3.5,
    "fare_amount": 15.0,
    "total_amount": 18.5,
    "store_and_fwd_flag": "N"
  }'
```

**Real JSON Response (`200 OK`):**
```json
{
  "prediction": 13.793666666666665,
  "model_version": "0.1.0",
  "correlation_id": "e0a6a30e-4560-45a0-a707-9f855551673e",
  "latency_ms": 53.66139100078726
}
```

Interactive OpenAPI documentation is available at `http://localhost:8000/docs`.

---

## 📋 Prerequisites

### For Containerized Serving
- **Docker Engine**: Version 24.0 or newer
- **Docker Compose**: Version 2.20 or newer (optional, for compose-based orchestration)
- **Network access**: To pull `ahmedfoly/prodml-api:0.1.0` from Docker Hub

### For Local Development & Training
- **Operating System**: Linux (Ubuntu 22.04+ recommended) or macOS
- **Python**: Version `3.12` (recommended) or `3.11`
- **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (strongly recommended for fast resolution) or standard `pip`
- **Git**: Version 2.30+
- **`curl` & `jq`**: For interacting with the API and inspecting structured JSON logs

---

## 🌐 API Endpoints Reference

The service runs on port `8000` and exposes four REST endpoints:

| Endpoint | Method | Purpose | Key Constraints |
| :--- | :---: | :--- | :--- |
| [`/health`](#1-get-health) | `GET` | Readiness & liveness probe | Returns `200 OK` only if model is pre-loaded in memory |
| [`/metadata`](#2-get-metadata) | `GET` | Model lineage & metadata | Returns version, train date, feature schema, artifact hash |
| [`/predict`](#3-post-predict) | `POST` | Single-trip duration prediction | `trip_distance`: $0 < d < 200$, `fare_amount`: $\ge 0$ |
| [`/predict/batch`](#4-post-predictbatch) | `POST` | High-throughput batch prediction | List of 1–10,000 valid trip objects |

### 1. `GET /health`
Checks whether the process is alive and confirms the model pipeline object is resident in memory.

**Curl:**
```bash
curl -i http://localhost:8000/health
```

**Response (`200 OK`):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/baseline.pkl"
}
```

### 2. `GET /metadata`
Returns metadata about the currently served model artifact, enabling reproducibility and auditability.

**Curl:**
```bash
curl -i http://localhost:8000/metadata
```

**Response (`200 OK`):**
```json
{
  "model_version": "0.1.0",
  "training_date": "2026-09-13T06:58:32.417000+00:00",
  "feature_names": [
    "cbd_congestion_fee",
    "congestion_surcharge",
    "fare_amount",
    "improvement_surcharge",
    "store_and_fwd_flag_encoded",
    "tip_amount",
    "tolls_amount",
    "total_amount",
    "trip_distance"
  ],
  "framework": "scikit-learn 1.9.1",
  "artifact_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
```

### 3. `POST /predict`
Generates a trip duration estimate for a single ride.

**Curl:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "trip_distance": 4.2,
    "fare_amount": 18.0,
    "total_amount": 22.5,
    "store_and_fwd_flag": "N",
    "tip_amount": 3.0,
    "tolls_amount": 0.0,
    "improvement_surcharge": 1.0,
    "congestion_surcharge": 0.0,
    "cbd_congestion_fee": 0.0
  }'
```

**Response (`200 OK`):**
```json
{
  "prediction": 16.42,
  "model_version": "0.1.0",
  "correlation_id": "7b2c55b8-5eb8-4221-8256-42dcfb2756aa",
  "latency_ms": 51.34
}
```

**Validation Error Example (`422 Unprocessable Entity`):**
Submitting invalid inputs (e.g. `trip_distance: -5`) produces a clean, structured 422 error without leaking internal stack traces:
```json
{
  "error": "Validation Error",
  "message": "trip_distance: Input should be greater than 0",
  "details": [
    {
      "field": "trip_distance",
      "message": "Input should be greater than 0"
    }
  ],
  "correlation_id": "93a6288c-7fbe-497d-bb62-efb7d87bc650"
}
```

### 4. `POST /predict/batch`
Performs vectorized batch prediction over an array of trips.

**Curl:**
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"trip_distance": 1.2, "fare_amount": 7.5, "total_amount": 9.0, "store_and_fwd_flag": "N"},
      {"trip_distance": 8.5, "fare_amount": 32.0, "total_amount": 38.0, "store_and_fwd_flag": "Y"}
    ]
  }'
```

**Response (`200 OK`):**
```json
{
  "predictions": [7.84, 28.15],
  "count": 2,
  "model_version": "0.1.0",
  "correlation_id": "bfd9921b-6893-41c1-840f-7b705fa75d71",
  "latency_ms": 54.12
}
```

---

## 💻 Local Development Workflow

### 1. Clone the Repository
```bash
git clone https://github.com/A7med-Foly/mlops-practitioner.git
cd mlops-practitioner
```

### 2. Environment Setup
Create a virtual environment and install development dependencies:

```bash
# Using uv (fastest)
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using standard python
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Install pre-commit hooks (Black, Ruff, EOF fixer)
pre-commit install
```

### 3. Dealing with the Data

#### Data Source
The pipeline processes official **NYC Taxi and Limousine Commission (TLC) Trip Record Data**. The default sample file is stored in Apache Parquet format:
```
data/green_tripdata_2026-05.parquet
```

To download additional official NYC Green Taxi monthly data:
```bash
mkdir -p data
# Example: Download Green Taxi trip data
curl -o data/green_tripdata_2026-05.parquet \
  https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2024-01.parquet
```

#### Ingestion & Cleaning Pipeline (`src/prodml/data.py`)
When `prodml-train` or `prodml.data.load_data()` is executed:
1. **Duration Calculation**: Derives target `trip_duration` (in minutes):
   $$\text{trip\_duration} = \frac{\text{lpep\_dropoff\_datetime} - \text{lpep\_pickup\_datetime}}{60}$$
2. **Outlier Filtering**: Removes corrupted records where:
   - `trip_distance <= 0`
   - `trip_duration <= 0` or `trip_duration > 99.5th percentile` (removes unrealistic multi-day taxi rides)
3. **Train / Validation / Test Splitting**: Performs a deterministic 60% Train (22,521 rows), 20% Validation (7,507 rows), and 20% Test (7,507 rows) split using `random_state=42`.

#### Configuring Custom Data Paths
You can point the pipeline to any custom dataset path without modifying code:
```bash
# Via environment variable
export PRODML_DATA_PATH="data/custom_tripdata.parquet"

# Or via CLI flag
prodml-train --data-path data/custom_tripdata.parquet
```

### 4. Training the Model Pipeline
The package installs the `prodml-train` CLI entrypoint:

```bash
# Train Random Forest with 100 trees
prodml-train --n-estimators 100

# View all CLI options
prodml-train --help
```

CLI Options:
- `--data-path PATH`: Path to input parquet dataset (default: `data/green_tripdata_2026-05.parquet`)
- `--model-path PATH`: Destination for serialized artifact (default: `models/baseline.pkl`)
- `--model-type [random_forest|linear]`: Model algorithm choice
- `--n-estimators INT`: Number of trees for Random Forest (default: 100)

### 5. ONNX Export & Latency Benchmarking
To convert the scikit-learn model to an ONNX graph with dynamic batching dimensions:

```bash
# Export models/baseline.pkl -> models/baseline.onnx
python -m prodml.export

# Run the 500-row latency benchmark (Pickle vs ONNX Runtime)
python scripts/benchmark.py
```

### 6. Running the Test Suite
The repository enforces a strict test coverage gate ($\ge 70\%$) configured in `pyproject.toml`:

```bash
# Run all tests and display coverage table
pytest

# Run tests with verbose output
pytest -v
```

Current test status: **41 passed**, **80.19% code coverage**.

### 7. Code Formatting & Linting
Run pre-commit checks locally before submitting code:

```bash
pre-commit run --all-files
```

### 8. Running the API Locally
Start the development server with live reload:

```bash
uvicorn prodml.api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🐳 Docker & Docker Compose

### Multi-Stage Docker Build
Build and run the production image locally:

```bash
# Build multi-stage image
docker build -f docker/Dockerfile -t prodml-api:latest .

# Run with container healthcheck
docker run -d --name prodml-api -p 8000:8000 prodml-api:latest

# Check container health status
docker inspect --format='{{json .State.Health.Status}}' prodml-api
```

### Orchestration with Docker Compose
Use [`docker/docker-compose.yml`] to run the service with persistent read-only model volume mounts and automatic restart policies:

```bash
# Start service in the background
docker compose -f docker/docker-compose.yml up -d

# View structured logs
docker compose -f docker/docker-compose.yml logs -f

# Shut down and cleanup
docker compose -f docker/docker-compose.yml down
```

---

## 📁 Repository Tree

```text
.
├── data/
│   └── green_tripdata_2026-05.parquet    # Sample NYC Green Taxi dataset
├── docker/
│   ├── Dockerfile                        # Multi-stage build (python:3.12-slim, non-root appuser)
│   └── docker-compose.yml                # Service orchestration with volume mounts & restart policies
├── models/
│   ├── baseline.onnx                     # Exported ONNX computational graph (dynamic batch axes)
│   └── baseline.pkl                      # Scikit-learn trained pipeline artifact
├── notebooks/
│   └── 00_baseline.ipynb                 # Original exploratory notebook
├── reports/
│   ├── module-1.md                       # Validation metrics, ONNX benchmarks, & maturity assessment
│   └── pull_request.md                   # PR description template with ticked Definition of Done
├── scripts/
│   └── benchmark.py                      # 500-row latency benchmark (Pickle vs ONNX Runtime)
├── src/prodml/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI application & lifespan context manager
│   │   └── schemas.py                    # Pydantic v2 validation models
│   ├── config.py                         # Settings management via pydantic-settings
│   ├── data.py                           # Ingestion, duration derivation, outlier filtering, splits
│   ├── decorators.py                     # Custom @timed execution latency decorator
│   ├── export.py                         # skl2onnx conversion pipeline
│   ├── features.py                       # Categorical encoding & DictVectorizer preparation
│   ├── logging_conf.py                   # Structured JSON logging formatter & correlation contextvars
│   ├── predict.py                        # DurationPredictor interface seam with input range checks
│   └── train.py                          # Training pipeline & CLI entrypoint (prodml-train)
├── tests/
│   ├── conftest.py                       # Session-scoped fixtures (FastAPI TestClient, mock model)
│   ├── test_api.py                       # API endpoint status & validation tests (with monkeypatch)
│   ├── test_features.py                  # Parameterized edge case tests
│   ├── test_logging.py                   # JSON formatting & correlation ID verification
│   ├── test_predict.py                   # Prediction sanity & determinism checks
│   ├── test_prodml.py                    # Package integration tests
│   └── test_serialization.py            # Pickle vs ONNX numerical parity test (atol=1e-4)
├── .dockerignore                         # Excludes .git, .venv, tests, notebooks from Docker build
├── .pre-commit-config.yaml               # Black, Ruff, and EOF formatting hooks
├── pyproject.toml                        # Dependencies, CLI entrypoints, & pytest coverage gate
└── README.md                             # Comprehensive project documentation
```
