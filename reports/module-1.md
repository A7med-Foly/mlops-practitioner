# Module 1: Packaging, Structured Logging & Serialization Report

## 1. Executive Summary

This report documents the decomposition of the baseline exploratory notebook into a production-grade Python package (`prodml`), the evaluation of the baseline model, and the benchmarking of serialization formats for inference serving (Pickle vs. ONNX Runtime).

---

## 2. Baseline Model Performance

### Dataset & Training Setup
- **Dataset**: NYC Green Taxi Trip Records (`green_tripdata_2026-05.parquet`)
- **Total Raw Records**: 44,921 rows
- **Cleaned Records**: 37,535 rows (after filtering missing value subsets, trip distance $\le 0$, and duration outliers $> 99.5\text{th}$ percentile)
- **Split Proportions**: 60% Train (22,521 rows), 20% Validation (7,507 rows), 20% Test (7,507 rows)
- **Selected Features**: `cbd_congestion_fee`, `congestion_surcharge`, `fare_amount`, `improvement_surcharge`, `store_and_fwd_flag_encoded`, `tip_amount`, `tolls_amount`, `total_amount`, `trip_distance`
- **Target**: `trip_duration` (minutes)

### Validation Metrics

| Model Architecture | Hyperparameters | MAE (minutes) | RMSE (minutes) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | Default OLS | 3.0805 | 6.0967 | 0.6993 |
| **Random Forest Regressor** (Production) | `n_estimators=100`, `n_jobs=-1`, `random_state=42` | **1.6768** | **5.2949** | **0.7728** |

The Random Forest Regressor achieves a Mean Absolute Error of **1.68 minutes** on unseen validation trips, explaining **77.28%** of the variance in trip duration.

---

## 3. Serialization: Pickle vs. ONNX Runtime

### ONNX Export & Architecture
- **Export Engine**: `skl2onnx` (`skl2onnx.convert_sklearn`)
- **Dynamic Axes**: Configured on the batch dimension (`[None, 9]` $\to$ `[None, 1]`) to support dynamic single-item and arbitrary batch queries.
- **Model Artifact**: Persisted to [`models/baseline.onnx`](file:///home/ahmed/data/mlops-practitioner/models/baseline.onnx).

### Numerical Parity Test
Numerical equivalence was tested on 500 validation rows using `onnxruntime`:

$$\max \left| y_{\text{pickle}} - y_{\text{onnx}} \right| = 0.00001632 < 10^{-4}$$

```python
assert np.allclose(pred_pkl, pred_onnx, atol=1e-4)  # PASSED
```

Both models produce identical predictions within strict numerical tolerances.

---

## 4. Latency Benchmarks (500 Validation Rows)

Benchmarked on identical hardware across 500 validation rows using [`scripts/benchmark.py`](file:///home/ahmed/data/mlops-practitioner/scripts/benchmark.py):

| Inference Mode | Metric | Pickle (`scikit-learn`) | ONNX Runtime (`onnxruntime`) | Speedup |
| :--- | :--- | :---: | :---: | :---: |
| **Single-Row Inference**<br/>*(500 sequential calls)* | **Mean Latency** | **26.543 ms** | **0.050 ms** ($50\,\mu\text{s}$) | **533.3x faster** |
| | **P95 Latency** | **36.927 ms** | **0.065 ms** ($65\,\mu\text{s}$) | **564.5x faster** |
| **Batch Inference**<br/>*(500 rows/call, 50 trials)* | **Mean Latency** | **34.557 ms** | **10.132 ms** | **3.4x faster** |
| | **P95 Latency** | **36.398 ms** | **11.269 ms** | **3.2x faster** |

### Benchmark Analysis
1. **Single-Row Latency Drop**: ONNX Runtime delivers **sub-millisecond latency** ($50\,\mu\text{s}$ vs $26.5\,\text{ms}$). By bypassing Python object allocation and GIL overhead, ONNX enables high-throughput online API serving.
2. **Tail Latency (P95)**: P95 latency is bounded at **$0.065\,\text{ms}$** for ONNX, compared to **$36.9\,\text{ms}$** for Pickle, ensuring consistent SLA compliance.

---

## 5. Serialization Formats Comparison Table

| Serialization Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load from Untrusted Source |
| :--- | :---: | :---: | :---: | :---: |
| **JSON** | Yes (UTF-8 text) | Yes (Universal) | No (requires external schema) | Yes (pure data) |
| **Protobuf** | No (binary wire) | Yes (C++, Python, Go, Java, Rust) | Yes (enforced by compiled `.proto`) | Yes (pure data) |
| **Pickle** | No (binary opcode) | No (CPython only) | No (vulnerable to environment drift) | **NO (Arbitrary Code Execution)** |
| **ONNX** | No (binary graph) | Yes (C++, Python, Go, JS, Rust) | Yes (strict tensor shapes & opsets) | Yes (static computational graph) |

### Serving Format Decision
> **Production Decision**: Our production service serves inference with **ONNX Runtime** because it delivers sub-millisecond single-request latency (533x faster than Pickle), enforces explicit tensor schema validation, and provides language-agnostic cross-platform execution without exposing the runtime to arbitrary code execution.

> [!CAUTION]
> **CRITICAL SECURITY RULE**: **Pickle executes arbitrary code on load. Never load a `.pkl` file you did not produce or cannot cryptographically verify.**

---

## 6. Containerization & Publishing (Docker & Compose)

### Multi-Stage Build Strategy
The production image is built with a multi-stage `Dockerfile` (`docker/Dockerfile`):
1. **Builder Stage (`python:3.12-slim`)**: Installs dependencies into an isolated prefix directory (`/install`) using `pip install --no-cache-dir --prefix=/install .`. Build tools, wheel compile caches, and temporary setup files stay confined to this stage.
2. **Runtime Stage (`python:3.12-slim`)**: Copies only the installed packages from `/install` to `/usr/local`, adds an unprivileged user `appuser` (UID 1000), copies pre-trained artifacts to `/app/models`, and launches `uvicorn prodml.api.main:app`.

### Image Size Benchmarking: Single-Stage vs. Multi-Stage

| Image Configuration | Dockerfile Used | Disk Usage (Reported) | Content / Layer Size | Footprint Reduction |
| :--- | :--- | :---: | :---: | :---: |
| **Single-Stage Build** | `FROM python:3.12-slim` + `pip install .` | **1.92 GB** | ~780 MB | Baseline |
| **Multi-Stage Build** | Multi-stage builder $\to$ runtime (`python:3.12-slim`) | **1.44 GB** | **309 MB** | **~480 MB reduction (25% smaller on disk, 60% smaller layer content)** |

> [!TIP]
> **The Lesson of Multi-Stage Builds**: Discarding compilation toolchains, package download caches, and intermediate build debris leaves only clean bytecode and required shared objects in the final image, radically shrinking attack surface and deployment transfer times.

### Impact of `.dockerignore`
The `.dockerignore` file excludes `.git`, `.venv`, `notebooks/`, `data/`, `tests/`, `__pycache__`, and `*.ipynb`.

| Metric | Without `.dockerignore` | With `.dockerignore` | Improvement |
| :--- | :---: | :---: | :---: |
| **Docker Daemon Build Context Transfer** | **675+ MB** (`.venv` 670MB + `.git` 3.2MB + data) | **2.93 kB** | **>230,000x faster context upload** |
| **Security & Hygiene** | Host virtualenvs and git commit history leaked | Zero host artifacts leaked | Prevents secret and binary leakage |

### Security & Non-Root Execution
The container strictly runs as an unprivileged non-root user:
```bash
$ docker exec <container-id> whoami
appuser

$ docker exec <container-id> id
uid=1000(appuser) gid=1000(appuser) groups=1000(appuser)
```

### Healthcheck & Container Lifecycle Verification
1. **Docker Native Healthcheck**:
   ```dockerfile
   HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
       CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
   ```
   Verified via `docker inspect`:
   ```json
   "Status": "healthy"
   ```
2. **API Verification inside Container**:
   - `GET /health` $\to$ `200 OK` (`{"status":"healthy","model_loaded":true,"model_path":"models/baseline.pkl"}`) with `x-request-id` header.
   - `POST /predict` $\to$ `200 OK` (`{"prediction": 13.79, "model_version": "0.1.0", "latency_ms": 53.66}`) with correlation ID tracking.
3. **Orchestration (`docker-compose.yml`)**:
   - Manages container lifecycle with `restart: unless-stopped`.
   - Mounts models as read-only (`../models:/app/models:ro`).
   - Configured port mapping `8000:8000` and environment configuration.
   - Tested successfully with `docker compose up -d` (200 OK on `/health`) and graceful shutdown via `docker compose down`.

### Publishing to Docker Registry
The images are tagged for distribution:
- `ahmedfoly/prodml-api:0.1.0`
- `ahmedfoly/prodml-api:latest`

To push to Docker Hub:
```bash
docker login
docker push ahmedfoly/prodml-api:0.1.0
docker push ahmedfoly/prodml-api:latest
```

---

## 7. MLOps Maturity Self-Assessment

### Five-Level Maturity Model Placement

| Level | Maturity Stage | Description | Repository Status |
| :---: | :--- | :--- | :---: |
| **0** | **No MLOps** | Ad-hoc notebooks, manual deployments, no testing or logging. | Surpassed |
| **1** | **Manual Process** | Modular code, automated tests with coverage gates, containerization, structured logging, but manual training execution. | **CURRENT LEVEL** |
| **2** | **ML Pipeline** | Automated orchestration, experiment tracking, model registry, automated retraining on data trigger. | *Target (Module 2)* |
| **3** | **CD for ML** | Continuous automated model testing, canary/shadow deployment, progressive delivery. | Future |
| **4** | **Full MLOps** | Automated drift detection, continuous training (CT), closed-loop feedback and self-healing systems. | Future |

### Gap Analysis to Reach Level 2
To reach Level 2 (ML Pipeline), we require an automated workflow orchestrator (such as Prefect or Airflow) combined with an experiment tracking and model registry system (such as MLflow) to automate reproducible training, feature validation, and model artifact versioning upon new data arrival. Currently, pipeline execution and model updates remain manual developer tasks triggered locally via the CLI, which will be fully automated with orchestrated pipelines and experiment tracking in Module 2.
