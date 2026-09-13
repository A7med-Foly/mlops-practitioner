"""FastAPI application for NYC green taxi trip duration prediction.

Exposes endpoints:
- GET  /health         : Health status confirming model is loaded in memory
- GET  /metadata       : Model version, training date, feature names, framework, SHA256 artifact hash
- POST /predict        : Single-record prediction
- POST /predict/batch  : Multi-record batch prediction
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import logging
from pathlib import Path
import time
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import sklearn

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import Settings, get_settings
from prodml.logging_conf import (
    get_correlation_id,
    set_correlation_id,
    setup_logging,
)
from prodml.predict import DurationPredictor

logger = logging.getLogger("prodml.api")


def compute_artifact_hash(file_path: Path) -> str:
    """Compute SHA256 checksum of the serialized model file."""
    if not file_path.exists():
        return "artifact-not-found"
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def compute_model_metadata(
    settings: Settings, predictor: DurationPredictor
) -> dict[str, Any]:
    """Extract metadata, feature names, framework, and hash from model artifact."""
    model_path = settings.model_path

    # Compute hash
    artifact_hash = compute_artifact_hash(model_path)

    # Compute training timestamp from file mtime
    if model_path.exists():
        mtime = model_path.stat().st_mtime
        training_date = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    else:
        training_date = datetime.now(timezone.utc).isoformat()

    # Extract feature names from vectorizer
    feature_names: list[str] = []
    model_obj = getattr(predictor, "model", None)
    if hasattr(model_obj, "named_steps") and "vectorizer" in model_obj.named_steps:
        dv = model_obj.named_steps["vectorizer"]
        if hasattr(dv, "get_feature_names_out"):
            feature_names = list(dv.get_feature_names_out())

    if not feature_names:
        feature_names = [
            "cbd_congestion_fee",
            "congestion_surcharge",
            "fare_amount",
            "improvement_surcharge",
            "store_and_fwd_flag_encoded",
            "tip_amount",
            "tolls_amount",
            "total_amount",
            "trip_distance",
        ]

    return {
        "model_version": "0.1.0",
        "training_date": training_date,
        "feature_names": feature_names,
        "framework": f"scikit-learn {sklearn.__version__}",
        "artifact_hash": artifact_hash,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: loads model once on startup."""
    settings = get_settings()
    setup_logging(level=settings.log_level)

    logger.info("Initializing application and loading model into memory...")
    try:
        predictor = DurationPredictor.load(settings.model_path)
        metadata = compute_model_metadata(settings, predictor)
        app.state.predictor = predictor
        app.state.metadata = metadata
        logger.info("Model loaded successfully into app.state.predictor.")
    except Exception as exc:
        logger.error("Failed to load model during startup: %s", exc)
        app.state.predictor = None
        app.state.metadata = {}

    yield

    logger.info("Shutting down application...")
    app.state.predictor = None
    app.state.metadata = {}


app = FastAPI(
    title="NYC Green Taxi Duration Prediction Service",
    description="Production ML inference API for estimating NYC Green Taxi trip durations.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    """Middleware attaching a correlation ID via contextvars and X-Request-ID response header."""
    header_id = request.headers.get("X-Request-ID")
    correlation_id = header_id if header_id else str(uuid.uuid4())
    set_correlation_id(correlation_id)

    logger.info("Incoming request %s %s", request.method, request.url.path)
    start_time = time.perf_counter()

    response: Response = await call_next(request)

    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Request-ID"] = correlation_id
    logger.info(
        "Request %s %s completed with status %d in %.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Turn validation errors into a clean 422 JSON response with a readable message."""
    formatted_errors = []
    messages = []
    for error in exc.errors():
        loc = [str(x) for x in error.get("loc", []) if x != "body"]
        field_name = ".".join(loc) if loc else "root"
        msg = error.get("msg", "Invalid value")
        formatted_errors.append({"field": field_name, "message": msg})
        messages.append(f"{field_name}: {msg}")

    summary_message = "; ".join(messages)
    logger.error("Validation rejection: %s", summary_message)

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": summary_message,
            "details": formatted_errors,
            "correlation_id": get_correlation_id(),
        },
        headers={"X-Request-ID": get_correlation_id() or ""},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Turn unexpected errors into a 500 logging the traceback without leaking it to client."""
    logger.exception(
        "Unexpected server error on %s %s: %s", request.method, request.url.path, exc
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please contact support with the correlation ID.",
            "correlation_id": get_correlation_id(),
        },
        headers={"X-Request-ID": get_correlation_id() or ""},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns 200 OK only if the model object is loaded in memory.",
)
async def health_check():
    """Verify that the process is healthy and the ML model is resident in memory."""
    settings = get_settings()
    predictor = getattr(app.state, "predictor", None)

    if predictor is None or getattr(predictor, "model", None) is None:
        logger.error("Health check failed: model object not loaded in memory.")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "model_loaded": False,
                "model_path": str(settings.model_path),
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_path=str(settings.model_path),
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Model metadata",
    description="Returns model version, training date, feature names, framework, and artifact hash.",
)
async def get_metadata():
    """Retrieve metadata of the currently served model artifact."""
    metadata = getattr(app.state, "metadata", None)
    if not metadata:
        settings = get_settings()
        predictor = getattr(app.state, "predictor", None)
        if predictor is not None:
            metadata = compute_model_metadata(settings, predictor)
            app.state.metadata = metadata
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model metadata is unavailable; model not loaded.",
            )

    return MetadataResponse(**metadata)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Single prediction",
    description="Predict trip duration in minutes for a single taxi trip.",
)
async def predict_single(payload: PredictionRequest):
    """Serve a single trip duration prediction."""
    predictor: DurationPredictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded into memory.",
        )

    start_time = time.perf_counter()
    feature_dict = payload.model_dump()
    predicted_duration = predictor.predict_one(feature_dict)
    latency_ms = (time.perf_counter() - start_time) * 1000

    metadata = getattr(app.state, "metadata", {})
    version = metadata.get("model_version", "0.1.0")

    return PredictionResponse(
        prediction=predicted_duration,
        model_version=version,
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Batch prediction",
    description="Predict trip duration in minutes for a list of taxi trips.",
)
async def predict_batch(payload: BatchPredictionRequest):
    """Serve batch trip duration predictions for a list of records."""
    predictor: DurationPredictor | None = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded into memory.",
        )

    start_time = time.perf_counter()
    feature_list = [trip.model_dump() for trip in payload.trips]
    predictions = predictor.predict_batch(feature_list)
    latency_ms = (time.perf_counter() - start_time) * 1000

    metadata = getattr(app.state, "metadata", {})
    version = metadata.get("model_version", "0.1.0")

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=version,
        correlation_id=get_correlation_id(),
        latency_ms=latency_ms,
        count=len(predictions),
    )
