"""Pydantic schemas for the prodml prediction API."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Input payload for a single trip duration prediction."""

    trip_distance: float = Field(
        ...,
        gt=0.0,
        lt=200.0,
        description="Trip distance in miles (must be > 0 and < 200).",
    )
    fare_amount: float = Field(
        ...,
        ge=0.0,
        description="Base fare amount in USD.",
    )
    total_amount: float = Field(
        ...,
        ge=0.0,
        description="Total amount charged in USD.",
    )
    store_and_fwd_flag: str = Field(
        default="N",
        pattern="^[YNyn]$",
        description="Whether trip record was held in vehicle memory ('Y' or 'N').",
    )
    tip_amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Tip amount in USD.",
    )
    tolls_amount: float = Field(
        default=0.0,
        ge=0.0,
        description="Tolls amount in USD.",
    )
    improvement_surcharge: float = Field(
        default=1.0,
        ge=0.0,
        description="Improvement surcharge in USD.",
    )
    congestion_surcharge: float = Field(
        default=0.0,
        ge=0.0,
        description="Congestion surcharge in USD.",
    )
    cbd_congestion_fee: float = Field(
        default=0.0,
        ge=0.0,
        description="CBD congestion fee in USD.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponse(BaseModel):
    """Output payload for a single trip duration prediction."""

    prediction: float = Field(
        ...,
        description="Predicted trip duration in minutes.",
    )
    model_version: str = Field(
        ...,
        description="Version of the model serving predictions.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Unique request trace identifier.",
    )
    latency_ms: float = Field(
        ...,
        description="Inference latency in milliseconds.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": 14.82,
                "model_version": "0.1.0",
                "correlation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "latency_ms": 1.25,
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    """Input payload for multiple trip duration predictions."""

    trips: list[PredictionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of trip records to score (1 to 1000 records).",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trips": [
                    {
                        "trip_distance": 2.1,
                        "fare_amount": 10.5,
                        "total_amount": 12.0,
                        "store_and_fwd_flag": "N",
                        "tip_amount": 1.5,
                        "tolls_amount": 0.0,
                        "improvement_surcharge": 1.0,
                        "congestion_surcharge": 0.0,
                        "cbd_congestion_fee": 0.0,
                    },
                    {
                        "trip_distance": 5.4,
                        "fare_amount": 22.0,
                        "total_amount": 26.5,
                        "store_and_fwd_flag": "N",
                        "tip_amount": 3.5,
                        "tolls_amount": 0.0,
                        "improvement_surcharge": 1.0,
                        "congestion_surcharge": 0.0,
                        "cbd_congestion_fee": 0.0,
                    },
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    """Output payload for batch trip duration predictions."""

    predictions: list[float] = Field(
        ...,
        description="List of predicted trip durations in minutes.",
    )
    model_version: str = Field(
        ...,
        description="Version of the model serving predictions.",
    )
    correlation_id: str | None = Field(
        default=None,
        description="Unique request trace identifier.",
    )
    latency_ms: float = Field(
        ...,
        description="Total batch inference latency in milliseconds.",
    )
    count: int = Field(
        ...,
        description="Number of trips scored.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predictions": [11.35, 23.40],
                "model_version": "0.1.0",
                "correlation_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "latency_ms": 3.42,
                "count": 2,
            }
        }
    )


class MetadataResponse(BaseModel):
    """Metadata response describing the active model artifact."""

    model_version: str = Field(
        ...,
        description="Version of the deployed model.",
    )
    training_date: str = Field(
        ...,
        description="Timestamp or date when model artifact was created.",
    )
    feature_names: list[str] = Field(
        ...,
        description="List of features expected by the model.",
    )
    framework: str = Field(
        ...,
        description="ML framework and version used to train the model.",
    )
    artifact_hash: str = Field(
        ...,
        description="SHA256 checksum of the persisted model artifact file.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_version": "0.1.0",
                "training_date": "2026-09-13T08:55:34+00:00",
                "feature_names": [
                    "cbd_congestion_fee",
                    "congestion_surcharge",
                    "fare_amount",
                    "improvement_surcharge",
                    "store_and_fwd_flag_encoded",
                    "tip_amount",
                    "tolls_amount",
                    "total_amount",
                    "trip_distance",
                ],
                "framework": "scikit-learn 1.9.1",
                "artifact_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            }
        }
    )


class HealthResponse(BaseModel):
    """Health status response."""

    status: str = Field(
        ...,
        description="Health status indicator ('healthy' or 'unhealthy').",
    )
    model_loaded: bool = Field(
        ...,
        description="True if the model object is loaded in memory.",
    )
    model_path: str = Field(
        ...,
        description="Resolved path of the active model file.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_path": "models/baseline.pkl",
            }
        }
    )
