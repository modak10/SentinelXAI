"""Pydantic request/response schemas for the FastAPI backend (Phase 7).

All response shapes are explicit and validated, so the Streamlit dashboard and
any external client get a stable contract. Feature records are keyed by the
engineered feature name (matching the model's training input, never raw
network-flow CSV columns — see ``src/sentinelxai/models/inference.py``).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureRecord(BaseModel):
    """One prediction input: engineered feature name -> value."""

    features: dict[str, float] = Field(..., description="Engineered feature name -> value.")


class BatchFeatureRequest(BaseModel):
    records: list[dict[str, float]] = Field(..., min_length=1, max_length=10_000)


class ExplanationOut(BaseModel):
    predicted_class: str
    top_features: list[dict[str, Any]] = Field(
        default_factory=list,
        description="SHAP-derived contributors: name, value, direction, weight.",
    )
    human_explanation: list[str] = Field(default_factory=list)


class DecisionOut(BaseModel):
    risk: str
    risk_rank: int
    confidence_band: str
    is_failure_candidate: bool
    recommendations: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    decision: DecisionOut
    explanation: ExplanationOut


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    version: str


class ModelResponse(BaseModel):
    available: bool
    model_name: str | None = None
    n_features: int | None = None
    feature_names: list[str] = Field(default_factory=list)
    label_names: list[str] = Field(default_factory=list)


class MetricsResponse(BaseModel):
    available: bool
    accuracy: float | None = None
    f1_macro: float | None = None
    f1_weighted: float | None = None
    training_time_seconds: float | None = None
    inference_time_per_sample_ms: float | None = None
    label_names: list[str] = Field(default_factory=list)


class FeatureImportanceItem(BaseModel):
    name: str
    importance: float
    rank: int


class FeatureImportanceResponse(BaseModel):
    available: bool
    method: str = "lightgbm_gain"
    importance: list[FeatureImportanceItem] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
