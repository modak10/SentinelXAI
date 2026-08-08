"""FastAPI backend (Phase 7).

Exposes the full decision-support pipeline over REST:

* ``POST /predict`` / ``POST /batch_predict`` — prediction + SHAP explanation + risk/recommendation
* ``POST /upload`` — CSV of engineered-feature rows → batch predictions
* ``GET /health``, ``GET /model``, ``GET /metrics``, ``GET /feature-importance``

The app is built by :func:`create_app` so it can be imported by tests (with a
synthetic model) and run by ``uvicorn src.api.main:app``. Model loading is
graceful: if the trained artifact is absent, ``/predict`` returns 503 with a
clear message instead of crashing (per CLAUDE.md "Never crash").
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile

from sentinelxai.config import get_config
from sentinelxai.database.store import SQLiteStore
from sentinelxai.decision import build_decision_payload, load_decision_config
from sentinelxai.explainability import SHAPExplainer, humanize_contributions
from sentinelxai.logging_setup import configure_logging, get_logger
from sentinelxai.models.inference import InferenceService

from .schemas import (
    BatchPredictResponse,
    BatchFeatureRequest,
    DecisionOut,
    ExplanationOut,
    FeatureImportanceItem,
    FeatureImportanceResponse,
    FeatureRecord,
    HealthResponse,
    MetricsResponse,
    ModelResponse,
    PredictResponse,
)

logger = get_logger("sentinelxai.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    cfg = get_config()
    app.state.config = cfg
    app.state.decision_cfg = load_decision_config()

    # Inference service — graceful if artifacts are missing.
    try:
        app.state.inference = InferenceService.from_config(cfg)
    except FileNotFoundError as exc:
        logger.warning("Inference model not loaded: %s", exc)
        app.state.inference = InferenceService()

    # SHAP explainer — only if a model is available.
    if app.state.inference.is_available:
        app.state.explainer = SHAPExplainer.from_model(
            app.state.inference._model, app.state.inference.feature_names
        )
    else:
        app.state.explainer = None

    # SQLite store for prediction history.
    db_path = cfg.paths.data_processed_dir / "predictions.db"
    app.state.store = SQLiteStore(db_path)

    # Load evaluation metrics if present (for GET /metrics).
    metrics_path = cfg.paths.reports_dir / "lightgbm" / "lightgbm_metrics.json"
    app.state.metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None

    logger.info("SentinelXAI API started (model_available=%s)", app.state.inference.is_available)
    yield
    try:
        app.state.store.close()
    except Exception:  # pragma: no cover - best effort
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="SentinelXAI API",
        version="0.1.0",
        description="Explainable AI Decision Intelligence for Network Intrusion Detection",
        lifespan=lifespan,
    )
    _register_routes(app)
    return app


def _build_response(
    app: FastAPI, features: dict, predicted_class: str, confidence: float, probabilities: dict
) -> PredictResponse:
    """Assemble the explanation + decision payload around a prediction."""
    from sentinelxai.explainability.shap_engine import LocalContribution

    decision_cfg = app.state.decision_cfg

    explanation_out = ExplanationOut(predicted_class=predicted_class)
    if app.state.explainer is not None:
        local = app.state.explainer.explain_single(features, predicted_class)
        explanation_out = ExplanationOut(
            predicted_class=predicted_class,
            top_features=[c.__dict__ for c in local.top_contributions],
            human_explanation=humanize_contributions(local),
        )

    contributions = [
        LocalContribution(
            name=f["name"],
            value=f["value"],
            direction=f["direction"],
            rank=i,
            weight=f["weight"],
        )
        for i, f in enumerate(explanation_out.top_features, start=1)
    ]
    decision_dict = build_decision_payload(
        predicted_class=predicted_class,
        confidence=confidence,
        top_contributions=contributions,
        cfg=decision_cfg,
        human_explanation=explanation_out.human_explanation,
    )
    decision_out = DecisionOut(
        risk=decision_dict["risk"],
        risk_rank=decision_dict["risk_rank"],
        confidence_band=decision_dict["confidence_band"],
        is_failure_candidate=decision_dict["is_failure_candidate"],
        recommendations=decision_dict["recommendations"],
    )
    return PredictResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probabilities,
        decision=decision_out,
        explanation=explanation_out,
    )


def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if app.state.inference.is_available else "degraded",
            model_available=app.state.inference.is_available,
            version="0.1.0",
        )

    @app.get("/model", response_model=ModelResponse)
    def model() -> ModelResponse:
        inf = app.state.inference
        if not inf.is_available:
            return ModelResponse(available=False)
        return ModelResponse(
            available=True,
            model_name="lightgbm",
            n_features=len(inf.feature_names),
            feature_names=inf.feature_names,
            label_names=inf.label_names,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics() -> MetricsResponse:
        m = app.state.metrics
        if not m:
            return MetricsResponse(available=False)
        return MetricsResponse(
            available=True,
            accuracy=m.get("accuracy"),
            f1_macro=m.get("f1_macro"),
            f1_weighted=m.get("f1_weighted"),
            training_time_seconds=m.get("training_time_seconds"),
            inference_time_per_sample_ms=m.get("inference_time_per_sample_ms"),
            label_names=m.get("label_names", []),
        )

    @app.get("/feature-importance", response_model=FeatureImportanceResponse)
    def feature_importance() -> FeatureImportanceResponse:
        inf = app.state.inference
        if not inf.is_available:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        booster = getattr(getattr(inf._model, "booster_", None), "feature_importance", None)
        import numpy as np

        if booster is not None:
            raw = np.asarray(booster(importance_type="gain"), dtype=float)
        else:  # pragma: no cover - lightgbm always exposes booster_
            raw = np.asarray(getattr(inf._model, "feature_importances_", []), dtype=float)
        order = np.argsort(raw)[::-1]
        items = [
            FeatureImportanceItem(name=inf.feature_names[i], importance=float(raw[i]), rank=r)
            for r, i in enumerate(order, start=1)
        ]
        return FeatureImportanceResponse(available=True, importance=items)

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: FeatureRecord) -> PredictResponse:
        return _predict_one(app, req.features)

    @app.post("/batch_predict", response_model=BatchPredictResponse)
    def batch_predict(req: BatchFeatureRequest) -> BatchPredictResponse:
        inf = app.state.inference
        if not inf.is_available:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        try:
            result = inf.predict(req.records)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        responses = [
            _predict_one(app, rec, pc, conf)
            for rec, pc, conf in zip(req.records, result.predicted_classes, result.confidences, strict=True)
        ]
        return BatchPredictResponse(predictions=responses)

    @app.post("/upload")
    async def upload(file: UploadFile) -> BatchPredictResponse:
        """Accept a CSV of engineered-feature rows; return batch predictions."""
        import io

        import pandas as pd

        inf = app.state.inference
        if not inf.is_available:
            raise HTTPException(status_code=503, detail="Model not loaded.")
        if file.content_type and "csv" not in file.content_type and not file.filename.endswith(".csv"):
            raise HTTPException(status_code=415, detail="Only CSV upload is supported.")
        try:
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")
        records = df.to_dict(orient="records")
        try:
            result = inf.predict(records)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        responses = [
            _predict_one(app, rec, pc, conf)
            for rec, pc, conf in zip(records, result.predicted_classes, result.confidences, strict=True)
        ]
        return BatchPredictResponse(predictions=responses)


def _predict_one(app, features, predicted_class=None, confidence=None) -> PredictResponse:
    inf = app.state.inference
    if not inf.is_available:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        result = inf.predict_single(features)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    pc = predicted_class or result.predicted_classes[0]
    conf = confidence if confidence is not None else result.confidences[0]
    proba = result.probabilities[0]
    resp = _build_response(app, features, pc, conf, proba)
    # Best-effort history logging — never break the response.
    try:
        app.state.store.log_prediction(
            predicted_class=pc,
            confidence=conf,
            risk=resp.decision.risk,
            input_features=features,
            probabilities=proba,
        )
    except Exception:  # pragma: no cover - best effort
        logger.warning("Failed to log prediction to SQLite", exc_info=True)
    return resp


app = create_app()
