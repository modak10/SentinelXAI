"""Integration tests for the FastAPI backend (Phase 7).

Uses a synthetic model injected into the app state so the endpoints are
exercised end-to-end without the real dataset or committed artifact.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sentinelxai.api.main import create_app
from sentinelxai.database.store import SQLiteStore


@pytest.fixture
def client(synthetic_model, tmp_path):
    app = create_app()
    with TestClient(app) as c:
        # Inject the synthetic pipeline + an isolated DB into app state.
        app.state.inference = synthetic_model["service"]
        app.state.explainer = synthetic_model["explainer"]
        app.state.store = SQLiteStore(tmp_path / "api_test.db")
        # Simulate "no metrics artifact" so the endpoint contract is deterministic.
        app.state.metrics = None
        yield c


def test_health_available(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_available"] is True


def test_model_endpoint(client):
    r = client.get("/model")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["model_name"] == "lightgbm"
    assert len(body["feature_names"]) > 0


def test_predict_returns_decision_and_explanation(client):
    record = {f: 0.5 for f in _features(client)}
    r = client.post("/predict", json={"features": record})
    assert r.status_code == 200
    body = r.json()
    assert "predicted_class" in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["decision"]["risk"]
    assert isinstance(body["decision"]["recommendations"], list)
    assert body["explanation"]["human_explanation"]


def test_feature_importance(client):
    r = client.get("/feature-importance")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert len(body["importance"]) == len(_features(client))


def test_metrics_unavailable_without_artifact(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_batch_predict(client):
    records = [{f: 0.4 for f in _features(client)} for _ in range(2)]
    r = client.post("/batch_predict", json={"records": records})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_predict_missing_feature_is_422(client):
    r = client.post("/predict", json={"features": {"Flow Duration": 0.5}})
    assert r.status_code == 422


def _features(client):
    # Pull feature names from the model endpoint to keep the test self-contained.
    return client.get("/model").json()["feature_names"]
