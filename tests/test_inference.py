"""Tests for the inference service (Phase 7 backbone)."""

from __future__ import annotations

import pytest

from sentinelxai.models.inference import InferenceService, ModelNotLoadedError


def test_predict_single_returns_valid_label(synthetic_model):
    svc = synthetic_model["service"]
    record = {f: 0.5 for f in synthetic_model["feature_names"]}
    result = svc.predict_single(record)
    assert len(result.predicted_classes) == 1
    assert result.predicted_classes[0] in synthetic_model["label_names"]
    assert 0.0 <= result.confidences[0] <= 1.0
    # Probabilities sum to ~1.
    assert abs(sum(result.probabilities[0].values()) - 1.0) < 1e-5


def test_predict_batch(synthetic_model):
    svc = synthetic_model["service"]
    records = [{f: 0.3 for f in synthetic_model["feature_names"]} for _ in range(3)]
    result = svc.predict(records)
    assert len(result.predicted_classes) == 3


def test_predict_missing_feature_raises(synthetic_model):
    svc = synthetic_model["service"]
    bad = {"Flow Duration": 0.5}  # missing the other required features
    with pytest.raises(ValueError):
        svc.predict([bad])


def test_predict_empty_raises(synthetic_model):
    svc = synthetic_model["service"]
    with pytest.raises(ValueError):
        svc.predict([])


def test_feature_ordering_is_respected(synthetic_model):
    svc = synthetic_model["service"]
    names = synthetic_model["feature_names"]
    # Out-of-order dict must still produce a correct-shape prediction.
    record = {names[2]: 0.9, names[0]: 0.1, names[1]: 0.2, names[3]: 0.4, names[4]: 0.7}
    result = svc.predict_single(record)
    assert result.feature_names == names


def test_unavailable_service_raises():
    svc = InferenceService()
    assert not svc.is_available
    with pytest.raises(ModelNotLoadedError):
        svc.predict_single({"x": 1.0})
    with pytest.raises(ModelNotLoadedError):
        _ = svc.feature_names
