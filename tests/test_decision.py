"""Tests for the decision intelligence module (Phase 5)."""

from __future__ import annotations


from sentinelxai.decision import (
    build_decision_payload,
    confidence_band,
    estimate_confidence,
    is_failure_candidate,
    load_decision_config,
    operational_risk,
    rank_alerts,
    recommendations_for,
)
from sentinelxai.explainability.shap_engine import LocalContribution


def _cfg():
    return load_decision_config()


def test_estimate_confidence_is_max_prob():
    probs = [0.1, 0.7, 0.2]
    assert abs(estimate_confidence(probs) - 0.7) < 1e-9


def test_confidence_band_thresholds():
    cfg = _cfg()
    assert confidence_band(0.95, cfg) == "HIGH"
    assert confidence_band(0.75, cfg) == "MEDIUM"
    assert confidence_band(0.55, cfg) == "LOW"
    assert confidence_band(0.10, cfg) == "VERY LOW"


def test_operational_risk_base_and_escalation():
    cfg = _cfg()
    # BENIGN at high confidence stays LOW.
    assert operational_risk("BENIGN", 0.99, cfg) == "LOW"
    # Bot at high confidence is CRITICAL.
    assert operational_risk("Bot", 0.99, cfg) == "CRITICAL"
    # PortScan (base MEDIUM) at low confidence escalates one level to HIGH.
    assert operational_risk("PortScan", 0.30, cfg) == "HIGH"
    # PortScan at high confidence stays MEDIUM.
    assert operational_risk("PortScan", 0.99, cfg) == "MEDIUM"


def test_recommendations_present():
    cfg = _cfg()
    recs = recommendations_for("Bot", cfg)
    assert isinstance(recs, list) and recs
    # Unknown class falls back to empty (no crash).
    assert recommendations_for("NONEXISTENT", cfg) == []


def test_is_failure_candidate():
    cfg = _cfg()
    assert is_failure_candidate(0.40, cfg) is True
    assert is_failure_candidate(0.90, cfg) is False


def test_rank_alerts_orders_by_risk_then_confidence():
    cfg = _cfg()
    alerts = [
        {"predicted_class": "PortScan", "risk": "LOW", "confidence": 0.9},
        {"predicted_class": "Bot", "risk": "CRITICAL", "confidence": 0.6},
        {"predicted_class": "DDoS", "risk": "HIGH", "confidence": 0.8},
    ]
    ranked = rank_alerts(alerts, cfg)
    assert ranked[0]["priority"] == 1
    assert ranked[0]["risk"] == "CRITICAL"
    assert all("priority" in a for a in ranked)


def test_build_decision_payload_shape():
    cfg = _cfg()
    contribs = [LocalContribution("SYN Flag Count", 0.4, "increase", 1, 0.6)]
    payload = build_decision_payload(
        predicted_class="Bot",
        confidence=0.92,
        top_contributions=contribs,
        cfg=cfg,
        human_explanation=["Elevated SYN flag count increased probability of Bot."],
    )
    assert payload["predicted_class"] == "Bot"
    assert payload["risk"] == "CRITICAL"
    assert payload["confidence_band"] == "HIGH"
    assert not payload["is_failure_candidate"]
    assert payload["recommendations"]
    assert payload["top_features"][0]["name"] == "SYN Flag Count"
    assert "human_explanation" in payload
