"""Decision Intelligence package (Phase 5).

Converts a model prediction + confidence into operational decision support:
confidence banding, a domain-policy risk level, deterministic investigation
recommendations, alert prioritization, and failure flagging. The module is
pure (no model dependency) so every function is unit-testable in isolation.
"""

from __future__ import annotations

from .config import DecisionConfig, load_decision_config
from .policy import (
    build_decision_payload,
    confidence_band,
    estimate_confidence,
    is_failure_candidate,
    operational_risk,
    rank_alerts,
    recommendations_for,
    risk_rank,
)

__all__ = [
    "DecisionConfig",
    "load_decision_config",
    "estimate_confidence",
    "confidence_band",
    "operational_risk",
    "risk_rank",
    "recommendations_for",
    "is_failure_candidate",
    "rank_alerts",
    "build_decision_payload",
]
