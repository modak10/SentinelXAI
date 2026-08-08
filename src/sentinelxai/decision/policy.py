"""Decision policy functions (Phase 5).

Pure, deterministic transforms from (predicted class, confidence) into the
operational layers the SOC analyst consumes. No model or SHAP dependency — the
caller supplies those — so every function here is independently testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from sentinelxai.explainability.shap_engine import LocalContribution
from sentinelxai.logging_setup import get_logger

from .config import DecisionConfig

logger = get_logger(__name__)


def estimate_confidence(probabilities: list[float] | np.ndarray) -> float:
    """Model confidence = maximum class probability (softmax max)."""
    arr = np.asarray(probabilities, dtype=float)
    if arr.size == 0:
        raise ValueError("probabilities must be non-empty")
    return float(np.max(arr))


def confidence_band(confidence: float, cfg: DecisionConfig) -> str:
    """Band a confidence score into HIGH / MEDIUM / LOW / VERY LOW."""
    if confidence >= cfg.confidence.high:
        return "HIGH"
    if confidence >= cfg.confidence.medium:
        return "MEDIUM"
    if confidence >= cfg.confidence.low:
        return "LOW"
    return "VERY LOW"


def risk_rank(name: str, cfg: DecisionConfig) -> int:
    """Numeric rank for a risk name (higher = more severe). Unknown -> 0."""
    return cfg.risk_rank_map.get(name, 0)


def operational_risk(attack_class: str, confidence: float, cfg: DecisionConfig) -> str:
    """Map (attack class, confidence) to an operational risk level.

    The base severity comes from the documented per-attack ``attack_severity``
    policy. When confidence is low and the class is not BENIGN, the risk is
    escalated one level — a low-certainty attack alert is treated more
    cautiously (per the spec's risk-modulation policy).
    """
    base = cfg.attack_severity.get(attack_class)
    if base is None:
        logger.warning("No severity policy for class %r; defaulting to LOW", attack_class)
        base = "LOW"

    if (
        attack_class != "BENIGN" or not cfg.modulation.benign_never_escalate
    ) and confidence < cfg.modulation.escalate_below_confidence:
        base_rank = risk_rank(base, cfg)
        # Escalate to the next higher-ranked level that exists in the config.
        higher = [lvl for lvl in cfg.risk_levels if lvl.rank > base_rank]
        if higher:
            higher_sorted = sorted(higher, key=lambda level: level.rank)
            return higher_sorted[0].name
    return base


def recommendations_for(attack_class: str, cfg: DecisionConfig) -> list[str]:
    """Deterministic, rule-based investigation guidance for an attack class."""
    return list(cfg.recommendations.get(attack_class, []))


def is_failure_candidate(confidence: float, cfg: DecisionConfig) -> bool:
    """Flag low-confidence predictions for the Failure Explorer."""
    return confidence < cfg.modulation.failure_confidence_threshold


def rank_alerts(alerts: list[dict[str, Any]], cfg: DecisionConfig) -> list[dict[str, Any]]:
    """Rank alerts by risk severity, then confidence (descending).

    Each input dict must contain ``risk`` (level name) and ``confidence``
    (float). Returns a new list with a ``priority`` integer (1 = top) added.
    Stable sort preserves original order for ties.
    """
    decorated = []
    for idx, alert in enumerate(alerts):
        decorated.append(
            (
                risk_rank(alert.get("risk", "LOW"), cfg),
                float(alert.get("confidence", 0.0)),
                idx,
                alert,
            )
        )
    decorated.sort(key=lambda t: (t[0], t[1], -t[2]), reverse=True)
    ranked: list[dict[str, Any]] = []
    for priority, (_, _, _, alert) in enumerate(decorated, start=1):
        item = dict(alert)
        item["priority"] = priority
        ranked.append(item)
    return ranked


@dataclass
class DecisionPayload:
    """Assembled decision support for one prediction."""

    predicted_class: str
    confidence: float
    confidence_band: str
    risk: str
    risk_rank: int
    is_failure_candidate: bool
    recommendations: list[str]
    top_features: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_decision_payload(
    *,
    predicted_class: str,
    confidence: float,
    top_contributions: list[LocalContribution],
    cfg: DecisionConfig,
    human_explanation: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble the full decision-support payload for one prediction.

    Combines confidence banding, operational risk, recommendations, and failure
    flagging. ``top_contributions`` are the SHAP-derived features (already
    computed by the caller); ``human_explanation`` is optional text.
    """
    band = confidence_band(confidence, cfg)
    risk = operational_risk(predicted_class, confidence, cfg)
    payload = DecisionPayload(
        predicted_class=predicted_class,
        confidence=confidence,
        confidence_band=band,
        risk=risk,
        risk_rank=risk_rank(risk, cfg),
        is_failure_candidate=is_failure_candidate(confidence, cfg),
        recommendations=recommendations_for(predicted_class, cfg),
        top_features=[asdict(c) for c in top_contributions],
    )
    out = payload.to_dict()
    if human_explanation is not None:
        out["human_explanation"] = human_explanation
    return out
