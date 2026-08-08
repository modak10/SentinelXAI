"""Decision-config loader (Phase 5).

Parses ``configs/decision.yaml`` into typed dataclasses. Keeping the policy in
YAML (rather than code) satisfies CLAUDE.md's "Never hardcode values" rule and
lets a security analyst revise risk mappings without touching Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sentinelxai.config import CONFIGS_DIR
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_DECISION_CONFIG_PATH = CONFIGS_DIR / "decision.yaml"


@dataclass(frozen=True)
class RiskLevel:
    name: str
    rank: int


@dataclass(frozen=True)
class ConfidenceThresholds:
    high: float
    medium: float
    low: float


@dataclass(frozen=True)
class RiskModulation:
    escalate_below_confidence: float
    benign_never_escalate: bool
    failure_confidence_threshold: float


@dataclass(frozen=True)
class DecisionConfig:
    risk_levels: tuple[RiskLevel, ...]
    risk_rank_map: dict[str, int] = field(repr=False)
    attack_severity: dict[str, str] = field(repr=False)
    confidence: ConfidenceThresholds = field(repr=False)
    modulation: RiskModulation = field(repr=False)
    recommendations: dict[str, list[str]] = field(repr=False)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Decision config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_decision_config(path: Path | None = None) -> DecisionConfig:
    """Load and validate ``configs/decision.yaml``."""
    raw = _load_yaml(path or DEFAULT_DECISION_CONFIG_PATH)

    levels = tuple(
        RiskLevel(name=lvl["name"], rank=int(lvl["rank"])) for lvl in raw["risk_levels"]
    )
    rank_map = {lvl.name: lvl.rank for lvl in levels}

    conf_raw = raw["confidence"]
    mod_raw = raw["risk_modulation"]

    return DecisionConfig(
        risk_levels=levels,
        risk_rank_map=rank_map,
        attack_severity=dict(raw["attack_severity"]),
        confidence=ConfidenceThresholds(
            high=float(conf_raw["high"]),
            medium=float(conf_raw["medium"]),
            low=float(conf_raw["low"]),
        ),
        modulation=RiskModulation(
            escalate_below_confidence=float(mod_raw["escalate_below_confidence"]),
            benign_never_escalate=bool(mod_raw["benign_never_escalate"]),
            failure_confidence_threshold=float(mod_raw["failure_confidence_threshold"]),
        ),
        recommendations={k: list(v) for k, v in raw.get("recommendations", {}).items()},
    )
