"""Explainability package (Phase 4).

Wraps TreeSHAP to produce local (per-prediction) and global (dataset-level)
feature-attribution explanations for the LightGBM intrusion-detection model.
All per-prediction output is derived strictly from SHAP values — never
fabricated (per CLAUDE.md / docs/JUDGE_QNA.md Q... "do not invent
explanations").
"""

from __future__ import annotations

from .human_explanation import humanize_contributions
from .shap_engine import (
    GlobalFeatureImportance,
    LocalContribution,
    LocalExplanation,
    SHAPExplainer,
)

__all__ = [
    "SHAPExplainer",
    "LocalExplanation",
    "LocalContribution",
    "GlobalFeatureImportance",
    "humanize_contributions",
]
