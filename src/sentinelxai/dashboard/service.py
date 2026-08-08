"""Shared dashboard service (Phase 6).

Loads the same inference/SHAP/decision singletons the API uses, cached via
``st.cache_resource`` so the (potentially large) model is loaded once per
Streamlit session. Every page calls :func:`get_service` and checks
``available`` before using the model.
"""

from __future__ import annotations


import streamlit as st

from sentinelxai.config import get_config
from sentinelxai.database.store import SQLiteStore
from sentinelxai.decision import load_decision_config
from sentinelxai.explainability import SHAPExplainer
from sentinelxai.logging_setup import configure_logging
from sentinelxai.models.inference import InferenceService


@st.cache_resource
def get_service():
    configure_logging()
    cfg = get_config()
    try:
        inference = InferenceService.from_config(cfg)
    except FileNotFoundError:
        inference = InferenceService()

    explainer = (
        SHAPExplainer.from_model(inference._model, inference.feature_names)
        if inference.is_available
        else None
    )
    decision_cfg = load_decision_config()
    store = SQLiteStore(cfg.paths.data_processed_dir / "predictions.db")
    return {
        "cfg": cfg,
        "inference": inference,
        "explainer": explainer,
        "decision_cfg": decision_cfg,
        "store": store,
        "available": inference.is_available,
    }
