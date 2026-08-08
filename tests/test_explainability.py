"""Tests for the SHAP explainability module (Phase 4)."""

from __future__ import annotations

import pandas as pd

from sentinelxai.explainability import SHAPExplainer, humanize_contributions


def test_global_importance_ranked(synthetic_model):
    explainer = synthetic_model["explainer"]
    X = pd.DataFrame(
        [{f: 0.5 for f in synthetic_model["feature_names"]}],
        columns=synthetic_model["feature_names"],
    )
    imp = explainer.global_importance(X)
    assert len(imp) == len(synthetic_model["feature_names"])
    ranks = [i.rank for i in imp]
    assert ranks == sorted(ranks)
    assert imp[0].rank == 1


def test_explain_single_returns_contributions(synthetic_model):
    explainer = synthetic_model["explainer"]
    record = {f: 0.5 for f in synthetic_model["feature_names"]}
    local = explainer.explain_single(record, predicted_class="PortScan")
    assert local.predicted_class == "PortScan"
    assert 1 <= len(local.top_contributions) <= explainer.n_top
    # Top contributor has the largest absolute SHAP value.
    vals = [abs(c.value) for c in local.top_contributions]
    assert vals == sorted(vals, reverse=True)


def test_humanize_mentions_class(synthetic_model):
    explainer = synthetic_model["explainer"]
    record = {f: 0.5 for f in synthetic_model["feature_names"]}
    local = explainer.explain_single(record, predicted_class="Bot")
    sentences = humanize_contributions(local)
    assert sentences  # non-empty
    assert any("Bot" in s for s in sentences)


def test_explainer_requires_positive_n_top():
    import pytest

    with pytest.raises(ValueError):
        SHAPExplainer.from_model(object(), ["a"], n_top=0)
