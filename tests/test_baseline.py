"""Integration tests for the reusable baseline training/evaluation harness.

Uses a tiny synthetic dataset and a fast estimator (LogisticRegression with
few iterations) so the suite stays fast — this tests that
train_and_evaluate_baseline wires fitting, timing, evaluation, and artifact
saving together correctly, not that any particular model performs well.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from sentinelxai.config import LogisticRegressionConfig, RandomForestConfig, XGBoostConfig
from sentinelxai.models.baseline import (
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    train_and_evaluate_baseline,
)


@pytest.fixture
def tiny_dataset():
    """Every class must appear in BOTH train and val — a naive first-N/last-N
    slice of a class-sorted array (the original version of this fixture)
    silently gave XGBoost a train set missing class 2 entirely, which
    surfaced as a confusing "mix of binary and multilabel-indicator
    targets" error in classification_report rather than an obvious one.
    """
    rng = np.random.default_rng(42)
    n_per_class, n_train_per_class = 20, 15
    X_train_parts, y_train_parts, X_val_parts, y_val_parts = [], [], [], []
    for label in (0, 1, 2):
        X_c = pd.DataFrame(
            {"feature_a": rng.normal(size=n_per_class), "feature_b": rng.normal(size=n_per_class)}
        )
        X_train_parts.append(X_c.iloc[:n_train_per_class])
        y_train_parts.append(np.full(n_train_per_class, label))
        X_val_parts.append(X_c.iloc[n_train_per_class:])
        y_val_parts.append(np.full(n_per_class - n_train_per_class, label))
    X_train = pd.concat(X_train_parts, ignore_index=True)
    y_train = np.concatenate(y_train_parts)
    X_val = pd.concat(X_val_parts, ignore_index=True)
    y_val = np.concatenate(y_val_parts)
    return X_train, y_train, X_val, y_val


def test_train_and_evaluate_baseline_saves_all_four_artifacts(tmp_path, tiny_dataset):
    X_train, y_train, X_val, y_val = tiny_dataset
    model = LogisticRegression(max_iter=50)

    result = train_and_evaluate_baseline(
        model_name="tiny_lr",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_names=["A", "B", "C"],
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    assert result.model_path.exists()
    assert result.metrics_path.exists()
    assert result.confusion_matrix_path.exists()
    assert result.classification_report_path.exists()


def test_train_and_evaluate_baseline_report_has_positive_timings(tmp_path, tiny_dataset):
    X_train, y_train, X_val, y_val = tiny_dataset
    model = LogisticRegression(max_iter=50)

    result = train_and_evaluate_baseline(
        model_name="tiny_lr",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_names=["A", "B", "C"],
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    assert result.report.training_time_seconds > 0
    assert result.report.inference_time_seconds >= 0
    assert result.report.n_train_samples == len(X_train)
    assert result.report.n_eval_samples == len(X_val)


def test_train_and_evaluate_baseline_saved_model_is_usable(tmp_path, tiny_dataset):
    """The persisted .joblib must be a working, re-loadable model, not just a file."""
    import joblib

    X_train, y_train, X_val, y_val = tiny_dataset
    model = LogisticRegression(max_iter=50)

    result = train_and_evaluate_baseline(
        model_name="tiny_lr",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_names=["A", "B", "C"],
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )

    loaded = joblib.load(result.model_path)
    preds = loaded.predict(X_val)
    assert len(preds) == len(X_val)


def test_train_and_evaluate_baseline_accepts_sample_weight(tmp_path, tiny_dataset):
    """XGBoost's call path passes sample_weight -- verify the harness forwards it."""
    from xgboost import XGBClassifier

    X_train, y_train, X_val, y_val = tiny_dataset
    model = XGBClassifier(n_estimators=5, max_depth=2, objective="multi:softprob", num_class=3)
    sample_weight = np.ones(len(y_train))

    result = train_and_evaluate_baseline(
        model_name="tiny_xgb",
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        label_names=["A", "B", "C"],
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        sample_weight=sample_weight,
    )
    assert result.model_path.exists()


# --- model builders (config -> estimator wiring) ---


def test_build_logistic_regression_applies_config():
    cfg = LogisticRegressionConfig(max_iter=77, C=0.5, solver="lbfgs", class_weight="balanced")
    model = build_logistic_regression(cfg, seed=42)
    assert model.max_iter == 77
    assert model.C == 0.5
    assert model.class_weight == "balanced"
    assert model.random_state == 42


def test_build_random_forest_applies_config():
    cfg = RandomForestConfig(
        n_estimators=33, max_depth=7, min_samples_leaf=3, n_jobs=1, class_weight="balanced_subsample"
    )
    model = build_random_forest(cfg, seed=42)
    assert model.n_estimators == 33
    assert model.max_depth == 7
    assert model.class_weight == "balanced_subsample"


def test_build_xgboost_applies_config_and_num_class():
    cfg = XGBoostConfig(n_estimators=22, max_depth=4, learning_rate=0.2, tree_method="hist", n_jobs=1)
    model = build_xgboost(cfg, num_class=15, seed=42)
    assert model.n_estimators == 22
    assert model.max_depth == 4
    assert model.get_params()["num_class"] == 15
