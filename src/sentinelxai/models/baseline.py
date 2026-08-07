"""Reusable baseline-model training/evaluation harness.

:func:`train_and_evaluate_baseline` is model-agnostic — it fits, times,
predicts, evaluates, and saves every artifact (model, metrics JSON,
confusion matrix PNG, classification report TXT) for any already-built
scikit-learn/XGBoost-compatible estimator. Built once, called three times
(Logistic Regression, Random Forest, XGBoost) with zero duplication.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sentinelxai.config import LogisticRegressionConfig, RandomForestConfig, XGBoostConfig
from sentinelxai.logging_setup import get_logger
from sentinelxai.models.metrics import (
    EvaluationReport,
    build_evaluation_report,
    save_classification_report_txt,
    save_confusion_matrix_plot,
)

logger = get_logger(__name__)


@dataclass
class BaselineResult:
    report: EvaluationReport
    model_path: Path
    metrics_path: Path
    confusion_matrix_path: Path
    classification_report_path: Path


def build_logistic_regression(cfg: LogisticRegressionConfig, seed: int) -> LogisticRegression:
    return LogisticRegression(
        max_iter=cfg.max_iter,
        C=cfg.C,
        solver=cfg.solver,
        class_weight=cfg.class_weight,
        random_state=seed,
    )


def build_random_forest(cfg: RandomForestConfig, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        n_jobs=cfg.n_jobs,
        class_weight=cfg.class_weight,
        random_state=seed,
    )


def build_xgboost(cfg: XGBoostConfig, num_class: int, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        tree_method=cfg.tree_method,
        n_jobs=cfg.n_jobs,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=seed,
    )


def train_and_evaluate_baseline(
    *,
    model_name: str,
    model,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    label_names: list[str],
    models_dir: Path,
    reports_dir: Path,
    sample_weight: np.ndarray | None = None,
) -> BaselineResult:
    """Fit, time, predict, evaluate, and save everything for one model.

    Evaluation is on VAL, never TEST — test stays untouched until a single
    final model is chosen (Milestone 4), per this project's leakage policy
    (docs/JUDGE_QNA.md Q8).
    """
    logger.info("Training %s on %d rows, %d features", model_name, len(X_train), X_train.shape[1])

    start = time.monotonic()
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    training_time = time.monotonic() - start

    start = time.monotonic()
    y_pred = model.predict(X_val)
    inference_time = time.monotonic() - start

    report = build_evaluation_report(
        model_name=model_name,
        y_true=y_val,
        y_pred=y_pred,
        label_names=label_names,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        n_train_samples=len(X_train),
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / f"{model_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)

    cm_path = reports_dir / "figures" / f"{model_name}_confusion_matrix.png"
    save_confusion_matrix_plot(
        report.confusion_matrix, label_names, cm_path, f"{model_name} — Confusion Matrix (val)"
    )

    classification_report_path = reports_dir / f"{model_name}_classification_report.txt"
    save_classification_report_txt(y_val, y_pred, label_names, classification_report_path)

    logger.info(
        "%s: accuracy=%.4f macro_f1=%.4f weighted_f1=%.4f train=%.1fs infer=%.3fs (%.4fms/sample)",
        model_name,
        report.accuracy,
        report.f1_macro,
        report.f1_weighted,
        training_time,
        inference_time,
        report.inference_time_per_sample_ms,
    )

    return BaselineResult(
        report=report,
        model_path=model_path,
        metrics_path=metrics_path,
        confusion_matrix_path=cm_path,
        classification_report_path=classification_report_path,
    )
