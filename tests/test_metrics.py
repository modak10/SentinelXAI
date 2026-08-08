from __future__ import annotations

import numpy as np

from sentinelxai.models.metrics import (
    build_comparison_table,
    build_evaluation_report,
    build_pairwise_verdict,
    save_confusion_matrix_plot,
)


def test_build_evaluation_report_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    report = build_evaluation_report(
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
        label_names=["A", "B", "C"],
        training_time_seconds=1.0,
        inference_time_seconds=0.5,
        n_train_samples=100,
    )
    assert report.accuracy == 1.0
    assert report.f1_macro == 1.0
    assert report.f1_weighted == 1.0
    assert report.precision_macro == 1.0
    assert report.recall_macro == 1.0


def test_build_evaluation_report_confusion_matrix_shape_and_diagonal():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])  # one BENIGN-like row misclassified
    report = build_evaluation_report(
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
        label_names=["A", "B", "C"],
        training_time_seconds=1.0,
        inference_time_seconds=0.5,
        n_train_samples=100,
    )
    cm = np.array(report.confusion_matrix)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)
    assert cm[2, 2] == 1  # class C perfectly predicted


def test_build_evaluation_report_inference_time_per_sample():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    report = build_evaluation_report(
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
        label_names=["A", "B"],
        training_time_seconds=1.0,
        inference_time_seconds=0.004,
        n_train_samples=100,
    )
    assert np.isclose(report.inference_time_per_sample_ms, 1.0)  # 4ms / 4 samples = 1ms/sample


def test_build_evaluation_report_classification_report_has_per_class_entries():
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    report = build_evaluation_report(
        model_name="test_model",
        y_true=y_true,
        y_pred=y_pred,
        label_names=["Alpha", "Beta", "Gamma"],
        training_time_seconds=1.0,
        inference_time_seconds=0.5,
        n_train_samples=100,
    )
    assert "Alpha" in report.classification_report
    assert "Beta" in report.classification_report
    assert "Gamma" in report.classification_report
    assert "macro avg" in report.classification_report


def test_build_evaluation_report_to_dict_is_json_serializable():
    import json

    report = build_evaluation_report(
        model_name="test_model",
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        label_names=["A", "B"],
        training_time_seconds=1.0,
        inference_time_seconds=0.5,
        n_train_samples=100,
    )
    json.dumps(report.to_dict())  # raises if anything isn't serializable


def test_save_confusion_matrix_plot_creates_file(tmp_path):
    cm = [[10, 2], [1, 15]]
    path = tmp_path / "figures" / "cm.png"
    save_confusion_matrix_plot(cm, ["A", "B"], path, "Test Confusion Matrix")
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_confusion_matrix_plot_handles_zero_row_sum(tmp_path):
    """A class with zero true samples in this eval batch must not raise
    (division by zero in row-normalization must be guarded).
    """
    cm = [[10, 0], [0, 0]]  # class B has zero support
    path = tmp_path / "cm.png"
    save_confusion_matrix_plot(cm, ["A", "B"], path, "Zero-row test")
    assert path.exists()


# --- build_comparison_table ---


def _fake_metrics(f1_macro: float, training_time: float = 10.0) -> dict:
    return {
        "accuracy": 0.9,
        "precision_macro": 0.5,
        "precision_weighted": 0.9,
        "recall_macro": 0.5,
        "recall_weighted": 0.9,
        "f1_macro": f1_macro,
        "f1_weighted": 0.9,
        "training_time_seconds": training_time,
        "inference_time_per_sample_ms": 0.01,
    }


def test_build_comparison_table_ranks_by_macro_f1_descending():
    results = {
        "logistic_regression": _fake_metrics(f1_macro=0.60),
        "random_forest": _fake_metrics(f1_macro=0.82),
        "xgboost": _fake_metrics(f1_macro=0.79),
    }
    table = build_comparison_table(results)
    rf_pos = table.index("random_forest")
    xgb_pos = table.index("xgboost")
    lr_pos = table.index("logistic_regression")
    assert rf_pos < xgb_pos < lr_pos  # best macro F1 listed first


def test_build_comparison_table_names_the_leader_in_recommendation_line():
    results = {"a": _fake_metrics(0.5), "b": _fake_metrics(0.9)}
    table = build_comparison_table(results)
    assert "**b** leads at 0.9000" in table


def test_build_comparison_table_includes_every_model():
    results = {"a": _fake_metrics(0.5), "b": _fake_metrics(0.6), "c": _fake_metrics(0.7)}
    table = build_comparison_table(results)
    assert "| a |" in table
    assert "| b |" in table
    assert "| c |" in table


def test_build_comparison_table_accepts_custom_title():
    results = {"a": _fake_metrics(0.5)}
    table = build_comparison_table(results, title="# Custom Title")
    assert table.startswith("# Custom Title")


# --- build_pairwise_verdict ---


def test_build_pairwise_verdict_reports_surpasses_when_challenger_higher():
    results = {
        "lightgbm": _fake_metrics(f1_macro=0.90),
        "xgboost": _fake_metrics(f1_macro=0.85),
    }
    verdict = build_pairwise_verdict(results, challenger="lightgbm", baseline="xgboost")
    assert "lightgbm surpasses xgboost" in verdict
    assert "0.9000 vs 0.8500" in verdict


def test_build_pairwise_verdict_reports_falls_short_when_challenger_lower():
    results = {
        "lightgbm": _fake_metrics(f1_macro=0.80),
        "xgboost": _fake_metrics(f1_macro=0.85),
    }
    verdict = build_pairwise_verdict(results, challenger="lightgbm", baseline="xgboost")
    assert "lightgbm falls short of xgboost" in verdict


def test_build_pairwise_verdict_reports_matches_when_equal():
    results = {
        "lightgbm": _fake_metrics(f1_macro=0.85),
        "xgboost": _fake_metrics(f1_macro=0.85),
    }
    verdict = build_pairwise_verdict(results, challenger="lightgbm", baseline="xgboost")
    assert "lightgbm matches xgboost" in verdict


def test_build_pairwise_verdict_handles_missing_model():
    results = {"lightgbm": _fake_metrics(f1_macro=0.90)}
    verdict = build_pairwise_verdict(results, challenger="lightgbm", baseline="xgboost")
    assert "Verdict unavailable" in verdict
    assert "xgboost" in verdict


def test_build_pairwise_verdict_includes_full_metric_table():
    results = {
        "lightgbm": _fake_metrics(f1_macro=0.90, training_time=50.0),
        "xgboost": _fake_metrics(f1_macro=0.85, training_time=120.0),
    }
    verdict = build_pairwise_verdict(results, challenger="lightgbm", baseline="xgboost")
    assert "| Training Time (s) | 120.0000 | 50.0000 | -70.0000 |" in verdict
