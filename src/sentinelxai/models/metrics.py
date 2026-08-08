"""Classification evaluation: metrics, confusion matrix, classification report.

Everything here operates on already-predicted labels (`y_true`/`y_pred`) —
it has no knowledge of *how* a model was trained, which is what makes it
reusable across Logistic Regression, Random Forest, XGBoost, and (in
Milestone 4) LightGBM without duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass
class EvaluationReport:
    model_name: str
    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    classification_report: dict
    confusion_matrix: list = field(repr=False)
    label_names: list[str]
    training_time_seconds: float
    inference_time_seconds: float
    inference_time_per_sample_ms: float
    n_train_samples: int
    n_eval_samples: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "precision_weighted": self.precision_weighted,
            "recall_macro": self.recall_macro,
            "recall_weighted": self.recall_weighted,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
            "label_names": self.label_names,
            "training_time_seconds": self.training_time_seconds,
            "inference_time_seconds": self.inference_time_seconds,
            "inference_time_per_sample_ms": self.inference_time_per_sample_ms,
            "n_train_samples": self.n_train_samples,
            "n_eval_samples": self.n_eval_samples,
        }


def build_evaluation_report(
    *,
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    training_time_seconds: float,
    inference_time_seconds: float,
    n_train_samples: int,
) -> EvaluationReport:
    """Compute every metric this milestone requires from one (y_true, y_pred) pair.

    `label_names` must be in encoder-index order (label_names[i] is the
    class whose encoded value is i) — this is what makes the confusion
    matrix and classification report's class labels line up correctly
    with the encoded predictions.
    """
    n_eval = len(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_dict = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(label_names))))

    return EvaluationReport(
        model_name=model_name,
        accuracy=accuracy,
        precision_macro=float(precision_macro),
        precision_weighted=float(precision_weighted),
        recall_macro=float(recall_macro),
        recall_weighted=float(recall_weighted),
        f1_macro=float(f1_macro),
        f1_weighted=float(f1_weighted),
        classification_report=report_dict,
        confusion_matrix=cm.tolist(),
        label_names=label_names,
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
        inference_time_per_sample_ms=1000 * inference_time_seconds / n_eval if n_eval else 0.0,
        n_train_samples=n_train_samples,
        n_eval_samples=n_eval,
    )


def save_confusion_matrix_plot(
    cm: list | np.ndarray, label_names: list[str], path: Path, title: str
) -> None:
    """Row-normalized (per true-class recall) confusion matrix, annotated with raw counts.

    Row-normalized coloring is used instead of raw counts because BENIGN
    alone is ~83% of the data (see docs/DATASET_GUIDE.md) — a raw-count
    heatmap would just show one saturated row and 14 invisible ones. Cell
    text still shows the exact count for precision.
    """
    cm_array = np.array(cm)
    row_sums = cm_array.sum(axis=1, keepdims=True)
    normalized = np.divide(cm_array, row_sums, out=np.zeros_like(cm_array, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=8)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Recall (row-normalized)")

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            count = int(cm_array[i, j])
            if count:
                text_color = "white" if normalized[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center", color=text_color, fontsize=6)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_classification_report_txt(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], path: Path) -> None:
    """Human-readable sklearn classification_report, as a standalone .txt artifact."""
    report_str = classification_report(y_true, y_pred, target_names=label_names, zero_division=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_str, encoding="utf-8")


def build_comparison_table(
    results: dict[str, dict], *, title: str = "# Baseline Model Comparison (Milestone 3)"
) -> str:
    """Build the model-comparison markdown from a {model_name: metrics_dict} map.

    `metrics_dict` is whatever :meth:`EvaluationReport.to_dict` produces —
    this works equally whether the caller just finished training in-process
    or loaded the JSON files back off disk (e.g. because each model was
    trained in its own separate script invocation, as Milestone 3's
    incremental validation required). One table-building implementation,
    reused across milestones (Milestone 3's baseline-only comparison and
    Milestone 4's 4-way baselines-vs-LightGBM comparison), matching the
    "reusable pipeline" requirement — `title` is the only thing that varies.
    """
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["f1_macro"])
    lines = [
        title,
        "",
        "Evaluated on the VAL split (never test — see docs/JUDGE_QNA.md Q8).",
        "",
        "| Model | Accuracy | Precision (M/W) | Recall (M/W) | F1 (M/W) | Train (s) | Infer (ms/sample) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, r in ranked:
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | {r['precision_macro']:.4f} / {r['precision_weighted']:.4f} "
            f"| {r['recall_macro']:.4f} / {r['recall_weighted']:.4f} "
            f"| {r['f1_macro']:.4f} / {r['f1_weighted']:.4f} "
            f"| {r['training_time_seconds']:.1f} | {r['inference_time_per_sample_ms']:.4f} |"
        )
    lines += [
        "",
        f"**Ranked by Macro F1** (this project's primary metric, per docs/JUDGE_QNA.md Q12 — "
        f"macro treats every class equally regardless of size, which matters given BENIGN is "
        f"~83% of the data): **{ranked[0][0]}** leads at {ranked[0][1]['f1_macro']:.4f}.",
    ]
    return "\n".join(lines)


def build_pairwise_verdict(
    results: dict[str, dict], *, challenger: str, baseline: str, metric: str = "f1_macro"
) -> str:
    """Explicit "does X surpass Y" verdict markdown for two models in a results map.

    Used by Milestone 4 to state clearly whether LightGBM surpasses the
    XGBoost baseline (rather than leaving the reader to infer it from a
    ranked table) — generic over which two models and which metric decide
    the verdict, so it is not tied to this one comparison.
    """
    if challenger not in results or baseline not in results:
        missing = {challenger, baseline} - set(results)
        return (
            f"**Verdict unavailable**: metrics for {sorted(missing)} are required "
            "to compare them, and at least one is missing."
        )

    challenger_metrics, baseline_metrics = results[challenger], results[baseline]
    delta = challenger_metrics[metric] - baseline_metrics[metric]
    verb = "surpasses" if delta > 0 else ("matches" if delta == 0 else "falls short of")

    lines = [
        f"**{challenger} {verb} {baseline} on {metric}** "
        f"({challenger_metrics[metric]:.4f} vs {baseline_metrics[metric]:.4f}, "
        f"{'+' if delta >= 0 else ''}{delta:.4f}).",
        "",
        f"| Metric | {baseline} | {challenger} | Delta |",
        "|---|---|---|---|",
    ]
    for label, key in [
        ("Accuracy", "accuracy"),
        ("Macro Precision", "precision_macro"),
        ("Macro Recall", "recall_macro"),
        ("Macro F1", "f1_macro"),
        ("Weighted F1", "f1_weighted"),
        ("Training Time (s)", "training_time_seconds"),
        ("Inference (ms/sample)", "inference_time_per_sample_ms"),
    ]:
        baseline_v, challenger_v = baseline_metrics[key], challenger_metrics[key]
        row_delta = challenger_v - baseline_v
        lines.append(
            f"| {label} | {baseline_v:.4f} | {challenger_v:.4f} "
            f"| {'+' if row_delta >= 0 else ''}{row_delta:.4f} |"
        )

    return "\n".join(lines)
