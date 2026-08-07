#!/usr/bin/env python
"""Exploratory data analysis over the processed TRAIN split only.

    python scripts/run_eda.py

Reads data/processed/train.parquet (produced by build_dataset.py) and
writes, under reports/ (config-driven, see configs/config.yaml):

    eda_summary.md              human-readable findings
    eda_feature_stats.csv       describe() + skew per feature
    figures/class_distribution.png
    figures/correlation_heatmap.png

Deliberately never touches val/test — see eda.py's module docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless — no display available in this environment
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.data.eda import (  # noqa: E402
    class_distribution,
    correlation_matrix,
    feature_summary_statistics,
    highly_correlated_pairs,
    numeric_feature_columns,
    zero_variance_features,
)
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger("sentinelxai.data.run_eda")

HIGH_CORR_THRESHOLD = 0.95


def _plot_class_distribution(dist: pd.Series, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    dist.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xscale("log")
    ax.set_xlabel("Row count (log scale)")
    ax.set_title("CICIDS2017 — Class Distribution (train split)")
    ax.invert_yaxis()
    fig.tight_layout()
    path = figures_dir / "class_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_correlation_heatmap(corr: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=6)
    ax.set_title("Feature Correlation Matrix (train split)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    path = figures_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _write_summary_md(
    *,
    path: Path,
    n_rows: int,
    n_features: int,
    dist: pd.Series,
    zero_var: list[str],
    high_corr_pairs: list[tuple[str, str, float]],
) -> None:
    lines = [
        "# EDA Summary (train split)",
        "",
        f"- Rows: {n_rows:,}",
        f"- Numeric feature columns: {n_features}",
        f"- Classes: {len(dist)}",
        "",
        "## Class Distribution",
        "",
        "| Class | Count | % of train |",
        "|---|---|---|",
    ]
    total = int(dist.sum())
    for label, count in dist.items():
        lines.append(f"| {label} | {count:,} | {100 * count / total:.4f}% |")

    lines += [
        "",
        "## Zero-Variance Features",
        "",
        (
            "None found."
            if not zero_var
            else "These features are constant in the train split and carry no "
            "signal — candidates to drop in Milestone 2:\n\n"
            + "\n".join(f"- {f}" for f in zero_var)
        ),
        "",
        f"## Highly Correlated Feature Pairs (|r| >= {HIGH_CORR_THRESHOLD})",
        "",
    ]
    if not high_corr_pairs:
        lines.append("None found.")
    else:
        lines.append("| Feature A | Feature B | r |")
        lines.append("|---|---|---|")
        for a, b, r in high_corr_pairs[:40]:
            lines.append(f"| {a} | {b} | {r} |")
        if len(high_corr_pairs) > 40:
            lines.append(f"\n...and {len(high_corr_pairs) - 40} more (see eda_feature_stats.csv).")

    lines += [
        "",
        "## Notes",
        "",
        "- Computed on the TRAIN split only, per this project's own "
        "leakage-avoidance policy (docs/JUDGE_QNA.md Q8) — val/test were not read.",
        "- Severe class imbalance confirmed (see docs/DATASET_GUIDE.md and "
        "data/processed/data_quality_report.json) — informs the Macro-F1 metric "
        "choice for Milestone 2.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    cfg = get_config()

    train_path = cfg.paths.data_processed_dir / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run `python scripts/build_dataset.py` first."
        )

    logger.info("Loading %s", train_path)
    train_df = pd.read_parquet(train_path)

    label_column = cfg.data.label_column
    feature_columns = numeric_feature_columns(train_df, label_column)
    logger.info("Analyzing %d rows, %d numeric feature columns", len(train_df), len(feature_columns))

    dist = class_distribution(train_df, label_column)
    stats = feature_summary_statistics(train_df, feature_columns)
    zero_var = zero_variance_features(train_df, feature_columns)
    corr = correlation_matrix(train_df, feature_columns)
    high_corr_pairs = highly_correlated_pairs(corr, threshold=HIGH_CORR_THRESHOLD)

    if zero_var:
        logger.warning("Zero-variance features found: %s", zero_var)
    logger.info("%d feature pairs with |r| >= %.2f", len(high_corr_pairs), HIGH_CORR_THRESHOLD)

    reports_dir = cfg.paths.reports_dir
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats.to_csv(reports_dir / "eda_feature_stats.csv")
    class_path = _plot_class_distribution(dist, figures_dir)
    corr_path = _plot_correlation_heatmap(corr, figures_dir)
    _write_summary_md(
        path=reports_dir / "eda_summary.md",
        n_rows=len(train_df),
        n_features=len(feature_columns),
        dist=dist,
        zero_var=zero_var,
        high_corr_pairs=high_corr_pairs,
    )

    logger.info("EDA complete. Wrote %s, %s, and eda_summary.md/eda_feature_stats.csv", class_path, corr_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("EDA run failed")
        raise
