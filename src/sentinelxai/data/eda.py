"""Exploratory data analysis helpers.

Pure computation only (no plotting, no file I/O) so it is unit-testable —
see ``tests/test_eda.py``. Rendering and report-writing live in
``scripts/run_eda.py``.

EDA is deliberately run against the **train split only** (see
``scripts/run_eda.py``), never val/test — inspecting test data during
exploration is itself a (subtle) form of leakage this project's own
Judge Q&A commits to avoiding (see docs/JUDGE_QNA.md Q8).
"""

from __future__ import annotations

import pandas as pd


def numeric_feature_columns(df: pd.DataFrame, label_column: str) -> list[str]:
    """All numeric, non-label, non-metadata columns eligible for EDA."""
    exclude = {label_column}
    return [
        c
        for c in df.columns
        if c not in exclude and not c.startswith("__") and pd.api.types.is_numeric_dtype(df[c])
    ]


def class_distribution(df: pd.DataFrame, label_column: str) -> pd.Series:
    """Class counts, sorted descending — the shape judges will ask about (Q6/Q13)."""
    return df[label_column].value_counts().sort_values(ascending=False)


def feature_summary_statistics(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """describe() plus skew, giving a fuller picture than mean/std alone."""
    stats = df[feature_columns].describe().T
    stats["skew"] = df[feature_columns].skew()
    return stats


def zero_variance_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    """Features that are constant across the training split — dead weight
    for a model and worth flagging before Milestone 2 feature engineering.
    """
    stds = df[feature_columns].std()
    return stds[stds == 0].index.tolist()


def correlation_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    return df[feature_columns].corr()


def highly_correlated_pairs(
    corr: pd.DataFrame, threshold: float = 0.95
) -> list[tuple[str, str, float]]:
    """Feature pairs above `threshold` absolute correlation — redundancy
    candidates for Milestone 2 feature selection, not acted on here.
    """
    pairs: list[tuple[str, str, float]] = []
    columns = corr.columns
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append((col_a, col_b, round(float(value), 4)))
    return sorted(pairs, key=lambda p: -abs(p[2]))
