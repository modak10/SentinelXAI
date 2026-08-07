"""Generic, reusable correlation-based redundancy selection.

Two complementary uses:

1. :func:`select_redundant_columns` — a generic greedy algorithm that,
   given a correlation matrix and a threshold, proposes columns to drop so
   no remaining pair exceeds it. Reusable on a future dataset with no
   curated ``configs/features.yaml`` drop list yet.

2. :func:`find_remaining_correlated_pairs` — the practical use in *this*
   project: verify, against the real engineered data, that the manually
   curated ``drop_columns`` list (docs/DATASET_GUIDE.md's Feature Audit)
   actually achieves what the correlation analysis intended. The curated
   list is preferred over blindly applying (1) here because *which* column
   in a pair to keep is an interpretability judgment (e.g. keep
   "SYN Flag Count" over "Fwd PSH Flags" because the former reads better
   in a SHAP plot) that a greedy algorithm has no way to make — but the
   result should still be checked, not just assumed.
"""

from __future__ import annotations

import pandas as pd

from sentinelxai.data.eda import correlation_matrix, highly_correlated_pairs
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


def select_redundant_columns(
    corr: pd.DataFrame, threshold: float, prefer_keep: frozenset[str] = frozenset()
) -> list[str]:
    """Greedily propose columns to drop so no remaining pair exceeds `threshold`.

    For each over-threshold pair (in the matrix's column order), the
    second column is dropped unless it's in `prefer_keep`, in which case
    the first is dropped instead. Deterministic given the same matrix and
    inputs. Does not mutate `corr`.
    """
    to_drop: set[str] = set()
    pairs = highly_correlated_pairs(corr, threshold=threshold)
    for col_a, col_b, _ in pairs:
        if col_a in to_drop or col_b in to_drop:
            continue  # already resolved via another pair in the same cluster
        if col_b in prefer_keep and col_a not in prefer_keep:
            to_drop.add(col_a)
        else:
            to_drop.add(col_b)
    return sorted(to_drop)


def find_remaining_correlated_pairs(
    df: pd.DataFrame, feature_columns: list[str], threshold: float
) -> list[tuple[str, str, float]]:
    """Recompute correlation on `df` and return any pair still >= `threshold`.

    Intended to run *after* the configured `drop_columns` have already
    been applied — an empty result confirms the curated list actually
    resolved every high-correlation pair found during the original audit;
    a non-empty result means either the dataset shifted or the curated
    list missed something, and is worth surfacing rather than assuming
    away.
    """
    corr = correlation_matrix(df, feature_columns)
    remaining = highly_correlated_pairs(corr, threshold=threshold)
    if remaining:
        logger.warning(
            "%d feature pair(s) still >= %.2f correlation after configured drops: %s",
            len(remaining),
            threshold,
            remaining,
        )
    else:
        logger.info("No remaining feature pairs >= %.2f correlation after configured drops", threshold)
    return remaining
