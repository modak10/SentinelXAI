from __future__ import annotations

import pandas as pd

from sentinelxai.data.eda import correlation_matrix
from sentinelxai.features.selection import (
    find_remaining_correlated_pairs,
    select_redundant_columns,
)


def _perfectly_correlated_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "A_dup": [2, 4, 6, 8, 10],  # r=1.0 with A
            "B": [5, 3, 1, 4, 2],
            "Independent": [7, 1, 9, 2, 5],
        }
    )


def test_select_redundant_columns_drops_one_side_of_perfect_pair():
    df = _perfectly_correlated_df()
    corr = correlation_matrix(df, list(df.columns))
    dropped = select_redundant_columns(corr, threshold=0.95)
    assert dropped == ["A_dup"]  # second column of the pair, default policy


def test_select_redundant_columns_respects_prefer_keep():
    df = _perfectly_correlated_df()
    corr = correlation_matrix(df, list(df.columns))
    dropped = select_redundant_columns(corr, threshold=0.95, prefer_keep=frozenset({"A_dup"}))
    assert dropped == ["A"]  # prefer_keep flips which side is dropped


def test_select_redundant_columns_no_pairs_above_threshold():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [3, 1, 2], "C": [2, 3, 1]})
    corr = correlation_matrix(df, list(df.columns))
    assert select_redundant_columns(corr, threshold=0.95) == []


def test_find_remaining_correlated_pairs_detects_unresolved_pair():
    df = _perfectly_correlated_df()
    # deliberately do NOT drop "A_dup" -- simulates an incomplete config
    remaining = find_remaining_correlated_pairs(df, list(df.columns), threshold=0.95)
    labels = {(a, b) for a, b, _ in remaining}
    assert ("A", "A_dup") in labels


def test_find_remaining_correlated_pairs_empty_after_resolving():
    df = _perfectly_correlated_df().drop(columns=["A_dup"])
    remaining = find_remaining_correlated_pairs(df, list(df.columns), threshold=0.95)
    assert remaining == []
