from __future__ import annotations

import pandas as pd

from sentinelxai.data.eda import (
    class_distribution,
    correlation_matrix,
    feature_summary_statistics,
    highly_correlated_pairs,
    numeric_feature_columns,
    zero_variance_features,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow Duration": [1, 2, 3, 4, 5, 6],
            "Flow Bytes/s": [10, 20, 30, 40, 50, 60],  # perfectly correlated with duration
            "Constant Feature": [7, 7, 7, 7, 7, 7],
            "__source_file": ["a.csv"] * 6,
            "Label": ["BENIGN", "BENIGN", "DDoS", "DDoS", "DDoS", "BENIGN"],
        }
    )


def test_numeric_feature_columns_excludes_label_and_metadata():
    cols = numeric_feature_columns(_sample_df(), "Label")
    assert set(cols) == {"Flow Duration", "Flow Bytes/s", "Constant Feature"}


def test_class_distribution_sorted_descending():
    dist = class_distribution(_sample_df(), "Label")
    assert dist.index[0] == "BENIGN"
    assert dist.iloc[0] == 3


def test_feature_summary_statistics_has_expected_columns():
    df = _sample_df()
    stats = feature_summary_statistics(df, numeric_feature_columns(df, "Label"))
    assert "mean" in stats.columns
    assert "skew" in stats.columns
    assert "Flow Duration" in stats.index


def test_zero_variance_features_detects_constant_column():
    df = _sample_df()
    result = zero_variance_features(df, numeric_feature_columns(df, "Label"))
    assert result == ["Constant Feature"]


def test_highly_correlated_pairs_finds_perfectly_correlated_features():
    df = _sample_df()
    corr = correlation_matrix(df, numeric_feature_columns(df, "Label"))
    pairs = highly_correlated_pairs(corr, threshold=0.95)
    labels = {(a, b) for a, b, _ in pairs}
    assert ("Flow Duration", "Flow Bytes/s") in labels
