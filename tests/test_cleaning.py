from __future__ import annotations

import numpy as np

from sentinelxai.data.cleaning import clean_dataset


def test_clean_dataset_replaces_infinity_and_drops_resulting_nan_rows(
    raw_like_df, sample_config
):
    n_infinite_cells = int(
        np.isinf(raw_like_df[["Flow Bytes/s", "Flow Packets/s"]].to_numpy()).sum()
    )
    assert n_infinite_cells == 2  # sanity check on the fixture itself

    clean_df, report = clean_dataset(raw_like_df, sample_config)

    assert report.infinity_values_replaced == 2
    assert not np.isinf(clean_df[["Flow Bytes/s", "Flow Packets/s"]].to_numpy()).any()
    assert clean_df.isna().sum().sum() == 0
    # the row with ±Infinity had no other valid values to recover -> dropped
    assert report.rows_dropped_nan == 1


def test_clean_dataset_drops_exact_duplicates(raw_like_df, sample_config):
    _, report = clean_dataset(raw_like_df, sample_config)
    assert report.rows_dropped_duplicate == 1


def test_clean_dataset_normalizes_corrupted_labels(raw_like_df, sample_config):
    clean_df, report = clean_dataset(raw_like_df, sample_config)
    assert "Web Attack - Brute Force" in clean_df["Label"].unique()
    assert not clean_df["Label"].str.contains("�").any()
    assert report.label_values_normalized == 1


def test_clean_dataset_preserves_rare_class(raw_like_df, sample_config):
    clean_df, _ = clean_dataset(raw_like_df, sample_config)
    assert (clean_df["Label"] == "Heartbleed").sum() == 1  # never dropped


def test_clean_dataset_row_accounting_is_exact(raw_like_df, sample_config):
    """rows_before - dropped_nan - dropped_duplicate == rows_after."""
    clean_df, report = clean_dataset(raw_like_df, sample_config)
    assert (
        report.rows_before - report.rows_dropped_nan - report.rows_dropped_duplicate
        == report.rows_after
        == len(clean_df)
    )
