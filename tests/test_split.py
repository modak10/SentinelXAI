from __future__ import annotations

import pandas as pd

from sentinelxai.data.split import stratified_split


def test_split_preserves_every_row(raw_like_df, sample_config):
    result = stratified_split(raw_like_df, "Label", sample_config.data.split, seed=42)
    total = len(result.train) + len(result.val) + len(result.test)
    assert total == len(raw_like_df)


def test_split_preserves_every_class_no_dropping(raw_like_df, sample_config):
    result = stratified_split(raw_like_df, "Label", sample_config.data.split, seed=42)
    combined_labels = set(result.train["Label"]) | set(result.val["Label"]) | set(result.test["Label"])
    assert combined_labels == set(raw_like_df["Label"].unique())


def test_split_rare_class_goes_entirely_to_train(raw_like_df, sample_config):
    """Heartbleed has 1 sample in the fixture -> n < 3 fallback -> all train."""
    result = stratified_split(raw_like_df, "Label", sample_config.data.split, seed=42)
    assert (result.train["Label"] == "Heartbleed").sum() == 1
    assert (result.val["Label"] == "Heartbleed").sum() == 0
    assert (result.test["Label"] == "Heartbleed").sum() == 0

    fallback_labels = {c["label"] for c in result.report["fallback_classes"]}
    assert "Heartbleed" in fallback_labels


def test_split_majority_class_is_reasonably_proportional(sample_config):
    df = pd.DataFrame({"Label": ["BENIGN"] * 100, "x": range(100)})
    result = stratified_split(df, "Label", sample_config.data.split, seed=42)
    assert len(result.train) == 70
    assert len(result.val) == 15
    assert len(result.test) == 15


def test_split_is_deterministic_given_same_seed(raw_like_df, sample_config):
    r1 = stratified_split(raw_like_df, "Label", sample_config.data.split, seed=7)
    r2 = stratified_split(raw_like_df, "Label", sample_config.data.split, seed=7)
    pd.testing.assert_frame_equal(
        r1.train.sort_values("Label").reset_index(drop=True),
        r2.train.sort_values("Label").reset_index(drop=True),
    )


def test_split_no_row_appears_in_two_splits(raw_like_df, sample_config):
    # Use a column that is unique per source row before any resets, by
    # tagging the fixture with a synthetic unique id first.
    df = raw_like_df.copy()
    df["_row_id"] = range(len(df))
    result = stratified_split(df, "Label", sample_config.data.split, seed=42)
    train_ids = set(result.train["_row_id"])
    val_ids = set(result.val["_row_id"])
    test_ids = set(result.test["_row_id"])
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
