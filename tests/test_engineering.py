from __future__ import annotations

import pandas as pd
import pytest

from sentinelxai.config import ClipRule, FeatureEngineeringConfig
from sentinelxai.features.engineering import (
    apply_feature_engineering,
    clip_invalid_values,
)


@pytest.fixture
def engineering_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Flow Duration": [10, -5, 20, -1, 0],       # 2 below floor (0)
            "Flow Bytes/s": [100.0, 200.0, -50.0, 0.0, -1.0],  # 2 below floor (0)
            "Init_Win_bytes_forward": [-1, -1, 100, 200, -1],  # sentinel, never clipped
            "Redundant Feature": [1, 2, 3, 4, 5],        # to be dropped
            "__source_file": ["a.csv"] * 5,               # to be excluded
            "Label": ["BENIGN", "DDoS", "BENIGN", "DDoS", "BENIGN"],
        }
    )


@pytest.fixture
def fe_config() -> FeatureEngineeringConfig:
    return FeatureEngineeringConfig(
        exclude_columns=("__source_file",),
        drop_columns=("Redundant Feature",),
        clip_columns=(
            ClipRule(name="Flow Duration", min_value=0.0),
            ClipRule(name="Flow Bytes/s", min_value=0.0),
        ),
        sentinel_preserve_columns=("Init_Win_bytes_forward",),
        correlation_removal_threshold=0.95,
    )


# --- clip_invalid_values ---


def test_clip_invalid_values_clips_below_floor(engineering_df, fe_config):
    result, counts = clip_invalid_values(engineering_df, fe_config.clip_columns)
    assert (result["Flow Duration"] >= 0).all()
    assert (result["Flow Bytes/s"] >= 0).all()
    assert counts == {"Flow Duration": 2, "Flow Bytes/s": 2}


def test_clip_invalid_values_reports_zero_for_clean_column():
    df = pd.DataFrame({"Clean": [1.0, 2.0, 3.0]})
    rules = (ClipRule(name="Clean", min_value=0.0),)
    result, counts = clip_invalid_values(df, rules)
    assert counts == {"Clean": 0}
    pd.testing.assert_series_equal(result["Clean"], df["Clean"])


def test_clip_invalid_values_never_drops_rows(engineering_df, fe_config):
    result, _ = clip_invalid_values(engineering_df, fe_config.clip_columns)
    assert len(result) == len(engineering_df)


def test_clip_invalid_values_preserves_sentinel_column_untouched(engineering_df, fe_config):
    """Init_Win_bytes_forward is NOT in clip_columns -- its -1 sentinel
    values must survive unchanged.
    """
    result, _ = clip_invalid_values(engineering_df, fe_config.clip_columns)
    pd.testing.assert_series_equal(
        result["Init_Win_bytes_forward"], engineering_df["Init_Win_bytes_forward"]
    )


def test_clip_invalid_values_does_not_mutate_input(engineering_df, fe_config):
    original = engineering_df.copy()
    clip_invalid_values(engineering_df, fe_config.clip_columns)
    pd.testing.assert_frame_equal(engineering_df, original)


# --- apply_feature_engineering (full pipeline) ---


def test_apply_feature_engineering_drops_and_excludes_columns(engineering_df, fe_config):
    result, report = apply_feature_engineering(engineering_df, fe_config)
    assert "Redundant Feature" not in result.columns
    assert "__source_file" not in result.columns
    assert report.columns_dropped == ["Redundant Feature"]
    assert report.columns_excluded == ["__source_file"]


def test_apply_feature_engineering_clips_and_reports_counts(engineering_df, fe_config):
    result, report = apply_feature_engineering(engineering_df, fe_config)
    assert (result["Flow Duration"] >= 0).all()
    assert report.values_clipped_per_column == {"Flow Duration": 2, "Flow Bytes/s": 2}
    assert report.total_values_clipped == 4


def test_apply_feature_engineering_never_drops_rows(engineering_df, fe_config):
    result, report = apply_feature_engineering(engineering_df, fe_config)
    assert len(result) == len(engineering_df)
    assert report.rows_before == report.rows_after == len(engineering_df)


def test_apply_feature_engineering_preserves_sentinel_values(engineering_df, fe_config):
    result, _ = apply_feature_engineering(engineering_df, fe_config)
    assert (result["Init_Win_bytes_forward"] == -1).sum() == 3  # unchanged from fixture


def test_apply_feature_engineering_reports_missing_configured_columns(fe_config):
    """A column named in config but absent from the actual DataFrame must
    be reported, not silently ignored or fatal.
    """
    df = pd.DataFrame({"Label": ["BENIGN", "DDoS"]})  # none of the configured columns present
    result, report = apply_feature_engineering(df, fe_config)
    assert set(report.columns_dropped_missing) == {"Redundant Feature"}
    assert set(report.columns_excluded_missing) == {"__source_file"}
    assert len(result) == 2  # rows untouched


def test_apply_feature_engineering_rejects_clip_sentinel_overlap(engineering_df):
    bad_config = FeatureEngineeringConfig(
        exclude_columns=(),
        drop_columns=(),
        clip_columns=(ClipRule(name="Flow Duration", min_value=0.0),),
        sentinel_preserve_columns=("Flow Duration",),  # overlaps clip_columns
        correlation_removal_threshold=0.95,
    )
    with pytest.raises(ValueError, match="overlaps clip_columns"):
        apply_feature_engineering(engineering_df, bad_config)
