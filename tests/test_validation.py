from __future__ import annotations

import pandas as pd
import pytest

from sentinelxai.data.cleaning import clean_dataset
from sentinelxai.data.validation import (
    SchemaValidationError,
    build_data_quality_report,
    detect_rare_classes,
    validate_labels,
    validate_schema,
)


def test_validate_schema_passes_on_clean_data(raw_like_df, sample_config):
    clean_df, _ = clean_dataset(raw_like_df, sample_config)
    validate_schema(clean_df, sample_config)  # should not raise


def test_validate_schema_raises_when_label_column_missing(sample_config):
    df = pd.DataFrame({"Flow Duration": [1, 2, 3]})
    with pytest.raises(SchemaValidationError, match="Label column"):
        validate_schema(df, sample_config)


def test_validate_schema_raises_on_duplicate_columns(sample_config):
    df = pd.DataFrame([[1, 2, "BENIGN"]], columns=["A", "A", "Label"])
    with pytest.raises(SchemaValidationError, match="duplicate column"):
        validate_schema(df, sample_config)


def test_validate_schema_raises_on_empty_dataframe(sample_config):
    df = pd.DataFrame(columns=["Label"])
    with pytest.raises(SchemaValidationError, match="empty"):
        validate_schema(df, sample_config)


def test_validate_labels_flags_unknown_values():
    df = pd.DataFrame({"Label": ["BENIGN", "BENIGN", "NotARealAttack"]})
    class_counts, unknown = validate_labels(df, "Label")
    assert class_counts == {"BENIGN": 2, "NotARealAttack": 1}
    assert unknown == ["NotARealAttack"]


def test_validate_labels_no_unknowns_for_known_taxonomy():
    df = pd.DataFrame({"Label": ["BENIGN", "DDoS", "Heartbleed"]})
    _, unknown = validate_labels(df, "Label")
    assert unknown == []


def test_detect_rare_classes_floor_boundary():
    counts = {"BENIGN": 1000, "AtFloor": 20, "BelowFloor": 19}
    rare = detect_rare_classes(counts, floor=20)
    assert rare == {"BelowFloor": 19}


def test_build_data_quality_report_never_drops_rare_classes(sample_config, raw_like_df):
    clean_df, cleaning_report = clean_dataset(raw_like_df, sample_config)
    class_counts, unknown_labels = validate_labels(clean_df, sample_config.data.label_column)
    rare_classes = detect_rare_classes(class_counts, sample_config.data.split.rare_class_floor)

    report = build_data_quality_report(
        cfg=sample_config,
        cleaning_report=cleaning_report,
        class_counts=class_counts,
        unknown_labels=unknown_labels,
        rare_classes=rare_classes,
    )

    assert "Heartbleed" in report["label_summary"]["class_counts"]
    assert report["label_summary"]["class_counts"]["Heartbleed"] == 1
    assert "Heartbleed" in report["rare_classes"]["classes_below_floor"]
