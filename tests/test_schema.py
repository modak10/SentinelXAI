from __future__ import annotations

import pandas as pd

from sentinelxai.config import DuplicateColumnRule
from sentinelxai.data.schema import (
    KNOWN_LABELS,
    clean_column_names,
    find_duplicate_column_groups,
    normalize_label,
    resolve_duplicate_columns,
)


def test_clean_column_names_strips_whitespace():
    raw = [" Destination Port", "Flow Duration ", "Total Fwd Packets"]
    assert clean_column_names(raw) == ["Destination Port", "Flow Duration", "Total Fwd Packets"]


def test_normalize_label_fixes_corrupted_web_attack_label():
    assert normalize_label("Web Attack � Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("Web Attack � Sql Injection") in KNOWN_LABELS
    assert normalize_label("Web Attack � XSS") in KNOWN_LABELS


def test_normalize_label_is_idempotent():
    once = normalize_label("Web Attack � Brute Force")
    twice = normalize_label(once)
    assert once == twice


def test_normalize_label_passes_through_clean_labels():
    assert normalize_label("BENIGN") == "BENIGN"
    assert normalize_label("  DDoS  ") == "DDoS"


def test_resolve_duplicate_columns_keeps_first_and_drops_second():
    df = pd.DataFrame(
        [[1, 2, 3]], columns=["Fwd Header Length", "Other", "Fwd Header Length"]
    )
    rules = (DuplicateColumnRule(name="Fwd Header Length", keep="first"),)

    result = resolve_duplicate_columns(df, rules)

    assert list(result.columns) == ["Fwd Header Length", "Other"]
    assert result["Fwd Header Length"].iloc[0] == 1  # kept the FIRST occurrence's data


def test_resolve_duplicate_columns_noop_when_already_unique():
    df = pd.DataFrame([[1, 2]], columns=["A", "B"])
    result = resolve_duplicate_columns(df, ())
    pd.testing.assert_frame_equal(result, df)


# --- Regression tests for the pandas-mangling bug (docs/DATASET_GUIDE.md) ---
# pd.read_csv silently renames a duplicate header "X" -> "X.1" *before* our
# code sees it, so a check for literal name equality against "X" never
# catches "X.1". These tests exercise exactly that scenario.


def test_resolve_duplicate_columns_catches_pandas_mangled_suffix_with_no_rules():
    """The actual bug: no rule needed, detection must be automatic."""
    df = pd.DataFrame([[1, 2, 3]], columns=["Fwd Header Length", "Other", "Fwd Header Length.1"])

    result = resolve_duplicate_columns(df)  # no rules passed at all

    assert list(result.columns) == ["Fwd Header Length", "Other"]
    assert result["Fwd Header Length"].iloc[0] == 1


def test_resolve_duplicate_columns_handles_three_way_mangled_duplicate():
    df = pd.DataFrame(
        [[1, 2, 3, 4]], columns=["X", "Other", "X.1", "X.2"]
    )
    result = resolve_duplicate_columns(df)
    assert list(result.columns) == ["X", "Other"]
    assert result["X"].iloc[0] == 1  # kept the first occurrence


def test_resolve_duplicate_columns_does_not_false_positive_on_dotted_name():
    """"Metric.2" alone (no sibling column literally named "Metric") must
    NOT be treated as a mangled duplicate — it's just a column name.
    """
    df = pd.DataFrame([[1, 2]], columns=["Metric.2", "Other"])
    result = resolve_duplicate_columns(df)
    pd.testing.assert_frame_equal(result, df)


def test_resolve_duplicate_columns_respects_keep_last_override():
    df = pd.DataFrame([[1, 2, 3]], columns=["Fwd Header Length", "Other", "Fwd Header Length.1"])
    rules = (DuplicateColumnRule(name="Fwd Header Length", keep="last"),)

    result = resolve_duplicate_columns(df, rules)

    assert list(result.columns) == ["Other", "Fwd Header Length"]
    assert result["Fwd Header Length"].iloc[0] == 3  # kept the LAST occurrence's data


def test_find_duplicate_column_groups_returns_only_duplicated_groups():
    columns = ["A", "B", "A.1", "C"]
    groups = find_duplicate_column_groups(columns)
    assert groups == {"A": [0, 2]}
