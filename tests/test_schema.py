from __future__ import annotations

import pandas as pd

from sentinelxai.config import DuplicateColumnRule
from sentinelxai.data.schema import (
    KNOWN_LABELS,
    clean_column_names,
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
