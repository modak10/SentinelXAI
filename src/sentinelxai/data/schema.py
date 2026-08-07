"""Canonical schema for the CICIDS2017 (MachineLearningCVE) dataset.

This module is the single source of truth for column names, the label
taxonomy, and the two concrete data-quality quirks discovered while
auditing the raw files:

1. The header repeats ``Fwd Header Length`` at two column positions with
   identical values (a known CICIDS2017 export artifact).
2. The "Web Attack" labels contain the Unicode replacement character
   (U+FFFD) in place of what was originally an en-dash — a corruption
   baked into the publicly released CSVs themselves, not a read error.

Everything here is exercised by ``tests/test_schema.py``.
"""

from __future__ import annotations

import re

import pandas as pd

from sentinelxai.config import DuplicateColumnRule

#: The complete, normalized label taxonomy found across all 8 raw files.
#: BENIGN plus 14 attack classes.
KNOWN_LABELS: frozenset[str] = frozenset(
    {
        "BENIGN",
        "DDoS",
        "PortScan",
        "Bot",
        "Infiltration",
        "FTP-Patator",
        "SSH-Patator",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "Heartbleed",
        "Web Attack - Brute Force",
        "Web Attack - Sql Injection",
        "Web Attack - XSS",
    }
)

#: Coarse attack-family grouping, used only for EDA summaries — the
#: fine-grained label above remains the ML target in Milestone 2.
ATTACK_FAMILY: dict[str, str] = {
    "BENIGN": "Benign",
    "DDoS": "DoS/DDoS",
    "DoS GoldenEye": "DoS/DDoS",
    "DoS Hulk": "DoS/DDoS",
    "DoS Slowhttptest": "DoS/DDoS",
    "DoS slowloris": "DoS/DDoS",
    "PortScan": "Recon",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
    "FTP-Patator": "Brute Force",
    "SSH-Patator": "Brute Force",
    "Heartbleed": "Web/Exploit",
    "Web Attack - Brute Force": "Web/Exploit",
    "Web Attack - Sql Injection": "Web/Exploit",
    "Web Attack - XSS": "Web/Exploit",
}

_REPLACEMENT_CHAR = "�"
_WHITESPACE_RE = re.compile(r"\s+")


def clean_column_names(columns: list[str]) -> list[str]:
    """Strip stray leading/trailing whitespace from raw header names.

    The raw CSVs are inconsistently padded (e.g. ``" Flow Duration"`` vs
    ``"Total Length of Fwd Packets"`` with no leading space) — this makes
    column access reliable regardless of source-file quirks.
    """
    return [c.strip() for c in columns]


def normalize_label(raw_label: str) -> str:
    """Normalize a raw label value to its canonical form.

    Handles the U+FFFD-corrupted "Web Attack" labels and collapses any
    incidental whitespace. Idempotent: safe to call on an already-clean
    label.
    """
    label = raw_label.strip()
    label = label.replace(_REPLACEMENT_CHAR, "-")
    label = _WHITESPACE_RE.sub(" ", label)
    return label


def resolve_duplicate_columns(
    df: pd.DataFrame, rules: tuple[DuplicateColumnRule, ...]
) -> pd.DataFrame:
    """Drop duplicate-named columns per config, keeping the configured copy.

    ``pandas`` permits non-unique column labels, so a duplicate like
    ``Fwd Header Length`` (present at two raw positions) survives a plain
    ``read_csv`` as two columns sharing one name. This resolves that
    deterministically and explicitly, rather than leaving pandas' default
    "last one wins on ``df[name]`` access" behavior as an implicit bug
    magnet.
    """
    if df.columns.is_unique:
        return df

    # Drop by integer position, not by label: the label is duplicated by
    # definition here, so df.drop(columns=[name]) would remove *every*
    # column sharing that name — including the one we mean to keep.
    drop_positions: set[int] = set()
    for rule in rules:
        positions = [i for i, c in enumerate(df.columns) if c == rule.name]
        if len(positions) < 2:
            continue
        keep_index = positions[0] if rule.keep == "first" else positions[-1]
        drop_positions.update(p for p in positions if p != keep_index)

    if not drop_positions:
        return df

    keep_positions = [i for i in range(df.shape[1]) if i not in drop_positions]
    return df.iloc[:, keep_positions]
