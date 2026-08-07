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


_MANGLE_SUFFIX_RE = re.compile(r"^(?P<base>.+)\.(?P<n>\d+)$")


def _base_name(name: str, known_names: set[str]) -> str:
    """Return the pre-mangle base name for a pandas-deduped column label.

    ``pandas`` silently renames the 2nd+ occurrence of a duplicate CSV
    header from ``"X"`` to ``"X.1"``, ``"X.2"``, ... during ``read_csv`` —
    *before* any of our code ever sees the column names. A later exact
    string match against the configured name ``"X"`` therefore misses
    ``"X.1"`` entirely, which is exactly how the ``Fwd Header Length``
    duplicate silently survived cleaning (see docs/DATASET_GUIDE.md).

    ``"X.N"`` is only treated as a mangled duplicate of ``"X"`` when ``"X"``
    is *also* present among the columns — that combination is how pandas'
    mangling always looks, so a genuinely distinct column that happens to
    be named e.g. ``"Metric.2"`` (with no sibling literally named
    ``"Metric"``) is never mistaken for a duplicate.
    """
    match = _MANGLE_SUFFIX_RE.match(name)
    if match and match.group("base") in known_names:
        return match.group("base")
    return name


def find_duplicate_column_groups(columns: list[str]) -> dict[str, list[int]]:
    """Group column positions by pre-mangle base name.

    Only groups with more than one column are returned. Works for both
    pandas-mangled duplicates (``"X"`` / ``"X.1"`` / ``"X.2"``) and literal
    duplicate labels (two columns both literally named ``"X"``, e.g. in a
    hand-built DataFrame in a test) uniformly.
    """
    known = set(columns)
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(columns):
        base = _base_name(name, known)
        groups.setdefault(base, []).append(i)
    return {base: positions for base, positions in groups.items() if len(positions) > 1}


def resolve_duplicate_columns(
    df: pd.DataFrame, rules: tuple[DuplicateColumnRule, ...] = ()
) -> pd.DataFrame:
    """Drop duplicate columns, detected generically — no config required.

    Detection (:func:`find_duplicate_column_groups`) is automatic and
    dataset-agnostic: it catches any number of duplicates of any column,
    including ones pandas has already mangled with a ``.1``/``.2`` suffix,
    without needing each one named in config. ``rules`` is only needed to
    *override* the default "keep the first occurrence" policy for a
    specific base name (pass ``keep="last"``) — most callers pass none.
    """
    duplicate_groups = find_duplicate_column_groups(list(df.columns))
    if not duplicate_groups:
        return df

    keep_policy = {rule.name: rule.keep for rule in rules}

    # Drop by integer position, not by label: the label may be duplicated,
    # so df.drop(columns=[name]) would remove *every* column sharing that
    # name — including the one we mean to keep.
    new_columns = list(df.columns)
    drop_positions: set[int] = set()
    for base, positions in duplicate_groups.items():
        policy = keep_policy.get(base, "first")
        keep_index = positions[0] if policy == "first" else positions[-1]
        drop_positions.update(p for p in positions if p != keep_index)
        # Normalize away any ".N" mangling on the survivor, so the kept
        # column's name is always the clean base name regardless of
        # whether "first" or "last" happened to be the mangled one.
        new_columns[keep_index] = base

    keep_positions = [i for i in range(df.shape[1]) if i not in drop_positions]
    result = df.iloc[:, keep_positions]
    result.columns = [new_columns[i] for i in keep_positions]
    return result
