"""Schema/label validation and data-quality report generation.

Nothing here mutates the dataset — validation only inspects it and raises
on structural problems (missing label column) while *reporting* (not
silently dropping) softer issues like unknown label values or rare
classes. The rare-class handling follows the Project Lead's explicit
decision: never discard a class, always document it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sentinelxai.config import AppConfig
from sentinelxai.data.cleaning import CleaningReport
from sentinelxai.data.schema import ATTACK_FAMILY, KNOWN_LABELS
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


class SchemaValidationError(Exception):
    """Raised when the dataset is structurally invalid (not just messy)."""


def validate_schema(df: pd.DataFrame, cfg: AppConfig) -> None:
    """Fail fast on structural problems that cleaning cannot fix."""
    if cfg.data.label_column not in df.columns:
        raise SchemaValidationError(
            f"Label column '{cfg.data.label_column}' missing from dataset. "
            f"Columns present: {list(df.columns)}"
        )
    if not df.columns.is_unique:
        duplicated = df.columns[df.columns.duplicated()].tolist()
        raise SchemaValidationError(
            f"Dataset has duplicate column names after cleaning: {duplicated}. "
            "This should have been resolved by schema.resolve_duplicate_columns."
        )
    if df.empty:
        raise SchemaValidationError("Dataset is empty after cleaning — nothing to validate.")


def validate_labels(df: pd.DataFrame, label_column: str) -> tuple[dict[str, int], list[str]]:
    """Return (class_counts, unknown_labels). Unknown labels are reported, not dropped."""
    class_counts = df[label_column].value_counts().to_dict()
    unknown_labels = sorted(set(class_counts) - KNOWN_LABELS)
    if unknown_labels:
        logger.warning(
            "Found %d label value(s) outside the known taxonomy: %s",
            len(unknown_labels),
            unknown_labels,
        )
    return class_counts, unknown_labels


def detect_rare_classes(class_counts: dict[str, int], floor: int) -> dict[str, int]:
    """Classes with fewer rows than `floor` — cannot guarantee a sample in
    every split at 70/15/15. Returned for reporting; never used to drop data.
    """
    return {label: count for label, count in class_counts.items() if count < floor}


def build_data_quality_report(
    *,
    cfg: AppConfig,
    cleaning_report: CleaningReport,
    class_counts: dict[str, int],
    unknown_labels: list[str],
    rare_classes: dict[str, int],
) -> dict[str, Any]:
    total_rows = sum(class_counts.values())
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "CICIDS2017 (MachineLearningCVE)",
        "source_files": list(cfg.data.raw_files),
        "cleaning": cleaning_report.to_dict(),
        "label_summary": {
            "total_rows": total_rows,
            "num_classes": len(class_counts),
            "class_counts": dict(sorted(class_counts.items(), key=lambda kv: -kv[1])),
            "class_percentages": {
                label: round(100 * count / total_rows, 4) for label, count in class_counts.items()
            },
            "attack_family_counts": _family_counts(class_counts),
            "unknown_labels": unknown_labels,
        },
        "rare_classes": {
            "floor": cfg.data.split.rare_class_floor,
            "classes_below_floor": rare_classes,
            "note": (
                "Rare classes are preserved in the dataset and split, never "
                "dropped. See split.py for the documented fallback strategy "
                "applied to each class listed here."
            ),
        },
    }


def _family_counts(class_counts: dict[str, int]) -> dict[str, int]:
    family_counts: dict[str, int] = {}
    for label, count in class_counts.items():
        family = ATTACK_FAMILY.get(label, "Unknown")
        family_counts[family] = family_counts.get(family, 0) + count
    return dict(sorted(family_counts.items(), key=lambda kv: -kv[1]))


def save_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    logger.info("Saved data quality report to %s", path)
