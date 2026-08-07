"""Column pruning and invalid-value repair for the model-ready feature matrix.

Two operations, both entirely config-driven (``configs/features.yaml`` ->
:class:`~sentinelxai.config.FeatureEngineeringConfig`) — nothing here is
hardcoded, and nothing here modifies data silently: every drop and every
clip is counted and returned in a :class:`FeatureEngineeringReport`.

1. **Column removal** — structural metadata (``exclude_columns``) and
   redundant/zero-variance features (``drop_columns``), per the Milestone 2
   Feature Audit (see docs/DATASET_GUIDE.md).
2. **Clipping** — repairs physically-impossible negative values
   (durations/rates/header-lengths cannot be negative) to a configured
   floor. Rows are never dropped for this — see the Project Lead's
   rare-class-preservation decision; Heartbleed alone has 4 of its 8 train
   samples affected, so dropping rows here would gut an already-critical
   class.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sentinelxai.config import ClipRule, FeatureEngineeringConfig
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureEngineeringReport:
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    columns_excluded: list[str]
    columns_excluded_missing: list[str]
    columns_dropped: list[str]
    columns_dropped_missing: list[str]
    values_clipped_per_column: dict[str, int]

    @property
    def total_values_clipped(self) -> int:
        return sum(self.values_clipped_per_column.values())

    def to_dict(self) -> dict:
        return {
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "columns_excluded": self.columns_excluded,
            "columns_excluded_missing": self.columns_excluded_missing,
            "columns_dropped": self.columns_dropped,
            "columns_dropped_missing": self.columns_dropped_missing,
            "values_clipped_per_column": self.values_clipped_per_column,
            "total_values_clipped": self.total_values_clipped,
        }


def _drop_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Drop `columns` from `df`, returning (result, names not found).

    Missing names are reported, not raised on — a column already absent
    (e.g. re-running against a differently-shaped frame) is a warning, not
    a fatal error, but it must never pass silently either.
    """
    present = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        logger.warning("Configured columns not found, skipping: %s", missing)
    return df.drop(columns=present), missing


def clip_invalid_values(
    df: pd.DataFrame, clip_columns: tuple[ClipRule, ...]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip each configured column to its floor, returning (result, counts).

    `counts` maps column name -> number of values that were below the
    floor (and therefore changed) — always present for every configured
    column, including 0 if nothing needed clipping, so the report is
    complete rather than silent about columns with no issues.
    """
    df = df.copy()
    counts: dict[str, int] = {}
    for rule in clip_columns:
        if rule.name not in df.columns:
            logger.warning("Clip column not found, skipping: %s", rule.name)
            continue
        below_floor = df[rule.name] < rule.min_value
        n_clipped = int(below_floor.sum())
        counts[rule.name] = n_clipped
        if n_clipped:
            df[rule.name] = df[rule.name].clip(lower=rule.min_value)
            logger.info(
                "Clipped %d value(s) in %r to floor %.1f", n_clipped, rule.name, rule.min_value
            )
    return df, counts


def apply_feature_engineering(
    df: pd.DataFrame, cfg: FeatureEngineeringConfig
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    """Apply the full, config-driven feature engineering pipeline.

    Order: exclude structural metadata -> drop redundant/zero-variance
    columns -> clip invalid values on what remains. Never drops rows —
    asserted at the end, not just assumed.
    """
    rows_before = len(df)
    columns_before = df.shape[1]

    sentinel_overlap = set(cfg.sentinel_preserve_columns) & {c.name for c in cfg.clip_columns}
    if sentinel_overlap:
        raise ValueError(
            f"sentinel_preserve_columns overlaps clip_columns at runtime: {sentinel_overlap}. "
            "This should have been caught at config load time."
        )

    df, excluded_missing = _drop_columns(df, cfg.exclude_columns)
    df, dropped_missing = _drop_columns(df, cfg.drop_columns)
    df, clip_counts = clip_invalid_values(df, cfg.clip_columns)

    report = FeatureEngineeringReport(
        rows_before=rows_before,
        rows_after=len(df),
        columns_before=columns_before,
        columns_after=df.shape[1],
        columns_excluded=list(cfg.exclude_columns),
        columns_excluded_missing=excluded_missing,
        columns_dropped=list(cfg.drop_columns),
        columns_dropped_missing=dropped_missing,
        values_clipped_per_column=clip_counts,
    )

    if report.rows_after != report.rows_before:
        raise AssertionError(
            "Feature engineering must never drop rows — "
            f"went from {report.rows_before} to {report.rows_after}"
        )

    logger.info(
        "Feature engineering: %d -> %d columns (excluded %d, dropped %d), %d value(s) clipped",
        report.columns_before,
        report.columns_after,
        len(report.columns_excluded) - len(report.columns_excluded_missing),
        len(report.columns_dropped) - len(report.columns_dropped_missing),
        report.total_values_clipped,
    )
    return df, report
