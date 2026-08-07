"""Cleaning stage of the dataset pipeline: infinities, NaNs, duplicates, labels.

Every step records what it changed into a :class:`CleaningReport` so the
pipeline never silently discards data — see
``docs/DATASET_GUIDE.md`` and MLOPS_CHECKLIST.md's "Generate report" step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from sentinelxai.config import AppConfig
from sentinelxai.data.schema import normalize_label
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    rows_before: int
    infinity_values_replaced: int
    rows_dropped_nan: int
    rows_dropped_duplicate: int
    rows_after: int
    label_values_normalized: int

    def to_dict(self) -> dict:
        return asdict(self)


def clean_dataset(df: pd.DataFrame, cfg: AppConfig) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply the documented cleaning pipeline and return (clean_df, report).

    Order matters and mirrors AGENT.md's DATASET pipeline:
    replace ±Infinity -> handle NaN -> drop duplicate rows -> validate labels.
    """
    rows_before = len(df)
    df = df.copy()

    infinity_columns = [c for c in cfg.data.cleaning.infinity_columns if c in df.columns]
    missing_infinity_cols = set(cfg.data.cleaning.infinity_columns) - set(infinity_columns)
    if missing_infinity_cols:
        logger.warning(
            "Configured infinity_columns not found in data, skipping: %s",
            missing_infinity_cols,
        )

    infinity_mask = pd.DataFrame(False, index=df.index, columns=infinity_columns)
    for col in infinity_columns:
        infinity_mask[col] = np.isinf(df[col])
    infinity_values_replaced = int(infinity_mask.to_numpy().sum())
    df[infinity_columns] = df[infinity_columns].replace([np.inf, -np.inf], np.nan)
    logger.info("Replaced %d ±Infinity values with NaN", infinity_values_replaced)

    rows_with_nan = df.isna().any(axis=1)
    rows_dropped_nan = 0
    if cfg.data.cleaning.nan_strategy == "drop_row":
        rows_dropped_nan = int(rows_with_nan.sum())
        df = df.loc[~rows_with_nan].copy()
        logger.info("Dropped %d rows containing NaN values", rows_dropped_nan)
    else:
        raise NotImplementedError(
            f"Unsupported nan_strategy: {cfg.data.cleaning.nan_strategy!r}"
        )

    rows_dropped_duplicate = 0
    if cfg.data.cleaning.drop_duplicate_rows:
        # Duplicates are judged on feature+label columns only, ignoring the
        # loader-added source-file trace column so a flow repeated across
        # files (or within one) is still recognized as a duplicate.
        subset = [c for c in df.columns if not c.startswith("__")]
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        rows_dropped_duplicate = before - len(df)
        logger.info("Dropped %d exact duplicate rows", rows_dropped_duplicate)

    label_col = cfg.data.label_column
    original_labels = df[label_col].copy()
    df[label_col] = df[label_col].astype(str).map(normalize_label)
    label_values_normalized = int((df[label_col] != original_labels).sum())
    if label_values_normalized:
        logger.info("Normalized %d label values (encoding artifact cleanup)", label_values_normalized)

    report = CleaningReport(
        rows_before=rows_before,
        infinity_values_replaced=infinity_values_replaced,
        rows_dropped_nan=rows_dropped_nan,
        rows_dropped_duplicate=rows_dropped_duplicate,
        rows_after=len(df),
        label_values_normalized=label_values_normalized,
    )
    return df, report
