"""Leakage-aware stratified train/val/test split.

Per the Project Lead's explicit decision: rare classes are NEVER dropped.
Every class — however small — is preserved end-to-end and, where a
standard 70/15/15 stratified split is mathematically impossible, this
module applies a documented fallback and records exactly which classes
needed it, rather than silently discarding rows.

Fallback rules, applied per class of size ``n``:

- ``n < 3``            -> all rows go to train; the class has no val/test
                           representation, which is unavoidable and reported.
- proportional 70/15/15 would give 0 rows to val or test (typical for
  classes below ``rare_class_floor``) -> reserve exactly 1 row for val and
  1 for test, remainder to train.
- otherwise            -> standard proportional split (rounded, with the
                           remainder assigned to train so counts sum to n).

This is a self-contained implementation (no scikit-learn dependency) so it
can encode the fallback precisely — ``sklearn.model_selection.
train_test_split`` raises rather than degrading gracefully for classes
smaller than the number of splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sentinelxai.config import SplitConfig
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    report: dict


def _allocate_counts(n: int, split_cfg: SplitConfig) -> tuple[int, int, int, str | None]:
    """Return (train_count, val_count, test_count, fallback_reason)."""
    if n < 3:
        return n, 0, 0, (
            f"fewer than 3 samples ({n}); all assigned to train, "
            "none available for val/test evaluation"
        )

    train_count = round(n * split_cfg.train)
    val_count = round(n * split_cfg.val)
    test_count = n - train_count - val_count  # forces exact sum == n

    if val_count < 1 or test_count < 1:
        val_count = max(val_count, 1)
        test_count = max(test_count, 1)
        train_count = n - val_count - test_count
        return train_count, val_count, test_count, (
            f"below rare_class_floor ({split_cfg.rare_class_floor}); proportional "
            "70/15/15 would have given 0 samples to val and/or test, so 1 sample "
            "each was reserved for val and test instead"
        )

    if n < split_cfg.rare_class_floor:
        return train_count, val_count, test_count, (
            f"below rare_class_floor ({split_cfg.rare_class_floor}) but proportional "
            "split still yielded >=1 sample in every split"
        )

    return train_count, val_count, test_count, None


def stratified_split(
    df: pd.DataFrame, label_column: str, split_cfg: SplitConfig, seed: int
) -> SplitResult:
    """Split `df` into train/val/test, stratified by `label_column`.

    Deterministic given `seed`. Every row in `df` ends up in exactly one
    output split; no rows and no classes are dropped.
    """
    rng = np.random.default_rng(seed)
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    per_class_allocation: dict[str, dict] = {}
    fallback_classes: list[dict] = []

    for label, group in df.groupby(label_column, sort=False):
        n = len(group)
        idx = group.index.to_numpy().copy()
        rng.shuffle(idx)

        train_count, val_count, test_count, reason = _allocate_counts(n, split_cfg)

        train_idx = idx[:train_count]
        val_idx = idx[train_count : train_count + val_count]
        test_idx = idx[train_count + val_count : train_count + val_count + test_count]

        train_parts.append(df.loc[train_idx])
        val_parts.append(df.loc[val_idx])
        test_parts.append(df.loc[test_idx])

        per_class_allocation[str(label)] = {
            "total": n,
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        }
        if reason:
            fallback_classes.append({"label": str(label), "n": n, "reason": reason})
            logger.warning("Split fallback applied for class %r (n=%d): %s", label, n, reason)

    train_df = pd.concat(train_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        "Split complete: train=%d val=%d test=%d (%d classes needed a fallback rule)",
        len(train_df),
        len(val_df),
        len(test_df),
        len(fallback_classes),
    )

    report = {
        "seed": seed,
        "target_ratios": {
            "train": split_cfg.train,
            "val": split_cfg.val,
            "test": split_cfg.test,
        },
        "actual_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "per_class_allocation": per_class_allocation,
        "fallback_classes": fallback_classes,
    }
    return SplitResult(train=train_df, val=val_df, test=test_df, report=report)
