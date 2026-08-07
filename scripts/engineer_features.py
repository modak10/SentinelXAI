#!/usr/bin/env python
"""Apply Milestone 2 feature engineering to the processed train/val/test splits.

    python scripts/engineer_features.py

Reads data/processed/{train,val,test}.parquet (Milestone 1 output) and
writes data/processed/engineered/{train,val,test}.parquet plus
data/processed/feature_engineering_report.json.

All three transforms (exclude metadata columns, drop redundant/zero-variance
columns, clip invalid values to a fixed floor) are FIT-FREE — the drop list
and clip bounds are constants from configs/features.yaml, not learned from
data — so applying them independently and identically to train/val/test
introduces no leakage. This is different in kind from a later step like
StandardScaler, which must be fit on train only.

Also runs a correlation cross-check (features/selection.py) on the TRAIN
split only (never val/test) to verify the curated drop_columns list
actually resolved every high-correlation pair the Milestone 1 EDA found.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json  # noqa: E402
import time  # noqa: E402

import pandas as pd  # noqa: E402

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.data.eda import numeric_feature_columns  # noqa: E402
from sentinelxai.features.engineering import apply_feature_engineering  # noqa: E402
from sentinelxai.features.selection import find_remaining_correlated_pairs  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger("sentinelxai.features.engineer_features")

SPLITS = ("train", "val", "test")


def main() -> None:
    configure_logging()
    started = time.monotonic()
    cfg = get_config()

    processed_dir = cfg.paths.data_processed_dir
    engineered_dir = processed_dir / "engineered"
    engineered_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, dict] = {}
    train_engineered: pd.DataFrame | None = None

    for split in SPLITS:
        input_path = processed_dir / f"{split}.parquet"
        if not input_path.exists():
            raise FileNotFoundError(
                f"{input_path} not found — run `python scripts/build_dataset.py` first."
            )
        logger.info("Loading %s", input_path)
        df = pd.read_parquet(input_path)

        engineered_df, report = apply_feature_engineering(df, cfg.features)
        reports[split] = report.to_dict()

        output_path = engineered_dir / f"{split}.parquet"
        engineered_df.to_parquet(output_path, index=False)
        logger.info(
            "%s: %d rows, %d -> %d columns, %d value(s) clipped -> %s",
            split,
            len(engineered_df),
            report.columns_before,
            report.columns_after,
            report.total_values_clipped,
            output_path,
        )

        if split == "train":
            train_engineered = engineered_df

    # Correlation cross-check: TRAIN split only, per this project's
    # leakage-avoidance policy (docs/JUDGE_QNA.md Q8) — val/test never read
    # for this kind of analysis.
    assert train_engineered is not None
    feature_columns = numeric_feature_columns(train_engineered, cfg.data.label_column)
    remaining_pairs = find_remaining_correlated_pairs(
        train_engineered, feature_columns, threshold=cfg.features.correlation_removal_threshold
    )

    summary = {
        "correlation_removal_threshold": cfg.features.correlation_removal_threshold,
        "final_feature_count": len(feature_columns),
        "final_feature_columns": feature_columns,
        "remaining_correlated_pairs_after_drop": [
            {"feature_a": a, "feature_b": b, "correlation": r} for a, b, r in remaining_pairs
        ],
        "splits": reports,
    }
    report_path = processed_dir / "feature_engineering_report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Saved feature engineering report to %s", report_path)

    elapsed = time.monotonic() - started
    if remaining_pairs:
        logger.warning(
            "=== Feature engineering complete in %.1fs — %d unresolved correlated pair(s), "
            "see %s ===",
            elapsed,
            len(remaining_pairs),
            report_path,
        )
    else:
        logger.info(
            "=== Feature engineering complete in %.1fs — %d final features, "
            "0 unresolved correlated pairs ===",
            elapsed,
            len(feature_columns),
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Feature engineering run failed")
        raise
