#!/usr/bin/env python
"""One-command dataset pipeline: raw CICIDS2017 CSVs -> processed, split dataset.

    python scripts/build_dataset.py

Pipeline (matches AGENT.md's DATASET section and MLOPS_CHECKLIST.md):

    Merge -> Validate -> Clean -> Remove duplicates -> Replace ±Infinity ->
    Handle NaN -> Validate labels -> Generate report -> Split -> Save

Outputs, all under `data/processed/` (config-driven, see configs/config.yaml):

    train.parquet, val.parquet, test.parquet
    data_quality_report.json
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running as `python scripts/build_dataset.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.data.cleaning import clean_dataset  # noqa: E402
from sentinelxai.data.loader import load_raw_dataset  # noqa: E402
from sentinelxai.data.split import stratified_split  # noqa: E402
from sentinelxai.data.validation import (  # noqa: E402
    build_data_quality_report,
    detect_rare_classes,
    save_report,
    validate_labels,
    validate_schema,
)
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402

logger = get_logger("sentinelxai.data.build_dataset")


def main() -> None:
    configure_logging()
    started = time.monotonic()
    cfg = get_config()

    logger.info("=== SentinelXAI dataset pipeline starting (seed=%d) ===", cfg.random_seed)

    logger.info("Step 1/6: Merge raw files")
    raw_df = load_raw_dataset(cfg)

    logger.info("Step 2/6: Clean (±Infinity, NaN, duplicates, label normalization)")
    clean_df, cleaning_report = clean_dataset(raw_df, cfg)
    del raw_df  # ~800MB+ in memory; release before the split step needs headroom

    logger.info("Step 3/6: Validate schema")
    validate_schema(clean_df, cfg)

    logger.info("Step 4/6: Validate labels")
    class_counts, unknown_labels = validate_labels(clean_df, cfg.data.label_column)
    rare_classes = detect_rare_classes(class_counts, cfg.data.split.rare_class_floor)
    if rare_classes:
        logger.warning(
            "%d class(es) below rare_class_floor=%d: %s",
            len(rare_classes),
            cfg.data.split.rare_class_floor,
            rare_classes,
        )

    logger.info("Step 5/6: Stratified split (train/val/test)")
    split_result = stratified_split(
        clean_df, cfg.data.label_column, cfg.data.split, cfg.random_seed
    )

    logger.info("Step 6/6: Save processed dataset + data quality report")
    cfg.paths.data_processed_dir.mkdir(parents=True, exist_ok=True)
    split_result.train.to_parquet(cfg.paths.data_processed_dir / "train.parquet", index=False)
    split_result.val.to_parquet(cfg.paths.data_processed_dir / "val.parquet", index=False)
    split_result.test.to_parquet(cfg.paths.data_processed_dir / "test.parquet", index=False)

    quality_report = build_data_quality_report(
        cfg=cfg,
        cleaning_report=cleaning_report,
        class_counts=class_counts,
        unknown_labels=unknown_labels,
        rare_classes=rare_classes,
    )
    quality_report["split"] = split_result.report
    save_report(quality_report, cfg.paths.data_processed_dir / "data_quality_report.json")

    elapsed = time.monotonic() - started
    logger.info(
        "=== Pipeline complete in %.1fs: train=%d val=%d test=%d rows, %d classes ===",
        elapsed,
        len(split_result.train),
        len(split_result.val),
        len(split_result.test),
        len(class_counts),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Dataset pipeline failed")
        raise
