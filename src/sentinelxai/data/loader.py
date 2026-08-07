"""Raw CICIDS2017 CSV loading and merging.

Reads the 8 daily CSVs listed in ``configs/data.yaml``, normalizes column
names, resolves the known duplicate ``Fwd Header Length`` column, and
concatenates them into a single DataFrame. Raw files under ``data/raw/``
are never modified — this module only reads them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentinelxai.config import AppConfig
from sentinelxai.data.schema import clean_column_names, resolve_duplicate_columns
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)

#: Column added during load to trace each row back to its source day —
#: dropped before the processed dataset is saved (see cleaning.py), but
#: useful for EDA and for a future day-aware (temporal) split strategy.
SOURCE_FILE_COLUMN = "__source_file"


def _load_single_csv(path: Path, encoding: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Raw CICIDS2017 file not found: {path}. "
            "Verify configs/data.yaml `raw_files` matches data/raw/MachineLearningCVE/."
        )
    df = pd.read_csv(path, encoding=encoding, low_memory=False)
    df.columns = clean_column_names(list(df.columns))
    df[SOURCE_FILE_COLUMN] = path.name
    logger.info("Loaded %s (%d rows, %d columns)", path.name, len(df), df.shape[1])
    return df


def load_raw_dataset(cfg: AppConfig) -> pd.DataFrame:
    """Load and concatenate all raw CICIDS2017 CSVs into one DataFrame.

    Column names are stripped and the duplicate ``Fwd Header Length``
    column is resolved per-file (all files share the same header) before
    concatenation, so downstream code always sees one consistent schema.
    """
    frames = [
        _load_single_csv(cfg.paths.data_raw_dir / filename, cfg.data.raw_file_encoding)
        for filename in cfg.data.raw_files
    ]
    frames = [resolve_duplicate_columns(f, cfg.data.duplicate_columns) for f in frames]

    combined = pd.concat(frames, axis=0, ignore_index=True)
    logger.info(
        "Merged %d raw files into %d total rows, %d columns",
        len(frames),
        len(combined),
        combined.shape[1],
    )
    return combined
