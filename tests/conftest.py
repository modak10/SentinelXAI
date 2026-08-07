"""Shared pytest fixtures.

Tests run against a small synthetic DataFrame, never the full 844MB raw
dataset — that keeps the suite fast and independent of the raw data being
present on disk. The one exception is the `slow`-marked integration test
in test_build_dataset.py, which is skipped by default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinelxai.config import (
    AppConfig,
    CleaningConfig,
    DataConfig,
    DuplicateColumnRule,
    PathsConfig,
    SplitConfig,
)


@pytest.fixture
def sample_config(tmp_path) -> AppConfig:
    """A minimal, fully-valid AppConfig pointed at a scratch directory."""
    paths = PathsConfig(
        data_raw_dir=tmp_path / "raw",
        data_interim_dir=tmp_path / "interim",
        data_processed_dir=tmp_path / "processed",
        models_dir=tmp_path / "models",
        logs_dir=tmp_path / "logs",
        reports_dir=tmp_path / "reports",
    )
    split = SplitConfig(train=0.70, val=0.15, test=0.15, stratify=True, rare_class_floor=20)
    cleaning = CleaningConfig(
        infinity_columns=("Flow Bytes/s", "Flow Packets/s"),
        nan_strategy="drop_row",
        drop_duplicate_rows=True,
    )
    data = DataConfig(
        raw_files=(),
        raw_file_encoding="utf-8",
        label_column="Label",
        duplicate_columns=(DuplicateColumnRule(name="Fwd Header Length", keep="first"),),
        split=split,
        cleaning=cleaning,
        leak_risk_columns=(),
    )
    return AppConfig(
        project_name="SentinelXAI-test",
        project_version="0.0.0",
        random_seed=42,
        log_level="DEBUG",
        paths=paths,
        data=data,
    )


@pytest.fixture
def raw_like_df() -> pd.DataFrame:
    """A small DataFrame mimicking the raw CICIDS2017 quirks:
    - two rows with ±Infinity in Flow Bytes/s (division-by-zero artifact)
    - one exact duplicate row
    - one row with a corrupted "Web Attack" label
    - a majority-class (BENIGN) and two minority classes, one extremely rare
    """
    rows = [
        # BENIGN majority class
        *[
            {"Flow Duration": 100 + i, "Flow Bytes/s": 500.0, "Flow Packets/s": 10.0, "Label": "BENIGN"}
            for i in range(20)
        ],
        # DDoS class, mid-sized
        *[
            {"Flow Duration": 5 + i, "Flow Bytes/s": 9000.0, "Flow Packets/s": 300.0, "Label": "DDoS"}
            for i in range(8)
        ],
        # Heartbleed — extremely rare (n < 3), must survive the pipeline
        {"Flow Duration": 1, "Flow Bytes/s": 1.0, "Flow Packets/s": 1.0, "Label": "Heartbleed"},
        # Corrupted Web Attack label (U+FFFD in place of an en-dash)
        {
            "Flow Duration": 42,
            "Flow Bytes/s": 200.0,
            "Flow Packets/s": 5.0,
            "Label": "Web Attack � Brute Force",
        },
        # ±Infinity artifacts (Flow Duration == 0 style division-by-zero)
        {"Flow Duration": 0, "Flow Bytes/s": np.inf, "Flow Packets/s": -np.inf, "Label": "BENIGN"},
    ]
    df = pd.DataFrame(rows)
    # Exact duplicate of the first BENIGN row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df
