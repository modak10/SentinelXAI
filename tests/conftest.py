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
    BaselineModelsConfig,
    CleaningConfig,
    ClipRule,
    DataConfig,
    DuplicateColumnRule,
    FeatureEngineeringConfig,
    LightGBMConfig,
    LinearModelPreprocessingConfig,
    LogisticRegressionConfig,
    OptunaConfig,
    PathsConfig,
    PortBucketRule,
    RandomForestConfig,
    SearchSpaceParam,
    SplitConfig,
    XGBoostConfig,
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
    features = FeatureEngineeringConfig(
        exclude_columns=("__source_file",),
        drop_columns=("Constant Feature",),
        clip_columns=(ClipRule(name="Flow Duration", min_value=0.0),),
        sentinel_preserve_columns=("Init_Win_bytes_forward",),
        correlation_removal_threshold=0.95,
    )
    baseline_models = BaselineModelsConfig(
        logistic_regression=LogisticRegressionConfig(
            max_iter=200, C=1.0, solver="lbfgs", class_weight="balanced"
        ),
        random_forest=RandomForestConfig(
            n_estimators=10, max_depth=5, min_samples_leaf=2, n_jobs=1,
            class_weight="balanced_subsample",
        ),
        xgboost=XGBoostConfig(
            n_estimators=10, max_depth=3, learning_rate=0.1, tree_method="hist", n_jobs=1
        ),
        linear_model_preprocessing=LinearModelPreprocessingConfig(
            destination_port_buckets=(
                PortBucketRule(name="port_well_known", min_port=None, max_port=1023),
                PortBucketRule(name="port_registered", min_port=1024, max_port=49151),
                PortBucketRule(name="port_dynamic", min_port=49152, max_port=None),
            ),
            destination_port_indicator_ports=(21, 22, 80, 443, 3389),
            log1p_all_features=True,
        ),
    )
    lightgbm = LightGBMConfig(objective="multiclass", n_jobs=1, verbosity=-1)
    optuna_cfg = OptunaConfig(
        n_trials=2,
        direction="maximize",
        sampler="TPE",
        pruner="median",
        seed=42,
        pruning_enabled=True,
        max_boost_round=20,
        early_stopping_rounds=5,
        early_stopping_metric="multi_logloss",
        primary_metric="f1_macro",
        secondary_metric="f1_weighted",
        search_space={
            "learning_rate": SearchSpaceParam(type="float", low=0.05, high=0.3, log=True),
            "num_leaves": SearchSpaceParam(type="int", low=8, high=32, log=False),
        },
    )
    return AppConfig(
        project_name="SentinelXAI-test",
        project_version="0.0.0",
        random_seed=42,
        log_level="DEBUG",
        paths=paths,
        data=data,
        features=features,
        baseline_models=baseline_models,
        lightgbm=lightgbm,
        optuna=optuna_cfg,
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
