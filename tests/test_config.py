"""Integration tests for the real configs/*.yaml files on disk.

Unlike tests/conftest.py's `sample_config` fixture (synthetic, for testing
data/*.py modules in isolation), these tests load the actual project
config and catch mistakes in the YAML itself -- a typo'd column name, an
invalid value, or an overlap between clip_columns and
sentinel_preserve_columns would otherwise only surface at pipeline run
time against the full dataset.
"""

from __future__ import annotations

import pytest

from sentinelxai.config import get_config


@pytest.fixture(autouse=True)
def _clear_config_cache():
    """get_config() is lru_cache'd; ensure each test sees a fresh load."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


def test_real_config_loads_without_error():
    cfg = get_config()
    assert cfg.project_name == "SentinelXAI"


def test_features_config_has_no_clip_sentinel_overlap():
    cfg = get_config()
    clip_names = {c.name for c in cfg.features.clip_columns}
    assert clip_names.isdisjoint(cfg.features.sentinel_preserve_columns)


def test_features_config_drop_columns_are_not_also_excluded():
    """drop_columns (data-driven findings) and exclude_columns (structural
    metadata) should be disjoint -- overlap would mean a column is
    documented twice for two different reasons.
    """
    cfg = get_config()
    assert set(cfg.features.drop_columns).isdisjoint(cfg.features.exclude_columns)


def test_features_config_sentinel_columns_present():
    cfg = get_config()
    assert "Init_Win_bytes_forward" in cfg.features.sentinel_preserve_columns
    assert "Init_Win_bytes_backward" in cfg.features.sentinel_preserve_columns


def test_features_config_clip_values_are_zero_floor():
    """Every configured clip in this dataset is a physical-validity floor
    of 0 (durations/rates/header-lengths cannot be negative) -- if a
    future edit introduces a different bound, this test should be updated
    deliberately, not silently pass.
    """
    cfg = get_config()
    assert all(c.min_value == 0.0 for c in cfg.features.clip_columns)


def test_data_config_duplicate_columns_no_longer_required():
    """Milestone 2 made duplicate-column detection automatic (schema.py) --
    configs/data.yaml's duplicate_columns is now override-only and empty
    by default. This test documents that expectation explicitly.
    """
    cfg = get_config()
    assert cfg.data.duplicate_columns == ()


def test_baseline_models_config_loads():
    cfg = get_config()
    assert cfg.baseline_models.logistic_regression.class_weight == "balanced"
    assert cfg.baseline_models.random_forest.n_estimators > 0
    assert cfg.baseline_models.xgboost.tree_method == "hist"


def test_baseline_models_port_buckets_cover_full_range_with_no_gaps():
    """well_known (0-1023) + registered (1024-49151) + dynamic (49152+)
    must partition the full 0-65535 port range with no gap and no overlap.
    """
    cfg = get_config()
    buckets = {b.name: b for b in cfg.baseline_models.linear_model_preprocessing.destination_port_buckets}
    assert buckets["port_well_known"].max_port + 1 == buckets["port_registered"].min_port
    assert buckets["port_registered"].max_port + 1 == buckets["port_dynamic"].min_port


def test_baseline_models_indicator_ports_are_valid_port_numbers():
    cfg = get_config()
    ports = cfg.baseline_models.linear_model_preprocessing.destination_port_indicator_ports
    assert all(0 <= p <= 65535 for p in ports)
    assert len(ports) == len(set(ports))  # no duplicates


def test_lightgbm_config_loads():
    cfg = get_config()
    assert cfg.lightgbm.objective == "multiclass"


def test_optuna_config_loads_and_recommended_trial_count():
    cfg = get_config()
    assert cfg.optuna.direction == "maximize"
    assert cfg.optuna.primary_metric == "f1_macro"
    # Project Lead's approved plan: "Recommended search: 20-30 trials"
    assert 20 <= cfg.optuna.n_trials <= 30


def test_optuna_search_space_covers_all_approved_parameters():
    """The Milestone 2 plan's approved LightGBM search space, verbatim."""
    cfg = get_config()
    expected = {
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "lambda_l1",
        "lambda_l2",
        "min_split_gain",
    }
    assert set(cfg.optuna.search_space.keys()) == expected


def test_optuna_search_space_bounds_are_well_formed():
    cfg = get_config()
    for name, spec in cfg.optuna.search_space.items():
        assert spec.low < spec.high, f"{name}: low must be < high"
        assert spec.type in ("float", "int")
