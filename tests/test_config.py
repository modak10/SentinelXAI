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
