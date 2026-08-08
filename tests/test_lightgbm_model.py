"""Tests for LightGBM model construction and Phase 3 final-training helpers."""

from __future__ import annotations

import json

import pytest

from sentinelxai.models.lightgbm_model import (
    build_final_training_config,
    build_lightgbm,
    load_best_params,
)

VALID_BEST_RESULT = {
    "primary_metric": "f1_macro",
    "best_f1_macro": 0.9201,
    "best_f1_weighted": 0.9988,
    "best_iteration": 175,
    "best_trial_number": 2,
    "n_trials_completed": 14,
    "n_trials_pruned": 11,
    "n_trials_total": 25,
    "best_params": {"learning_rate": 0.08, "num_leaves": 22},
}


# --- build_lightgbm ---


def test_build_lightgbm_applies_fixed_config_and_tuned_params(sample_config):
    model = build_lightgbm(
        {"learning_rate": 0.05, "num_leaves": 31},
        num_class=3,
        n_estimators=100,
        cfg=sample_config.lightgbm,
        seed=sample_config.random_seed,
    )
    assert model.objective == sample_config.lightgbm.objective
    assert model.n_jobs == sample_config.lightgbm.n_jobs
    assert model.n_estimators == 100
    assert model.num_class == 3
    assert model.learning_rate == 0.05
    assert model.num_leaves == 31
    assert model.random_state == sample_config.random_seed


# --- load_best_params ---


def test_load_best_params_returns_dict_for_valid_file(tmp_path):
    path = tmp_path / "optuna_best_params.json"
    path.write_text(json.dumps(VALID_BEST_RESULT), encoding="utf-8")
    loaded = load_best_params(path)
    assert loaded["best_iteration"] == 175
    assert loaded["best_params"] == {"learning_rate": 0.08, "num_leaves": 22}


def test_load_best_params_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_best_params(tmp_path / "does_not_exist.json")


def test_load_best_params_raises_when_required_key_missing(tmp_path):
    incomplete = dict(VALID_BEST_RESULT)
    del incomplete["best_iteration"]
    path = tmp_path / "optuna_best_params.json"
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required key"):
        load_best_params(path)


# --- build_final_training_config ---


def test_build_final_training_config_merges_fixed_and_tuned_settings(sample_config):
    config = build_final_training_config(
        lgbm_cfg=sample_config.lightgbm, best_result=VALID_BEST_RESULT, seed=99
    )
    assert config["fixed_settings"]["objective"] == sample_config.lightgbm.objective
    assert config["fixed_settings"]["random_state"] == 99
    assert config["tuned_hyperparameters"] == VALID_BEST_RESULT["best_params"]
    assert config["n_estimators"] == VALID_BEST_RESULT["best_iteration"]
    assert config["optuna_provenance"]["best_trial_number"] == 2
    assert config["optuna_provenance"]["best_f1_macro"] == pytest.approx(0.9201)


def test_build_final_training_config_is_json_serializable(sample_config):
    config = build_final_training_config(
        lgbm_cfg=sample_config.lightgbm, best_result=VALID_BEST_RESULT, seed=99
    )
    json.dumps(config)  # must not raise
