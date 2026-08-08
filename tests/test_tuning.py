"""Tests for Optuna hyperparameter search — tiny synthetic data, few trials.

Uses `sample_config`'s `optuna` section (n_trials=2, a 2-parameter search
space) so the suite stays fast; this validates the search *mechanism*
(objective wiring, sign-flip correctness, pruning callback compatibility,
artifact saving), not that any particular hyperparameters are good.
"""

from __future__ import annotations

import json

import numpy as np
import optuna
import pytest

from sentinelxai.config import SearchSpaceParam
from sentinelxai.models.tuning import (
    extract_best_result,
    make_objective,
    run_study,
    save_study_artifacts,
    suggest_from_search_space,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def tiny_multiclass_data():
    rng = np.random.default_rng(42)
    n_per_class, n_classes = 30, 3
    X = rng.normal(size=(n_per_class * n_classes, 4)).astype(np.float32)
    y = np.repeat(np.arange(n_classes), n_per_class)
    # simple split, every class present in both halves
    idx = rng.permutation(len(y))
    train_idx, val_idx = idx[:60], idx[60:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# --- suggest_from_search_space ---


def test_suggest_from_search_space_float_within_bounds():
    study = optuna.create_study()
    trial = study.ask()
    spec = SearchSpaceParam(type="float", low=0.1, high=0.5, log=False)
    value = suggest_from_search_space(trial, "x", spec)
    assert 0.1 <= value <= 0.5


def test_suggest_from_search_space_int_within_bounds():
    study = optuna.create_study()
    trial = study.ask()
    spec = SearchSpaceParam(type="int", low=4, high=10, log=False)
    value = suggest_from_search_space(trial, "x", spec)
    assert isinstance(value, int)
    assert 4 <= value <= 10


def test_suggest_from_search_space_rejects_unknown_type():
    study = optuna.create_study()
    trial = study.ask()
    spec = SearchSpaceParam(type="categorical", low=0, high=1, log=False)
    with pytest.raises(ValueError, match="Unsupported search_space type"):
        suggest_from_search_space(trial, "x", spec)


# --- end-to-end study on tiny data ---


def test_run_study_completes_and_reports_positive_f1(tiny_multiclass_data, sample_config):
    X_train, y_train, X_val, y_val = tiny_multiclass_data
    objective = make_objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight=np.ones(len(y_train)),
        num_class=3,
        lgbm_cfg=sample_config.lightgbm,
        optuna_cfg=sample_config.optuna,
        seed=42,
    )
    study = run_study(objective, sample_config.optuna)

    assert len(study.trials) == sample_config.optuna.n_trials
    best = extract_best_result(study, sample_config.optuna)
    assert best["best_f1_macro"] >= 0.0  # sign correctly un-flipped, never negative
    assert best["best_f1_weighted"] >= 0.0
    assert best["n_trials_total"] == sample_config.optuna.n_trials
    assert set(best["best_params"].keys()) == set(sample_config.optuna.search_space.keys())


def test_run_study_user_attrs_present_on_every_completed_trial(tiny_multiclass_data, sample_config):
    X_train, y_train, X_val, y_val = tiny_multiclass_data
    objective = make_objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight=np.ones(len(y_train)),
        num_class=3,
        lgbm_cfg=sample_config.lightgbm,
        optuna_cfg=sample_config.optuna,
        seed=42,
    )
    study = run_study(objective, sample_config.optuna)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.COMPLETE:
            assert "f1_macro" in trial.user_attrs
            assert "best_iteration" in trial.user_attrs
            assert "training_time_seconds" in trial.user_attrs


def test_save_study_artifacts_writes_both_files(tmp_path, tiny_multiclass_data, sample_config):
    X_train, y_train, X_val, y_val = tiny_multiclass_data
    objective = make_objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        sample_weight=np.ones(len(y_train)),
        num_class=3,
        lgbm_cfg=sample_config.lightgbm,
        optuna_cfg=sample_config.optuna,
        seed=42,
    )
    study = run_study(objective, sample_config.optuna)
    best = extract_best_result(study, sample_config.optuna)

    best_params_path = tmp_path / "best_params.json"
    trials_csv_path = tmp_path / "trials.csv"
    save_study_artifacts(study, best, best_params_path, trials_csv_path)

    assert best_params_path.exists()
    assert trials_csv_path.exists()
    loaded = json.loads(best_params_path.read_text(encoding="utf-8"))
    assert loaded["best_f1_macro"] == best["best_f1_macro"]

    import pandas as pd

    trials_df = pd.read_csv(trials_csv_path)
    assert len(trials_df) == sample_config.optuna.n_trials
