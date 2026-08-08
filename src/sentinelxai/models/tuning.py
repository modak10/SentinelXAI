"""Optuna hyperparameter search for the final LightGBM model.

Metric design (see configs/optuna.yaml for the full rationale):
    - Early stopping during each trial's fit, and Optuna's pruning signal,
      both watch `multi_logloss` (a robust per-boosting-round metric).
    - The actual trial-selection metric is Macro F1 (`primary_metric`),
      computed once after each trial's fit completes.

Implementation note: :func:`optuna_integration.LightGBMPruningCallback`
requires the study's `direction` to match the *polarity* of the pruning
metric — `multi_logloss` is minimize-type, so the study is created with
`direction="minimize"` and the objective returns `-f1_macro` (or
`-f1_weighted`). This is a standard, well-known trick for reconciling
"maximize my real metric" with "prune on this other minimize-type metric"
— :func:`extract_best_result` undoes the negation when reporting results,
so nothing downstream of this module ever sees a negative F1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
from optuna.pruners import BasePruner, MedianPruner, NopPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna_integration import LightGBMPruningCallback
from sklearn.metrics import f1_score

from sentinelxai.config import LightGBMConfig, OptunaConfig, SearchSpaceParam
from sentinelxai.logging_setup import get_logger
from sentinelxai.models.lightgbm_model import build_lightgbm

logger = get_logger(__name__)


def suggest_from_search_space(trial: optuna.Trial, name: str, spec: SearchSpaceParam) -> float:
    """Dispatch to trial.suggest_float/suggest_int based on the config's `type`."""
    if spec.type == "float":
        return trial.suggest_float(name, spec.low, spec.high, log=spec.log)
    if spec.type == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
    raise ValueError(f"Unsupported search_space type {spec.type!r} for parameter {name!r}")


def _build_sampler(optuna_cfg: OptunaConfig) -> BaseSampler:
    if optuna_cfg.sampler == "TPE":
        return TPESampler(seed=optuna_cfg.seed)
    raise ValueError(f"Unsupported sampler: {optuna_cfg.sampler!r}")


def _build_pruner(optuna_cfg: OptunaConfig) -> BasePruner:
    if not optuna_cfg.pruning_enabled:
        return NopPruner()
    if optuna_cfg.pruner == "median":
        return MedianPruner()
    raise ValueError(f"Unsupported pruner: {optuna_cfg.pruner!r}")


def make_objective(
    *,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    sample_weight: np.ndarray,
    num_class: int,
    lgbm_cfg: LightGBMConfig,
    optuna_cfg: OptunaConfig,
    seed: int,
):
    """Build the Optuna objective closure over the (already-loaded) data."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            name: suggest_from_search_space(trial, name, spec)
            for name, spec in optuna_cfg.search_space.items()
        }
        model = build_lightgbm(
            params, num_class=num_class, n_estimators=optuna_cfg.max_boost_round, cfg=lgbm_cfg, seed=seed
        )

        callbacks = [
            lgb.early_stopping(optuna_cfg.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ]
        if optuna_cfg.pruning_enabled:
            callbacks.append(LightGBMPruningCallback(trial, optuna_cfg.early_stopping_metric))

        start = time.monotonic()
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            eval_set=[(X_val, y_val)],
            eval_metric=optuna_cfg.early_stopping_metric,
            callbacks=callbacks,
        )
        training_time = time.monotonic() - start

        y_pred = model.predict(X_val)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        best_iteration = int(model.best_iteration_ or optuna_cfg.max_boost_round)

        trial.set_user_attr("f1_macro", f1_macro)
        trial.set_user_attr("f1_weighted", f1_weighted)
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("training_time_seconds", training_time)

        logger.info(
            "Trial %d: f1_macro=%.4f f1_weighted=%.4f best_iter=%d time=%.1fs params=%s",
            trial.number,
            f1_macro,
            f1_weighted,
            best_iteration,
            training_time,
            params,
        )

        primary_value = f1_macro if optuna_cfg.primary_metric == "f1_macro" else f1_weighted
        return -primary_value  # study direction is "minimize" -- see module docstring

    return objective


def run_study(objective, optuna_cfg: OptunaConfig) -> optuna.Study:
    """Run the Optuna study. Direction is always "minimize" internally — see module docstring."""
    study = optuna.create_study(
        direction="minimize",
        sampler=_build_sampler(optuna_cfg),
        pruner=_build_pruner(optuna_cfg),
    )
    study.optimize(objective, n_trials=optuna_cfg.n_trials)
    return study


def extract_best_result(study: optuna.Study, optuna_cfg: OptunaConfig) -> dict:
    """Undo the sign flip and assemble the "best parameters" artifact."""
    best_trial = study.best_trial
    return {
        "primary_metric": optuna_cfg.primary_metric,
        "best_f1_macro": best_trial.user_attrs.get("f1_macro"),
        "best_f1_weighted": best_trial.user_attrs.get("f1_weighted"),
        "best_iteration": best_trial.user_attrs.get("best_iteration"),
        "best_trial_number": best_trial.number,
        "n_trials_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "n_trials_pruned": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        ),
        "n_trials_total": len(study.trials),
        "best_params": dict(best_trial.params),
    }


def save_study_artifacts(study: optuna.Study, best_result: dict, best_params_path: Path, trials_csv_path: Path) -> None:
    """Save the two required artifacts: best parameters (JSON) and full trial history (CSV)."""
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with best_params_path.open("w", encoding="utf-8") as fh:
        json.dump(best_result, fh, indent=2)
    logger.info("Saved best parameters to %s", best_params_path)

    trials_csv_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(trials_csv_path, index=False)
    logger.info("Saved trial history (%d trials) to %s", len(study.trials), trials_csv_path)
