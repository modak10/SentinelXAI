"""LightGBM model construction — the project's final model (Milestone 4).

One construction path (:func:`build_lightgbm`) used identically by both
Optuna's search (``tuning.py``, one call per trial) and the final model
training script — a hyperparameter bug can't sneak in between search and
final fit because both go through the same function.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb

from sentinelxai.config import LightGBMConfig

_REQUIRED_BEST_RESULT_KEYS = (
    "best_params",
    "best_iteration",
    "best_f1_macro",
    "best_f1_weighted",
    "best_trial_number",
)


def build_lightgbm(
    params: dict, *, num_class: int, n_estimators: int, cfg: LightGBMConfig, seed: int
) -> lgb.LGBMClassifier:
    """Construct an LGBMClassifier from fixed config + a tunable-params dict.

    `params` is either an Optuna trial's suggested hyperparameters (during
    search) or the final chosen best_params (after tuning) — the fixed
    settings (objective, n_jobs, verbosity) always come from
    configs/lightgbm.yaml, never duplicated inline.
    """
    return lgb.LGBMClassifier(
        objective=cfg.objective,
        num_class=num_class,
        n_jobs=cfg.n_jobs,
        verbosity=cfg.verbosity,
        random_state=seed,
        n_estimators=n_estimators,
        **params,
    )


def load_best_params(path: Path) -> dict:
    """Load and validate the JSON written by ``tuning.save_study_artifacts``.

    Fails loudly (rather than letting a malformed/missing key surface as a
    confusing KeyError deep inside final-model training) if any field the
    final training script depends on is absent.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/tune_lightgbm.py` first."
        )
    with path.open("r", encoding="utf-8") as fh:
        best_result = json.load(fh)

    missing = [k for k in _REQUIRED_BEST_RESULT_KEYS if k not in best_result]
    if missing:
        raise ValueError(f"{path} is missing required key(s) {missing}")
    return best_result


def build_final_training_config(
    *, lgbm_cfg: LightGBMConfig, best_result: dict, seed: int
) -> dict:
    """Assemble the "training configuration" artifact required for the final model.

    Merges the fixed settings (configs/lightgbm.yaml) with the tuned
    hyperparameters and Optuna provenance (best_result, from
    reports/lightgbm/optuna_best_params.json) into one self-describing
    record — so the final model's exact configuration is reproducible from
    this file alone, without needing to cross-reference the Optuna trial
    history.
    """
    return {
        "fixed_settings": {
            "objective": lgbm_cfg.objective,
            "n_jobs": lgbm_cfg.n_jobs,
            "verbosity": lgbm_cfg.verbosity,
            "random_state": seed,
        },
        "tuned_hyperparameters": dict(best_result["best_params"]),
        "n_estimators": best_result["best_iteration"],
        "optuna_provenance": {
            "best_trial_number": best_result["best_trial_number"],
            "best_f1_macro": best_result["best_f1_macro"],
            "best_f1_weighted": best_result["best_f1_weighted"],
        },
    }
