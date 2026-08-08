#!/usr/bin/env python
"""Optuna hyperparameter search for the final LightGBM model (Milestone 4, Phase 2).

    python scripts/tune_lightgbm.py

Reads data/processed/engineered/{train,val}.parquet (same as the
Milestone 3 baselines). Reuses, unchanged: float32 downcasting, label
encoding (fit on train only), and balanced sample weights — see
src/sentinelxai/models/preprocessing.py.

Evaluation during search is on VAL only — test stays untouched until the
final model is chosen (docs/JUDGE_QNA.md Q8).

Outputs under reports/lightgbm/:
    optuna_best_params.json   best hyperparameters + best metric values
    optuna_trials.csv         full trial history (params, value, state, user_attrs)
    optuna_summary.md         human-readable recap (top trials, timing)
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402
from sentinelxai.models.preprocessing import (  # noqa: E402
    compute_sample_weights,
    downcast_numeric_columns,
    encode_labels,
    fit_label_encoder,
)
from sentinelxai.models.tuning import (  # noqa: E402
    extract_best_result,
    make_objective,
    run_study,
    save_study_artifacts,
)

logger = get_logger("sentinelxai.models.tune_lightgbm")


def _load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / "engineered" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/engineer_features.py` first."
        )
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def _write_summary_md(study: optuna.Study, best_result: dict, path: Path) -> None:
    trials_df = study.trials_dataframe()
    completed = trials_df[trials_df["state"] == "COMPLETE"].copy()
    completed["f1_macro"] = completed["user_attrs_f1_macro"]
    top5 = completed.sort_values("f1_macro", ascending=False).head(5)

    lines = [
        "# LightGBM Optuna Search Summary (Milestone 4, Phase 2)",
        "",
        f"- Trials total: {best_result['n_trials_total']}",
        f"- Trials completed: {best_result['n_trials_completed']}",
        f"- Trials pruned: {best_result['n_trials_pruned']}",
        f"- Best trial: #{best_result['best_trial_number']}",
        f"- Best Macro F1 (val): {best_result['best_f1_macro']:.4f}",
        f"- Best Weighted F1 (val): {best_result['best_f1_weighted']:.4f}",
        f"- Best iteration (boosting rounds): {best_result['best_iteration']}",
        "",
        "## Best Parameters",
        "",
        "```json",
    ]
    lines.append(json.dumps(best_result["best_params"], indent=2))
    lines.append("```")
    lines += ["", "## Top 5 Trials by Macro F1", "", "| Trial | Macro F1 | Weighted F1 | Duration |", "|---|---|---|---|"]
    for _, row in top5.iterrows():
        lines.append(
            f"| {int(row['number'])} | {row['f1_macro']:.4f} | {row['user_attrs_f1_weighted']:.4f} "
            f"| {row['duration']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Optuna summary to %s", path)


def main() -> None:
    configure_logging()
    cfg = get_config()
    started = time.monotonic()

    train_df = _load_split(cfg.paths.data_processed_dir, "train")
    val_df = _load_split(cfg.paths.data_processed_dir, "val")

    label_col = cfg.data.label_column
    feature_columns = [c for c in train_df.columns if c != label_col]

    # Reused, unchanged from Milestone 3 (see module docstring).
    train_df = downcast_numeric_columns(train_df, feature_columns)
    val_df = downcast_numeric_columns(val_df, feature_columns)
    gc.collect()

    encoder = fit_label_encoder(train_df[label_col])
    y_train = encode_labels(encoder, train_df[label_col])
    y_val = encode_labels(encoder, val_df[label_col])
    label_names = list(encoder.classes_)
    sample_weight = compute_sample_weights(y_train)

    logger.info(
        "Starting Optuna search: %d trials, max_boost_round=%d, early_stopping=%d, pruning=%s",
        cfg.optuna.n_trials,
        cfg.optuna.max_boost_round,
        cfg.optuna.early_stopping_rounds,
        cfg.optuna.pruning_enabled,
    )

    objective = make_objective(
        X_train=train_df[feature_columns],
        y_train=y_train,
        X_val=val_df[feature_columns],
        y_val=y_val,
        sample_weight=sample_weight,
        num_class=len(label_names),
        lgbm_cfg=cfg.lightgbm,
        optuna_cfg=cfg.optuna,
        seed=cfg.random_seed,
    )
    study = run_study(objective, cfg.optuna)
    best_result = extract_best_result(study, cfg.optuna)

    reports_dir = cfg.paths.reports_dir / "lightgbm"
    save_study_artifacts(
        study,
        best_result,
        reports_dir / "optuna_best_params.json",
        reports_dir / "optuna_trials.csv",
    )
    _write_summary_md(study, best_result, reports_dir / "optuna_summary.md")

    elapsed = time.monotonic() - started
    logger.info(
        "=== Optuna search complete in %.1fs (%.1f min): best f1_macro=%.4f (trial #%d) ===",
        elapsed,
        elapsed / 60,
        best_result["best_f1_macro"],
        best_result["best_trial_number"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Optuna search failed")
        raise
