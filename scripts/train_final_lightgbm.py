#!/usr/bin/env python
"""Train the final LightGBM model from Optuna's best hyperparameters (Milestone 4, Phase 3).

    python scripts/train_final_lightgbm.py

Reads reports/lightgbm/optuna_best_params.json (produced by
scripts/tune_lightgbm.py) and data/processed/engineered/{train,val}.parquet.
Reuses, unchanged: float32 downcasting, label encoding (fit on train only),
balanced sample weights, and the model-agnostic train_and_evaluate_baseline
harness (src/sentinelxai/models/baseline.py) — the same function that
trained Logistic Regression, Random Forest, and XGBoost in Milestone 3.

Evaluation is on VAL only — test stays untouched until the final model is
chosen across all of Milestone 4 (docs/JUDGE_QNA.md Q8).

Outputs:
    models/lightgbm/lightgbm.joblib             the trained model
    models/lightgbm/label_encoder.joblib         fit-on-train label encoder
    reports/lightgbm/lightgbm_metrics.json       full EvaluationReport
    reports/lightgbm/lightgbm_classification_report.txt
    reports/lightgbm/figures/lightgbm_confusion_matrix.png
    reports/lightgbm/lightgbm_training_config.json   fixed + tuned params, provenance
    reports/lightgbm/lightgbm_feature_list.json      exact feature column order
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402
from sentinelxai.models.baseline import train_and_evaluate_baseline  # noqa: E402
from sentinelxai.models.lightgbm_model import (  # noqa: E402
    build_final_training_config,
    build_lightgbm,
    load_best_params,
)
from sentinelxai.models.preprocessing import (  # noqa: E402
    compute_sample_weights,
    downcast_numeric_columns,
    encode_labels,
    fit_label_encoder,
)

logger = get_logger("sentinelxai.models.train_final_lightgbm")


def _load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / "engineered" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/engineer_features.py` first."
        )
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def main() -> None:
    configure_logging()
    cfg = get_config()
    started = time.monotonic()

    reports_dir = cfg.paths.reports_dir / "lightgbm"
    models_dir = cfg.paths.models_dir / "lightgbm"

    best_result = load_best_params(reports_dir / "optuna_best_params.json")
    logger.info(
        "Loaded best Optuna trial #%d: f1_macro=%.4f, best_iteration=%d",
        best_result["best_trial_number"],
        best_result["best_f1_macro"],
        best_result["best_iteration"],
    )

    train_df = _load_split(cfg.paths.data_processed_dir, "train")
    val_df = _load_split(cfg.paths.data_processed_dir, "val")

    label_col = cfg.data.label_column
    feature_columns = [c for c in train_df.columns if c != label_col]

    # Reused, unchanged from Milestones 3 and 4 Phase 1/2 (see module docstring).
    train_df = downcast_numeric_columns(train_df, feature_columns)
    val_df = downcast_numeric_columns(val_df, feature_columns)
    gc.collect()

    encoder = fit_label_encoder(train_df[label_col])
    y_train = encode_labels(encoder, train_df[label_col])
    y_val = encode_labels(encoder, val_df[label_col])
    label_names = list(encoder.classes_)
    sample_weight = compute_sample_weights(y_train)

    model = build_lightgbm(
        best_result["best_params"],
        num_class=len(label_names),
        n_estimators=best_result["best_iteration"],
        cfg=cfg.lightgbm,
        seed=cfg.random_seed,
    )

    result = train_and_evaluate_baseline(
        model_name="lightgbm",
        model=model,
        X_train=train_df[feature_columns],
        y_train=y_train,
        X_val=val_df[feature_columns],
        y_val=y_val,
        label_names=label_names,
        models_dir=models_dir,
        reports_dir=reports_dir,
        sample_weight=sample_weight,
    )

    # Additional Phase 3 artifacts beyond what train_and_evaluate_baseline saves.
    encoder_path = models_dir / "label_encoder.joblib"
    joblib.dump(encoder, encoder_path)
    logger.info("Saved label encoder to %s", encoder_path)

    training_config = build_final_training_config(
        lgbm_cfg=cfg.lightgbm, best_result=best_result, seed=cfg.random_seed
    )
    training_config_path = reports_dir / "lightgbm_training_config.json"
    with training_config_path.open("w", encoding="utf-8") as fh:
        json.dump(training_config, fh, indent=2)
    logger.info("Saved training configuration to %s", training_config_path)

    feature_list_path = reports_dir / "lightgbm_feature_list.json"
    with feature_list_path.open("w", encoding="utf-8") as fh:
        json.dump({"feature_columns": feature_columns}, fh, indent=2)
    logger.info("Saved feature list (%d columns) to %s", len(feature_columns), feature_list_path)

    elapsed = time.monotonic() - started
    logger.info(
        "=== Final LightGBM training complete in %.1fs: accuracy=%.4f macro_f1=%.4f "
        "weighted_f1=%.4f train=%.1fs infer=%.4fms/sample ===",
        elapsed,
        result.report.accuracy,
        result.report.f1_macro,
        result.report.f1_weighted,
        result.report.training_time_seconds,
        result.report.inference_time_per_sample_ms,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Final LightGBM training failed")
        raise
