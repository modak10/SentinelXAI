#!/usr/bin/env python
"""Train and evaluate the Milestone 3 baseline models, in order.

    python scripts/train_baselines.py --models logistic_regression
    python scripts/train_baselines.py --models random_forest
    python scripts/train_baselines.py --models xgboost
    python scripts/train_baselines.py --models logistic_regression,random_forest,xgboost

Reads data/processed/engineered/{train,val}.parquet (Milestone 2 output).
Evaluation is on VAL, never TEST — test stays untouched until a single
final model is chosen in Milestone 4 (docs/JUDGE_QNA.md Q8).

Outputs per model, under models/baselines/ and reports/baselines/:
    <model>.joblib
    <model>_metrics.json
    <model>_classification_report.txt
    figures/<model>_confusion_matrix.png

When more than one model is run in the same invocation, also writes
reports/baselines/model_comparison.md summarizing and ranking them.

Note: this dev environment has ~7.65GB total RAM. Feature columns are
downcast to float32 immediately after load (see
preprocessing.py::downcast_numeric_columns) and `gc.collect()` is called
between model blocks — without this, Logistic Regression's extra
port-bucketed/log1p'd/scaled copies of the 1.76M-row train split
exhausted available memory during real testing.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402
from sentinelxai.models.baseline import (  # noqa: E402
    BaselineResult,
    build_logistic_regression,
    build_random_forest,
    build_xgboost,
    train_and_evaluate_baseline,
)
from sentinelxai.models.metrics import build_comparison_table  # noqa: E402
from sentinelxai.models.preprocessing import (  # noqa: E402
    apply_scaler,
    build_linear_model_features,
    compute_sample_weights,
    downcast_numeric_columns,
    encode_labels,
    fit_label_encoder,
    fit_scaler,
)

logger = get_logger("sentinelxai.models.train_baselines")

ALL_MODELS = ("logistic_regression", "random_forest", "xgboost")


def _load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / "engineered" / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python scripts/engineer_features.py` first."
        )
    logger.info("Loading %s", path)
    return pd.read_parquet(path)


def _write_comparison_report(results: dict[str, BaselineResult], path: Path) -> None:
    table = build_comparison_table({name: r.report.to_dict() for name, r in results.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table, encoding="utf-8")
    logger.info("Saved model comparison to %s", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(ALL_MODELS),
        help=f"Comma-separated subset of {ALL_MODELS}, in the order to run.",
    )
    args = parser.parse_args()
    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = set(requested) - set(ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s) {unknown}, must be a subset of {ALL_MODELS}")

    configure_logging()
    cfg = get_config()

    train_df = _load_split(cfg.paths.data_processed_dir, "train")
    val_df = _load_split(cfg.paths.data_processed_dir, "val")

    label_col = cfg.data.label_column
    feature_columns = [c for c in train_df.columns if c != label_col]

    # Memory: downcast immediately, before any per-model branching creates
    # further copies (see module docstring).
    train_df = downcast_numeric_columns(train_df, feature_columns)
    val_df = downcast_numeric_columns(val_df, feature_columns)
    gc.collect()

    encoder = fit_label_encoder(train_df[label_col])
    y_train = encode_labels(encoder, train_df[label_col])
    y_val = encode_labels(encoder, val_df[label_col])
    label_names = list(encoder.classes_)

    models_dir = cfg.paths.models_dir / "baselines"
    reports_dir = cfg.paths.reports_dir / "baselines"

    results: dict[str, BaselineResult] = {}
    started = time.monotonic()

    if "logistic_regression" in requested:
        logger.info("=== Logistic Regression ===")
        X_train_lr = build_linear_model_features(
            train_df[feature_columns],
            "Destination Port",
            cfg.baseline_models.linear_model_preprocessing,
            skip_log1p_columns=cfg.features.sentinel_preserve_columns,
        )
        X_val_lr = build_linear_model_features(
            val_df[feature_columns],
            "Destination Port",
            cfg.baseline_models.linear_model_preprocessing,
            skip_log1p_columns=cfg.features.sentinel_preserve_columns,
        )
        scale_columns = list(X_train_lr.columns)
        scaler = fit_scaler(X_train_lr, scale_columns)
        X_train_lr = apply_scaler(scaler, X_train_lr, scale_columns)
        X_val_lr = apply_scaler(scaler, X_val_lr, scale_columns)

        model = build_logistic_regression(cfg.baseline_models.logistic_regression, cfg.random_seed)
        results["logistic_regression"] = train_and_evaluate_baseline(
            model_name="logistic_regression",
            model=model,
            X_train=X_train_lr,
            y_train=y_train,
            X_val=X_val_lr,
            y_val=y_val,
            label_names=label_names,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )
        del X_train_lr, X_val_lr, scaler, model
        gc.collect()

    if "random_forest" in requested:
        logger.info("=== Random Forest ===")
        model = build_random_forest(cfg.baseline_models.random_forest, cfg.random_seed)
        results["random_forest"] = train_and_evaluate_baseline(
            model_name="random_forest",
            model=model,
            X_train=train_df[feature_columns],
            y_train=y_train,
            X_val=val_df[feature_columns],
            y_val=y_val,
            label_names=label_names,
            models_dir=models_dir,
            reports_dir=reports_dir,
        )
        del model
        gc.collect()

    if "xgboost" in requested:
        logger.info("=== XGBoost ===")
        sample_weight = compute_sample_weights(y_train)
        model = build_xgboost(cfg.baseline_models.xgboost, num_class=len(label_names), seed=cfg.random_seed)
        results["xgboost"] = train_and_evaluate_baseline(
            model_name="xgboost",
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
        del model, sample_weight
        gc.collect()

    if len(results) > 1:
        _write_comparison_report(results, reports_dir / "model_comparison.md")

    elapsed = time.monotonic() - started
    logger.info("=== Baseline training complete in %.1fs for %s ===", elapsed, list(results.keys()))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Baseline training run failed")
        raise
