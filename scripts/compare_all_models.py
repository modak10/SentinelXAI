#!/usr/bin/env python
"""Build the final Milestone 4 comparison: baselines vs LightGBM.

    python scripts/compare_all_models.py

Reads the already-saved metrics JSONs from Milestone 3
(reports/baselines/*_metrics.json: logistic_regression, random_forest,
xgboost) and Milestone 4 (reports/lightgbm/lightgbm_metrics.json) — ZERO
retraining. Reuses metrics.py::build_comparison_table (the same function
Milestone 3 used) for the ranked table, then appends an explicit verdict on
whether LightGBM surpasses the XGBoost baseline, as Phase 4 requires.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402
from sentinelxai.models.metrics import build_comparison_table, build_pairwise_verdict  # noqa: E402

logger = get_logger("sentinelxai.models.compare_all_models")


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    logger.info("Loaded %s", path)
    return metrics


def main() -> None:
    configure_logging()
    cfg = get_config()
    baselines_dir = cfg.paths.reports_dir / "baselines"
    lightgbm_dir = cfg.paths.reports_dir / "lightgbm"

    results: dict[str, dict] = {}
    for path in sorted(baselines_dir.glob("*_metrics.json")):
        metrics = _load_metrics(path)
        results[metrics["model_name"]] = metrics

    lightgbm_metrics_path = lightgbm_dir / "lightgbm_metrics.json"
    if not lightgbm_metrics_path.exists():
        raise FileNotFoundError(
            f"{lightgbm_metrics_path} not found — run "
            "scripts/train_final_lightgbm.py first."
        )
    lightgbm_metrics = _load_metrics(lightgbm_metrics_path)
    results[lightgbm_metrics["model_name"]] = lightgbm_metrics

    missing_baselines = {"logistic_regression", "random_forest", "xgboost"} - set(results)
    if missing_baselines:
        logger.warning(
            "Comparison is missing baseline model(s) %s — run scripts/train_baselines.py "
            "for the full 4-way comparison.",
            sorted(missing_baselines),
        )

    table = build_comparison_table(
        results, title="# Final Model Comparison (Milestone 4): Baselines vs LightGBM"
    )
    verdict = build_pairwise_verdict(
        results, challenger="lightgbm", baseline="xgboost", metric="f1_macro"
    )
    full_report = f"{table}\n\n## LightGBM vs XGBoost\n\n{verdict}\n"

    output_path = lightgbm_dir / "final_model_comparison.md"
    output_path.write_text(full_report, encoding="utf-8")
    logger.info("Saved final comparison of %d model(s) to %s", len(results), output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Final model comparison failed")
        raise
