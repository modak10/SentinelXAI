#!/usr/bin/env python
"""Build the baseline model comparison table from already-saved metrics JSONs.

    python scripts/compare_baselines.py

Complements scripts/train_baselines.py: when each model is trained in its
own separate invocation (this milestone's incremental validation process —
run one, validate its real output, then move to the next), the in-process
comparison in train_baselines.py never sees more than one result at a time.
This reads whatever reports/baselines/*_metrics.json files already exist
and builds the same comparison table from disk, with ZERO retraining.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentinelxai.config import get_config  # noqa: E402
from sentinelxai.logging_setup import configure_logging, get_logger  # noqa: E402
from sentinelxai.models.metrics import build_comparison_table  # noqa: E402

logger = get_logger("sentinelxai.models.compare_baselines")


def main() -> None:
    configure_logging()
    cfg = get_config()
    reports_dir = cfg.paths.reports_dir / "baselines"

    metrics_files = sorted(reports_dir.glob("*_metrics.json"))
    if not metrics_files:
        raise FileNotFoundError(
            f"No *_metrics.json found under {reports_dir} — run "
            "scripts/train_baselines.py for at least one model first."
        )

    results: dict[str, dict] = {}
    for path in metrics_files:
        with path.open("r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        results[metrics["model_name"]] = metrics
        logger.info("Loaded %s", path)

    table = build_comparison_table(results)
    output_path = reports_dir / "model_comparison.md"
    output_path.write_text(table, encoding="utf-8")
    logger.info("Saved comparison of %d model(s) to %s", len(results), output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Baseline comparison failed")
        raise
