"""Structured logging setup.

Every module in this project logs through :func:`get_logger` — see
AGENT.md's "Never use print()" rule. Call :func:`configure_logging` once,
early, from any entrypoint (CLI script, FastAPI app, Streamlit app); it is
idempotent and safe to call multiple times.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from threading import Lock

import yaml

from sentinelxai.config import CONFIGS_DIR, get_config

_configured = False
_lock = Lock()


def configure_logging() -> None:
    """Load ``configs/logging.yaml`` and apply it via ``dictConfig``."""
    global _configured
    with _lock:
        if _configured:
            return

        cfg = get_config()
        cfg.paths.logs_dir.mkdir(parents=True, exist_ok=True)

        logging_config_path = CONFIGS_DIR / "logging.yaml"
        with logging_config_path.open("r", encoding="utf-8") as fh:
            logging_config = yaml.safe_load(fh)

        # File handlers in logging.yaml use paths relative to the project
        # root (e.g. "logs/application.log"); rewrite them to the resolved,
        # config-driven logs directory before handing off to dictConfig.
        for handler in logging_config.get("handlers", {}).values():
            if "filename" in handler:
                filename = Path(handler["filename"]).name
                handler["filename"] = str(cfg.paths.logs_dir / filename)

        logging.config.dictConfig(logging_config)
        _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, initializing logging on first use."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
