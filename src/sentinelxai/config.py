"""Typed, config-driven settings loader.

Loads ``configs/config.yaml`` and ``configs/data.yaml``, applies ``.env``
overrides (via python-dotenv), and exposes everything through small,
immutable dataclasses. Every path returned here is absolute, resolved
against the project root — nothing downstream should ever hardcode a path.

Usage::

    from sentinelxai.config import get_config

    cfg = get_config()
    cfg.paths.data_raw_dir
    cfg.data.split.train
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


@dataclass(frozen=True)
class PathsConfig:
    data_raw_dir: Path
    data_interim_dir: Path
    data_processed_dir: Path
    models_dir: Path
    logs_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float
    stratify: bool
    rare_class_floor: int


@dataclass(frozen=True)
class DuplicateColumnRule:
    name: str
    keep: str


@dataclass(frozen=True)
class CleaningConfig:
    infinity_columns: tuple[str, ...]
    nan_strategy: str
    drop_duplicate_rows: bool


@dataclass(frozen=True)
class DataConfig:
    raw_files: tuple[str, ...]
    raw_file_encoding: str
    label_column: str
    duplicate_columns: tuple[DuplicateColumnRule, ...]
    split: SplitConfig
    cleaning: CleaningConfig
    leak_risk_columns: tuple[str, ...]


@dataclass(frozen=True)
class ClipRule:
    name: str
    min_value: float


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    exclude_columns: tuple[str, ...]
    drop_columns: tuple[str, ...]
    clip_columns: tuple[ClipRule, ...]
    sentinel_preserve_columns: tuple[str, ...]
    correlation_removal_threshold: float


@dataclass(frozen=True)
class LogisticRegressionConfig:
    max_iter: int
    C: float
    solver: str
    class_weight: str


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    n_jobs: int
    class_weight: str


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    tree_method: str
    n_jobs: int


@dataclass(frozen=True)
class PortBucketRule:
    name: str
    min_port: int | None
    max_port: int | None


@dataclass(frozen=True)
class LinearModelPreprocessingConfig:
    destination_port_buckets: tuple[PortBucketRule, ...]
    destination_port_indicator_ports: tuple[int, ...]
    log1p_all_features: bool


@dataclass(frozen=True)
class BaselineModelsConfig:
    logistic_regression: LogisticRegressionConfig
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    linear_model_preprocessing: LinearModelPreprocessingConfig


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    project_version: str
    random_seed: int
    log_level: str
    paths: PathsConfig
    data: DataConfig
    features: FeatureEngineeringConfig
    baseline_models: BaselineModelsConfig


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve_path(raw: str, override_env_var: str | None = None) -> Path:
    """Resolve a config-relative path, allowing a .env override."""
    if override_env_var and (env_value := os.getenv(override_env_var)):
        raw = env_value
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and cache the full application configuration.

    Cached with ``lru_cache`` so config is parsed once per process; call
    ``get_config.cache_clear()`` in tests that need a fresh load.
    """
    load_dotenv(PROJECT_ROOT / ".env", override=False)

    raw_config = _load_yaml(CONFIGS_DIR / "config.yaml")
    raw_data = _load_yaml(CONFIGS_DIR / "data.yaml")
    raw_features = _load_yaml(CONFIGS_DIR / "features.yaml")
    raw_baselines = _load_yaml(CONFIGS_DIR / "baseline_models.yaml")

    paths_raw = raw_config["paths"]
    paths = PathsConfig(
        data_raw_dir=_resolve_path(paths_raw["data_raw_dir"], "DATA_RAW_DIR"),
        data_interim_dir=_resolve_path(paths_raw["data_interim_dir"]),
        data_processed_dir=_resolve_path(
            paths_raw["data_processed_dir"], "DATA_PROCESSED_DIR"
        ),
        models_dir=_resolve_path(paths_raw["models_dir"]),
        logs_dir=_resolve_path(paths_raw["logs_dir"]),
        reports_dir=_resolve_path(paths_raw["reports_dir"]),
    )

    split_raw = raw_data["split"]
    split = SplitConfig(
        train=float(split_raw["train"]),
        val=float(split_raw["val"]),
        test=float(split_raw["test"]),
        stratify=bool(split_raw["stratify"]),
        rare_class_floor=int(split_raw["rare_class_floor"]),
    )
    if not abs((split.train + split.val + split.test) - 1.0) < 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {split.train + split.val + split.test}"
        )

    cleaning_raw = raw_data["cleaning"]
    cleaning = CleaningConfig(
        infinity_columns=tuple(cleaning_raw["infinity_columns"]),
        nan_strategy=cleaning_raw["nan_strategy"],
        drop_duplicate_rows=bool(cleaning_raw["drop_duplicate_rows"]),
    )

    duplicate_columns = tuple(
        DuplicateColumnRule(name=d["name"], keep=d["keep"])
        for d in raw_data.get("duplicate_columns", [])
    )

    data = DataConfig(
        raw_files=tuple(raw_data["raw_files"]),
        raw_file_encoding=raw_data["raw_file_encoding"],
        label_column=raw_data["label_column"],
        duplicate_columns=duplicate_columns,
        split=split,
        cleaning=cleaning,
        leak_risk_columns=tuple(raw_data.get("leak_risk_columns", [])),
    )

    features = FeatureEngineeringConfig(
        exclude_columns=tuple(raw_features.get("exclude_columns", [])),
        drop_columns=tuple(raw_features.get("drop_columns", [])),
        clip_columns=tuple(
            ClipRule(name=c["name"], min_value=float(c["min_value"]))
            for c in raw_features.get("clip_columns", [])
        ),
        sentinel_preserve_columns=tuple(raw_features.get("sentinel_preserve_columns", [])),
        correlation_removal_threshold=float(
            raw_features.get("correlation_removal_threshold", 0.95)
        ),
    )
    clip_names = {c.name for c in features.clip_columns}
    sentinel_overlap = clip_names & set(features.sentinel_preserve_columns)
    if sentinel_overlap:
        raise ValueError(
            f"Columns listed in both clip_columns and sentinel_preserve_columns "
            f"(configs/features.yaml) — a column cannot be both clipped and "
            f"sentinel-preserved: {sentinel_overlap}"
        )

    lr_raw = raw_baselines["logistic_regression"]
    rf_raw = raw_baselines["random_forest"]
    xgb_raw = raw_baselines["xgboost"]
    lmp_raw = raw_baselines["linear_model_preprocessing"]
    baseline_models = BaselineModelsConfig(
        logistic_regression=LogisticRegressionConfig(
            max_iter=int(lr_raw["max_iter"]),
            C=float(lr_raw["C"]),
            solver=lr_raw["solver"],
            class_weight=lr_raw["class_weight"],
        ),
        random_forest=RandomForestConfig(
            n_estimators=int(rf_raw["n_estimators"]),
            max_depth=int(rf_raw["max_depth"]),
            min_samples_leaf=int(rf_raw["min_samples_leaf"]),
            n_jobs=int(rf_raw["n_jobs"]),
            class_weight=rf_raw["class_weight"],
        ),
        xgboost=XGBoostConfig(
            n_estimators=int(xgb_raw["n_estimators"]),
            max_depth=int(xgb_raw["max_depth"]),
            learning_rate=float(xgb_raw["learning_rate"]),
            tree_method=xgb_raw["tree_method"],
            n_jobs=int(xgb_raw["n_jobs"]),
        ),
        linear_model_preprocessing=LinearModelPreprocessingConfig(
            destination_port_buckets=tuple(
                PortBucketRule(
                    name=b["name"],
                    min_port=b.get("min_port"),
                    max_port=b.get("max_port"),
                )
                for b in lmp_raw["destination_port_buckets"]
            ),
            destination_port_indicator_ports=tuple(
                int(p) for p in lmp_raw["destination_port_indicator_ports"]
            ),
            log1p_all_features=bool(lmp_raw["log1p_all_features"]),
        ),
    )

    return AppConfig(
        project_name=raw_config["project"]["name"],
        project_version=raw_config["project"]["version"],
        random_seed=int(raw_config["random_seed"]),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        paths=paths,
        data=data,
        features=features,
        baseline_models=baseline_models,
    )
