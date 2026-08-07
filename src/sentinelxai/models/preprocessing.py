"""Model-specific preprocessing, applied on top of the Milestone 2 engineered features.

Tree models (Random Forest, XGBoost, and later LightGBM) consume the
engineered features as-is — no scaling, no log transform, raw
``Destination Port``. Logistic Regression needs its own branch (port
bucketing, log1p, scaling) — see the Milestone 2 Feature Engineering
Plan's "Scaling requirement per model" table and
``configs/baseline_models.yaml``.

Every transform here is either fit-free (port bucketing, log1p — fixed
thresholds, no learned parameters) or explicitly fit-on-train-only
(:class:`~sklearn.preprocessing.StandardScaler`, the label encoder) —
callers are responsible for fitting once on train and re-using the fitted
object for val/test, never re-fitting per split. That discipline is what
makes this leakage-safe; nothing in this module hides it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from sentinelxai.config import LinearModelPreprocessingConfig
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


# --- Memory management -------------------------------------------------------


def downcast_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Downcast `columns` from float64/int64 to float32.

    The dev environment this project runs in has ~7.65GB total RAM; at
    1.76M rows, holding several float64 copies of a ~60-79 column feature
    matrix alive simultaneously (raw + port-bucketed + log1p'd + scaled)
    is enough to exhaust it (`numpy._core._exceptions._ArrayMemoryError`
    was hit during real Logistic Regression training before this was
    added). float32's ~7 significant decimal digits is far more precision
    than these engineered network-flow features need — this is a memory
    fix, not an accuracy trade-off.
    """
    df = df.copy()
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(np.float32)
    return df


# --- Label encoding ---------------------------------------------------------


def fit_label_encoder(y_train: pd.Series) -> LabelEncoder:
    """Fit a LabelEncoder on TRAIN labels only."""
    encoder = LabelEncoder()
    encoder.fit(y_train)
    logger.info("Fit label encoder on %d classes: %s", len(encoder.classes_), list(encoder.classes_))
    return encoder


def encode_labels(encoder: LabelEncoder, y: pd.Series) -> np.ndarray:
    """Transform labels with an already-fitted encoder.

    Raises rather than silently failing if `y` contains a label the
    encoder never saw during fit (would otherwise surface as a cryptic
    sklearn ValueError deep inside `.transform()`).
    """
    unseen = set(y.unique()) - set(encoder.classes_)
    if unseen:
        raise ValueError(
            f"Label(s) not present in the encoder's fit set (fit on train only): {unseen}. "
            "This should not happen given the Milestone 1 rare-class-preserving split — "
            "investigate before proceeding."
        )
    return encoder.transform(y)


# --- Logistic Regression-only preprocessing ---------------------------------


def bucket_destination_port(
    df: pd.DataFrame, port_column: str, cfg: LinearModelPreprocessingConfig
) -> pd.DataFrame:
    """Replace the raw, high-cardinality `port_column` with bucket + indicator dummies.

    Fit-free: every boundary is a fixed constant from config, so this is
    safe to apply independently and identically to train/val/test.
    """
    df = df.copy()
    port = df[port_column]

    for bucket in cfg.destination_port_buckets:
        mask = pd.Series(True, index=df.index)
        if bucket.min_port is not None:
            mask &= port >= bucket.min_port
        if bucket.max_port is not None:
            mask &= port <= bucket.max_port
        df[bucket.name] = mask.astype(int)

    for p in cfg.destination_port_indicator_ports:
        df[f"port_is_{p}"] = (port == p).astype(int)

    return df.drop(columns=[port_column])


def log1p_transform(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Apply log1p to `columns`. Fit-free (no learned parameters).

    Raises if any value is negative — log1p is only valid on this
    project's data because every feature was already clipped to >= 0 in
    Milestone 2 (see configs/features.yaml); a negative value here would
    mean that guarantee was violated somewhere upstream, which should
    fail loudly, not silently produce NaN.
    """
    df = df.copy()
    for col in columns:
        if (df[col] < 0).any():
            raise ValueError(
                f"log1p_transform: column {col!r} contains negative values — "
                "clip to >= 0 before applying log1p (see features/engineering.py)."
            )
        df[col] = np.log1p(df[col])
    return df


def build_linear_model_features(
    df: pd.DataFrame,
    port_column: str,
    cfg: LinearModelPreprocessingConfig,
    skip_log1p_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Full fit-free preprocessing for Logistic Regression: port bucketing + log1p.

    `skip_log1p_columns` must include any column allowed to hold a
    legitimate negative sentinel — this project's
    `Init_Win_bytes_forward`/`Init_Win_bytes_backward` (see
    configs/features.yaml `sentinel_preserve_columns`, -1 means "not
    observed") are otherwise indistinguishable from a data-quality bug to
    a blind "log1p everything numeric" pass, and log1p(-1) is -inf.

    Does NOT scale — scaling needs a separate fit-on-train-only step (see
    :func:`fit_scaler`/:func:`apply_scaler`) that this function's caller
    orchestrates across train/val/test, not this function itself.
    """
    df = bucket_destination_port(df, port_column, cfg)
    if cfg.log1p_all_features:
        numeric_columns = [
            c
            for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in skip_log1p_columns
        ]
        df = log1p_transform(df, numeric_columns)
    return df


def fit_scaler(df: pd.DataFrame, columns: list[str]) -> StandardScaler:
    """Fit a StandardScaler on TRAIN data only."""
    scaler = StandardScaler()
    scaler.fit(df[columns])
    return scaler


def apply_scaler(scaler: StandardScaler, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Transform `df` with an already-fitted scaler."""
    df = df.copy()
    df[columns] = scaler.transform(df[columns])
    return df


# --- Class imbalance (shared) ------------------------------------------------


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample 'balanced' weights — the multi-class analog of `class_weight`
    for estimators (like XGBoost) that take `sample_weight` instead.
    """
    return compute_sample_weight(class_weight="balanced", y=y)
