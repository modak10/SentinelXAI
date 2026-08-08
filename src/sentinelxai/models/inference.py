"""Inference service (Phase 7 backbone).

Loads the trained LightGBM model, its fit-on-train label encoder, and the
exact engineered-feature column order, then turns JSON feature records into
validated predictions + per-class probabilities. This is the single object the
FastAPI backend, the Streamlit dashboard, and the test-suite all share — so the
model is loaded exactly once and prediction logic is never duplicated.

The model input is the **engineered feature set** (see
``scripts/engineer_features.py``): tree models consume these raw, with no
scaling. Callers (API/dashboard) therefore accept records keyed by engineered
feature name, which is exactly what the training pipeline fed the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentinelxai.config import get_config
from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    predicted_classes: list[str]
    probabilities: list[dict[str, float]]  # each: class_name -> probability
    confidences: list[float]  # max-class probability
    feature_names: list[str]
    label_names: list[str]

    def to_dict(self) -> dict:
        return {
            "predicted_classes": self.predicted_classes,
            "probabilities": self.probabilities,
            "confidences": self.confidences,
            "feature_names": self.feature_names,
            "label_names": self.label_names,
        }


class ModelNotLoadedError(RuntimeError):
    """Raised when prediction is attempted before a model is available."""


class InferenceService:
    """Loads model artifacts once and serves validated predictions."""

    def __init__(
        self,
        model=None,
        encoder=None,
        feature_names: list[str] | None = None,
    ) -> None:
        self._model = model
        self._encoder = encoder
        self._feature_names = list(feature_names) if feature_names else None

    # --- loading -------------------------------------------------------------

    @classmethod
    def from_paths(
        cls,
        model_path: Path,
        encoder_path: Path,
        feature_list_path: Path,
    ) -> "InferenceService":
        """Load all three artifacts, failing loudly if any is missing."""
        for p in (model_path, encoder_path, feature_list_path):
            if not Path(p).exists():
                raise FileNotFoundError(f"Required model artifact not found: {p}")

        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        with Path(feature_list_path).open("r", encoding="utf-8") as fh:
            feature_names = list(__import__("json").load(fh)["feature_columns"])

        logger.info(
            "Loaded inference model from %s (%d features, %d classes)",
            model_path,
            len(feature_names),
            len(list(encoder.classes_)),
        )
        return cls(model=model, encoder=encoder, feature_names=feature_names)

    @classmethod
    def from_config(cls, cfg=None) -> "InferenceService":
        """Load using the standard artifact locations from ``configs/config.yaml``.

        Mirrors ``scripts/train_final_lightgbm.py``: model at
        ``models/lightgbm/lightgbm.joblib``, encoder at
        ``models/lightgbm/label_encoder.joblib``, feature list at
        ``reports/lightgbm/lightgbm_feature_list.json``.
        """
        cfg = cfg or get_config()
        return cls.from_paths(
            model_path=cfg.paths.models_dir / "lightgbm" / "lightgbm.joblib",
            encoder_path=cfg.paths.models_dir / "lightgbm" / "label_encoder.joblib",
            feature_list_path=cfg.paths.reports_dir / "lightgbm" / "lightgbm_feature_list.json",
        )

    # --- properties ----------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._encoder is not None and self._feature_names is not None

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            raise ModelNotLoadedError("Feature names are not set — model not loaded.")
        return self._feature_names

    @property
    def label_names(self) -> list[str]:
        if self._encoder is None:
            raise ModelNotLoadedError("Encoder not loaded.")
        return list(self._encoder.classes_)

    # --- prediction ----------------------------------------------------------

    def _to_dataframe(self, records: list[dict]) -> pd.DataFrame:
        missing = [c for c in self.feature_names if not any(c in r for r in records)]
        if missing:
            raise ValueError(
                f"Input is missing required engineered feature(s): {missing[:10]}"
                f"{'...' if len(missing) > 10 else ''}. "
                "Pass records keyed by the engineered feature names."
            )
        df = pd.DataFrame(records)
        # Strict ordering + column selection — extra keys are ignored, missing
        # columns would already have raised above.
        df = df[self.feature_names].copy()
        return df.astype(np.float32)

    def predict(self, records: list[dict]) -> PredictionResult:
        """Predict for one or more engineered-feature records."""
        if not self.is_available:
            raise ModelNotLoadedError(
                "No model loaded. Train the final model "
                "(`python scripts/train_final_lightgbm.py`) or set artifacts explicitly."
            )
        if not records:
            raise ValueError("records must be non-empty")

        X = self._to_dataframe(records)
        proba = self._model.predict_proba(X)
        label_names = self.label_names

        predicted_indices = np.argmax(proba, axis=1)
        predicted_classes = [label_names[i] for i in predicted_indices]
        confidences = [float(proba[i, predicted_indices[i]]) for i in range(len(predicted_indices))]
        probabilities = [
            {label_names[j]: float(proba[i, j]) for j in range(proba.shape[1])}
            for i in range(proba.shape[0])
        ]
        return PredictionResult(
            predicted_classes=predicted_classes,
            probabilities=probabilities,
            confidences=confidences,
            feature_names=self.feature_names,
            label_names=label_names,
        )

    def predict_single(self, record: dict) -> PredictionResult:
        return self.predict([record])
