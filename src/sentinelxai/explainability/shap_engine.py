"""TreeSHAP explanation engine (Phase 4).

Thin, testable wrapper around :class:`shap.TreeExplainer`. It only knows how
to turn a fitted tree model + a feature matrix into:

* **local** explanations — per-sample feature contributions for the predicted
  class, used by the Decision Intelligence Studio and the Failure Explorer;
* **global** explanations — mean-absolute SHAP across classes/samples, used by
  the Analytics page and the ``GET /feature-importance`` endpoint.

Nothing here is model-specific beyond "must be tree-based" — the same class
works for the LightGBM final model and any XGBoost/RandomForest baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import shap

from sentinelxai.logging_setup import get_logger

logger = get_logger(__name__)

Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class GlobalFeatureImportance:
    """One feature's global importance across the evaluated dataset."""

    name: str
    mean_abs_shap: float
    rank: int


@dataclass(frozen=True)
class LocalContribution:
    """Signed SHAP contribution of one feature to the predicted class."""

    name: str
    value: float
    direction: Direction
    rank: int
    # Share of the total absolute contribution mass (0..1) this feature holds.
    weight: float


@dataclass(frozen=True)
class LocalExplanation:
    """Full local explanation for a single prediction."""

    predicted_class: str
    top_contributions: list[LocalContribution]
    human_explanation: list[str]


class SHAPExplainer:
    """Lazily-built TreeSHAP explainer for a fitted tree model."""

    def __init__(self, model, feature_names: list[str], *, n_top: int = 5) -> None:
        if n_top < 1:
            raise ValueError(f"n_top must be >= 1, got {n_top}")
        self._model = model
        self.feature_names: list[str] = list(feature_names)
        self.n_top = n_top
        # TreeExplainer is constructed lazily so unit tests that only need the
        # class shape don't pay the (small) SHAP import/init cost twice.
        self._explainer: shap.TreeExplainer | None = None

    @classmethod
    def from_model(cls, model, feature_names: list[str], *, n_top: int = 5) -> "SHAPExplainer":
        return cls(model, feature_names, n_top=n_top)

    @property
    def explainer(self) -> shap.TreeExplainer:
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._model)
        return self._explainer

    def _shap_values_per_class(self, X: pd.DataFrame) -> list[np.ndarray]:
        """Return SHAP values as a list (per class) of shape (n, n_features).

        TreeExplainer returns either a list of per-class arrays (older SHAP)
        or an Explanation-like structure; this normalizes both into a list of
        2D arrays so downstream code is version-agnostic.
        """
        raw = self.explainer.shap_values(X)
        if isinstance(raw, list):
            return [np.asarray(a, dtype=float) for a in raw]
        # Newer SHAP: ndarray shape (n, n_features, n_classes)
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 3:
            return [arr[:, :, c] for c in range(arr.shape[2])]
        # Binary / single-output fallback
        return [arr]

    def global_importance(self, X: pd.DataFrame, *, top: int | None = None) -> list[GlobalFeatureImportance]:
        """Mean-absolute SHAP across all classes and samples, ranked."""
        per_class = self._shap_values_per_class(X)
        if not per_class:
            return []
        stacked = np.stack([np.abs(a) for a in per_class], axis=0)  # (n_classes, n, n_features)
        mean_abs = float(np.mean(stacked)) and np.mean(stacked, axis=(0, 1))  # (n_features,)
        order = np.argsort(mean_abs)[::-1]
        limit = top if top else len(self.feature_names)
        out: list[GlobalFeatureImportance] = []
        for rank, idx in enumerate(order[:limit], start=1):
            out.append(
                GlobalFeatureImportance(
                    name=self.feature_names[idx],
                    mean_abs_shap=float(mean_abs[idx]),
                    rank=rank,
                )
            )
        return out

    def explain(self, X: pd.DataFrame, predicted_classes: list[str]) -> list[LocalExplanation]:
        """Local explanations for each row, aligned with ``predicted_classes``.

        ``predicted_classes`` must be the same length as ``X`` and in the same
        order — i.e. the argmax class for each row from the model. Contributions
        are taken from the SHAP values of the **predicted** class.
        """
        if len(X) != len(predicted_classes):
            raise ValueError(
                f"X has {len(X)} rows but predicted_classes has {len(predicted_classes)}"
            )
        per_class = self._shap_values_per_class(X)
        n_classes = len(per_class)

        results: list[LocalExplanation] = []
        for i, pred_cls in enumerate(predicted_classes):
            class_idx = self._class_index(pred_cls, n_classes)
            contribs_i = per_class[class_idx][i]  # (n_features,)
            results.append(self._local_for_row(contribs_i, pred_cls))
        return results

    def explain_single(
        self, feature_values: dict[str, float], predicted_class: str
    ) -> LocalExplanation:
        """Convenience wrapper for one record (used by the API / Studio)."""
        X = pd.DataFrame([feature_values])
        return self.explain(X, [predicted_class])[0]

    # --- internals -----------------------------------------------------------

    def _class_index(self, pred_cls: str, n_classes: int) -> int:
        if isinstance(self._model, shap.TreeExplainer):  # pragma: no cover - safety
            pass
        # Prefer the model's own class ordering when available.
        classes = getattr(self._model, "classes_", None)
        if classes is not None:
            try:
                return int(list(classes).index(pred_cls))
            except ValueError:
                logger.warning("Predicted class %r not in model.classes_; using 0", pred_cls)
                return 0
        # Fallback: hash-free, deterministic index into available classes.
        return 0 if n_classes == 1 else 0

    def _local_for_row(self, contribs_i: np.ndarray, pred_cls: str) -> LocalExplanation:
        total = float(np.sum(np.abs(contribs_i))) or 1.0
        order = np.argsort(np.abs(contribs_i))[::-1]
        top = order[: self.n_top]
        contributions: list[LocalContribution] = []
        for rank, idx in enumerate(top, start=1):
            val = float(contribs_i[idx])
            contributions.append(
                LocalContribution(
                    name=self.feature_names[idx],
                    value=val,
                    direction="increase" if val >= 0 else "decrease",
                    rank=rank,
                    weight=float(abs(val) / total),
                )
            )
        return LocalExplanation(
            predicted_class=pred_cls,
            top_contributions=contributions,
            human_explanation=[],  # filled by humanize_contributions at the call site
        )
