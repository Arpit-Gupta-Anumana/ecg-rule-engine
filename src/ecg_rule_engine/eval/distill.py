"""Ensemble and distillation of named criteria.

Two strategies:

1. k-of-N ensemble (preferred, fully interpretable):
   Given M named criteria (columns in X ∈ {0,1}^{n×M}), flag the disease
   whenever at least k of the M fire. Sweep k on the val set, pick the k that
   maximizes F1 (or sensitivity subject to min specificity, configurable).

2. Shallow CART distillation:
   Train sklearn DecisionTreeClassifier(max_depth<=4) on [variant fires +
   selected raw features]. Produces a small decision tree you can print or
   render to Graphviz.

Both return an `EnsembleModel` object with `predict(X) -> np.ndarray[0/1]`
and a `describe()` method that yields a human-readable summary.

Train / val / test split discipline is the caller's responsibility — pass in
pre-split X_train / X_val / y_train / y_val.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from .metrics import compute_metrics


# ----------------------------------------------------------------------------
# k-of-N ensemble
# ----------------------------------------------------------------------------

@dataclass
class KofNModel:
    variant_columns: list[str]
    k: int
    val_f1: float
    val_sensitivity: float
    val_specificity: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        votes = X[self.variant_columns].astype(int).sum(axis=1).values
        return (votes >= self.k).astype(int)

    def describe(self) -> str:
        return (
            f"k-of-N ensemble: fire if >= {self.k} of {len(self.variant_columns)} "
            f"variants fire.\n"
            f"  variants: {self.variant_columns}\n"
            f"  val F1={self.val_f1:.3f}  sens={self.val_sensitivity:.3f}  "
            f"spec={self.val_specificity:.3f}"
        )


def train_k_of_n(
    X_train_variants: pd.DataFrame,
    y_train: pd.Series,
    X_val_variants: pd.DataFrame,
    y_val: pd.Series,
    *,
    optimize_for: Literal["f1", "sensitivity_at_specificity"] = "f1",
    min_specificity: float = 0.90,
) -> KofNModel:
    """Sweep k = 1..N on the val set and pick the best.

    If `optimize_for="sensitivity_at_specificity"`, picks the k that maximizes
    sensitivity subject to specificity >= `min_specificity`; if nothing meets
    that bar, falls back to the k with max F1.
    """
    variants = list(X_train_variants.columns)
    votes_val = X_val_variants[variants].astype(int).sum(axis=1).values
    best = None
    best_score = -1.0

    for k in range(1, len(variants) + 1):
        preds = (votes_val >= k).astype(int)
        m = compute_metrics(y_val.values, preds)
        if optimize_for == "f1":
            score = m.f1 if not np.isnan(m.f1) else -1.0
        else:
            if m.specificity >= min_specificity:
                score = m.sensitivity if not np.isnan(m.sensitivity) else -1.0
            else:
                score = -1.0
        if score > best_score:
            best_score = score
            best = (k, m)

    if best is None:  # fallback
        k = 1
        preds = (votes_val >= k).astype(int)
        m = compute_metrics(y_val.values, preds)
        best = (k, m)

    k, m = best
    return KofNModel(
        variant_columns=variants,
        k=k,
        val_f1=float(m.f1) if not np.isnan(m.f1) else 0.0,
        val_sensitivity=float(m.sensitivity) if not np.isnan(m.sensitivity) else 0.0,
        val_specificity=float(m.specificity) if not np.isnan(m.specificity) else 0.0,
    )


# ----------------------------------------------------------------------------
# Shallow CART distillation
# ----------------------------------------------------------------------------

@dataclass
class CartDistilledModel:
    feature_columns: list[str]
    model: object  # sklearn DecisionTreeClassifier; type-free to avoid hard dep at import time
    val_f1: float
    val_sensitivity: float
    val_specificity: float
    graphviz_source: str | None = field(default=None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        from sklearn.tree import DecisionTreeClassifier  # noqa: F401 - sanity check sklearn present
        return self.model.predict(X[self.feature_columns].values).astype(int)  # type: ignore[attr-defined]

    def describe(self) -> str:
        from sklearn.tree import export_text  # type: ignore
        return export_text(self.model, feature_names=self.feature_columns)  # type: ignore[arg-type]


def train_cart(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    max_depth: int = 3,
    min_samples_leaf: int = 20,
    class_weight: str | dict | None = "balanced",
    random_state: int = 0,
) -> CartDistilledModel:
    from sklearn.tree import DecisionTreeClassifier, export_graphviz  # type: ignore

    features = list(X_train.columns)
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(X_train.values, y_train.values)
    preds = model.predict(X_val.values)
    m = compute_metrics(y_val.values, preds)
    try:
        gviz = export_graphviz(
            model,
            feature_names=features,
            class_names=["neg", "pos"],
            filled=True, impurity=False, proportion=True,
        )
    except Exception:  # noqa: BLE001
        gviz = None
    return CartDistilledModel(
        feature_columns=features,
        model=model,
        val_f1=float(m.f1) if not np.isnan(m.f1) else 0.0,
        val_sensitivity=float(m.sensitivity) if not np.isnan(m.sensitivity) else 0.0,
        val_specificity=float(m.specificity) if not np.isnan(m.specificity) else 0.0,
        graphviz_source=gviz,
    )


__all__ = [
    "KofNModel",
    "train_k_of_n",
    "CartDistilledModel",
    "train_cart",
]
