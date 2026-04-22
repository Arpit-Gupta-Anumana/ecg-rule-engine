"""Binary classification metrics + bootstrap CIs + subgroup breakdowns.

Everything is intentionally written against numpy/pandas arrays so it can be
used without sklearn (which we do still use for the distillation step).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinaryMetrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    sensitivity: float   # = recall = TPR
    specificity: float   # = TNR
    ppv: float           # = precision
    npv: float
    accuracy: float
    f1: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "sensitivity": self.sensitivity, "specificity": self.specificity,
            "ppv": self.ppv, "npv": self.npv,
            "accuracy": self.accuracy, "f1": self.f1,
        }


def _safe(num: float, denom: float) -> float:
    return float(num / denom) if denom > 0 else float("nan")


def compute_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> BinaryMetrics:
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    tp = int(((yt == 1) & (yp == 1)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    sens = _safe(tp, tp + fn)
    spec = _safe(tn, tn + fp)
    ppv = _safe(tp, tp + fp)
    npv = _safe(tn, tn + fn)
    acc = _safe(tp + tn, tp + fp + tn + fn)
    prec = ppv
    rec = sens
    f1 = _safe(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
    return BinaryMetrics(
        n=int(len(yt)), tp=tp, fp=fp, tn=tn, fn=fn,
        sensitivity=sens, specificity=spec, ppv=ppv, npv=npv, accuracy=acc, f1=f1,
    )


def bootstrap_ci(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    metric: str = "f1",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (point_estimate, lower, upper) of a metric via percentile bootstrap."""
    yt = np.asarray(y_true).astype(int)
    yp = np.asarray(y_pred).astype(int)
    n = len(yt)
    if n == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)

    def _metric(y_t, y_p) -> float:
        m = compute_metrics(y_t, y_p)
        return getattr(m, metric)

    point = _metric(yt, yp)
    stats = np.empty(n_boot)
    idx = np.arange(n)
    for b in range(n_boot):
        pick = rng.choice(idx, size=n, replace=True)
        stats[b] = _metric(yt[pick], yp[pick])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return point, lo, hi


def metrics_table(
    y_true: pd.Series,
    preds: dict[str, pd.Series],
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per prediction series (variant or ensemble); columns are metrics w/ CIs."""
    rows: list[dict] = []
    for name, yp in preds.items():
        m = compute_metrics(y_true, yp)
        row: dict = m.to_dict()
        row["variant"] = name
        for metric in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"):
            p, lo, hi = bootstrap_ci(y_true, yp, metric=metric, n_boot=n_boot, seed=seed)
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variant")
    # order columns nicely
    lead = ["n", "tp", "fp", "tn", "fn"]
    metric_cols: list[str] = []
    for m in ("sensitivity", "specificity", "ppv", "npv", "f1", "accuracy"):
        metric_cols += [m, f"{m}_lo", f"{m}_hi"]
    return df[lead + metric_cols]


def subgroup_table(
    y_true: pd.Series,
    y_pred: pd.Series,
    subgroup: pd.Series,
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Compute metrics within each level of `subgroup`."""
    rows: list[dict] = []
    for key, mask in subgroup.groupby(subgroup, dropna=False).groups.items():
        idx = mask
        yt_sub = y_true.loc[idx]
        yp_sub = y_pred.loc[idx]
        if len(yt_sub) == 0:
            continue
        m = compute_metrics(yt_sub, yp_sub)
        row: dict = {"subgroup": key, **m.to_dict()}
        for metric in ("sensitivity", "specificity", "f1"):
            p, lo, hi = bootstrap_ci(yt_sub, yp_sub, metric=metric, n_boot=n_boot, seed=seed)
            row[f"{metric}_lo"] = lo
            row[f"{metric}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows).set_index("subgroup")


__all__ = [
    "BinaryMetrics",
    "compute_metrics",
    "bootstrap_ci",
    "metrics_table",
    "subgroup_table",
]
