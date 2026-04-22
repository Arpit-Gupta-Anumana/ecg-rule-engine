"""Run the rule engine across a whole feature DataFrame.

Produces:
- `run_engine` -> dict[disease -> DataFrame], each with one row per ECG and
  columns: `disease_fired`, `fired_<variant>` for every variant, and
  `excluded_by` (| separated), plus `trace_json` (compact).
- `variant_prediction_matrix` -> wide DataFrame (ecg_id x variants) of 0/1.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..dsl.schema import Disease
from ..engine.evaluator import (
    DiseaseResult,
    EvalContext,
    evaluate_disease,
)


def _row_context(row: pd.Series) -> EvalContext:
    feats: dict[str, float] = {}
    sex: str | None = None
    age: int | None = None
    for k, v in row.items():
        if k == "sex":
            sex = str(v) if pd.notna(v) else None
            continue
        if k == "age_years":
            if pd.notna(v):
                age = int(v)
            continue
        if pd.notna(v) and isinstance(v, (int, float)):
            feats[str(k)] = float(v)
    return EvalContext(features=feats, sex=sex, age_years=age)


def run_disease(disease: Disease, features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ecg_id, row in features.iterrows():
        ctx = _row_context(row)
        r: DiseaseResult = evaluate_disease(disease, ctx)
        rec: dict[str, Any] = {
            "ecg_id": ecg_id,
            "disease_fired": int(r.fired),
            "excluded_by": "|".join(r.excluded_by),
        }
        for v in r.variant_results:
            rec[f"fired_{v.variant_name}"] = int(v.fired)
        rec["trace_json"] = json.dumps(r.to_dict(), separators=(",", ":"))
        rows.append(rec)
    return pd.DataFrame(rows).set_index("ecg_id")


def run_engine(
    diseases: dict[str, Disease],
    features: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {name: run_disease(d, features) for name, d in diseases.items()}


def variant_prediction_matrix(
    run_result: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Flatten to one column per (disease, variant) for ensemble training."""
    pieces: list[pd.DataFrame] = []
    for disease, df in run_result.items():
        fired_cols = [c for c in df.columns if c.startswith("fired_")]
        sub = df[fired_cols].copy()
        sub.columns = [f"{disease}::{c[len('fired_'):]}" for c in fired_cols]
        pieces.append(sub)
    if not pieces:
        return pd.DataFrame()
    return pd.concat(pieces, axis=1)


__all__ = ["run_disease", "run_engine", "variant_prediction_matrix"]
