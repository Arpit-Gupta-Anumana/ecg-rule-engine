"""Per-disease markdown report generator.

Pulls together:
- Loaded Disease + variant definitions (with manual citations).
- Per-variant sens/spec/PPV/NPV/F1 + bootstrap CIs (metrics_table).
- Per-subgroup (sex, age bin) sensitivity/specificity (subgroup_table).
- GE cross-check agreement matrix (Agreement).
- k-of-N ensemble summary and/or CART distillation summary.
- A handful of fire-trace examples (true positives, false positives, false negatives).

The report is a self-contained markdown file per disease.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..dsl.schema import Disease
from .distill import CartDistilledModel, KofNModel
from .ge_crosscheck import Agreement
from .metrics import metrics_table, subgroup_table


@dataclass
class DiseaseReport:
    disease: Disease
    engine_output: pd.DataFrame          # per-ecg fire + trace_json columns
    y_true: pd.Series
    meta: pd.DataFrame                   # patient_id / sex / age_years
    split: pd.Series | None = None
    ge_labels: pd.Series | None = None
    agreement: Agreement | None = None
    k_of_n: KofNModel | None = None
    cart: CartDistilledModel | None = None
    test_metrics: dict[str, dict[str, float]] | None = None


def _md_df(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=float_fmt) if not df.empty else "_(no rows)_"


def _age_bin(age: float | None) -> str:
    if age is None or pd.isna(age):
        return "unknown"
    age = int(age)
    if age < 18: return "<18"
    if age < 40: return "18-39"
    if age < 60: return "40-59"
    if age < 80: return "60-79"
    return "80+"


def _format_variant(v) -> str:
    page = f" (manual p.{v.manual_page})" if v.manual_page is not None else ""
    return f"- **{v.name}**{page}: {v.description or '_no description_'}"


def _example_traces(engine_output: pd.DataFrame, y_true: pd.Series, k: int = 3) -> list[dict[str, Any]]:
    """Pick up to k true-positive, false-positive, and false-negative examples."""
    df = engine_output.copy()
    df["y_true"] = y_true.reindex(df.index).astype("Int64")
    out: list[dict[str, Any]] = []
    for label, mask in (
        ("True positive",  (df["disease_fired"] == 1) & (df["y_true"] == 1)),
        ("False positive", (df["disease_fired"] == 1) & (df["y_true"] == 0)),
        ("False negative", (df["disease_fired"] == 0) & (df["y_true"] == 1)),
    ):
        sub = df[mask].head(k)
        for ecg_id, row in sub.iterrows():
            try:
                tr = json.loads(row["trace_json"])
            except Exception:  # noqa: BLE001
                tr = None
            out.append({"kind": label, "ecg_id": ecg_id, "trace": tr})
    return out


def _trace_md(trace: dict, indent: int = 0) -> str:
    if not trace:
        return "_(no trace)_"
    prefix = "  " * indent
    marker = "x" if trace.get("fired") else " "
    note = f"  ({trace['note']})" if trace.get("note") else ""
    line = f"{prefix}- [{marker}] {trace.get('kind')}: {trace.get('detail','')}{note}"
    parts = [line]
    for c in trace.get("children", []) or []:
        parts.append(_trace_md(c, indent + 1))
    return "\n".join(parts)


def render_disease_report(rep: DiseaseReport, *, n_boot: int = 500) -> str:
    d = rep.disease
    parts: list[str] = []
    parts.append(f"# {d.disease} — rule engine report\n")
    parts.append(f"**Manual source:** {d.manual_source}\n")
    if d.description:
        parts.append(f"{d.description}\n")
    parts.append(f"**Exclusions:** {', '.join(d.exclusions) or '_(none)_'}\n")
    parts.append("## Variants encoded\n")
    for v in d.variants:
        parts.append(_format_variant(v))
    parts.append("")

    # Per-variant metrics
    variant_cols = [c for c in rep.engine_output.columns if c.startswith("fired_")]
    preds = {c[len("fired_"):]: rep.engine_output[c].astype(int) for c in variant_cols}
    preds["disease_OR"] = rep.engine_output["disease_fired"].astype(int)
    if rep.k_of_n is not None:
        preds[f"k_of_n(k={rep.k_of_n.k})"] = pd.Series(
            rep.k_of_n.predict(rep.engine_output[variant_cols].rename(
                columns=lambda c: c[len('fired_'):]
            )),
            index=rep.engine_output.index,
        )
    # align y_true index
    y = rep.y_true.reindex(rep.engine_output.index).fillna(0).astype(int)
    mt = metrics_table(y, preds, n_boot=n_boot)
    parts.append("## Per-variant metrics (with 95% bootstrap CI)\n")
    parts.append(_md_df(mt))
    parts.append("")

    # Subgroup tables (prefer disease-OR as the "engine flagging")
    if "sex" in rep.meta.columns:
        s = subgroup_table(y, preds["disease_OR"], rep.meta["sex"].reindex(y.index), n_boot=n_boot)
        parts.append("## Metrics by sex (disease_OR prediction)\n")
        parts.append(_md_df(s))
        parts.append("")
    if "age_years" in rep.meta.columns:
        age_bins = rep.meta["age_years"].reindex(y.index).map(_age_bin)
        s = subgroup_table(y, preds["disease_OR"], age_bins, n_boot=n_boot)
        parts.append("## Metrics by age band (disease_OR prediction)\n")
        parts.append(_md_df(s))
        parts.append("")

    # GE cross-check
    if rep.agreement is not None:
        a = rep.agreement
        parts.append("## GE device cross-check\n")
        parts.append(
            f"| | GE: positive | GE: negative |\n"
            f"|---|---|---|\n"
            f"| **ours: positive** | {a.both_positive} | {a.ours_only} |\n"
            f"| **ours: negative** | {a.ge_only} | {a.both_negative} |\n"
        )
        parts.append(
            f"\nN={a.n}, agreement_rate={a.agreement_rate():.3f}, "
            f"Cohen's κ={a.cohens_kappa():.3f}\n"
        )

    # Ensemble
    if rep.k_of_n is not None:
        parts.append("## k-of-N ensemble (val-selected)\n")
        parts.append("```\n" + rep.k_of_n.describe() + "\n```\n")

    # CART
    if rep.cart is not None:
        parts.append("## CART distillation (max_depth)\n")
        parts.append("```\n" + rep.cart.describe() + "\n```")
        if rep.cart.graphviz_source:
            parts.append("\n<details><summary>Graphviz source</summary>\n\n```\n" +
                         rep.cart.graphviz_source + "\n```\n</details>\n")

    # Held-out test metrics, if supplied
    if rep.test_metrics:
        parts.append("## Held-out test set metrics\n")
        df = pd.DataFrame(rep.test_metrics).T
        parts.append(_md_df(df))
        parts.append("")

    # Fire-trace examples
    examples = _example_traces(rep.engine_output, y)
    if examples:
        parts.append("## Fire-trace examples\n")
        for ex in examples:
            parts.append(f"### {ex['kind']} — ECG `{ex['ecg_id']}`\n")
            parts.append("```\n" + _trace_md(ex["trace"]) + "\n```\n")

    return "\n".join(parts) + "\n"


def write_disease_report(rep: DiseaseReport, out_dir: str | Path, *, n_boot: int = 500) -> Path:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    md = render_disease_report(rep, n_boot=n_boot)
    path = d / f"{rep.disease.disease}.md"
    path.write_text(md, encoding="utf-8")
    return path


__all__ = ["DiseaseReport", "render_disease_report", "write_disease_report"]
