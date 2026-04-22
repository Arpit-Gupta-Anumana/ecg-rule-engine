"""Deterministic rule evaluator for the ECG Rule DSL.

Evaluates a `Disease` against a feature dict and returns a `DiseaseResult`
containing fire results for every variant, an exclusion flag, and a full
`TraceNode` tree describing exactly which clauses matched with which values.

Semantics:

- If any `exclusion` flag in `disease.exclusions` is truthy (>=1) in the input
  feature vector, NO variant fires; we return `excluded_by` listing the
  offending exclusions. This models the manual's "don't diagnose LVH if LBBB"
  conventions explicitly at the disease level.
- `ThresholdClause.value` may be a scalar, a `SexThreshold`, or `AgeThreshold`.
  Sex/age resolution uses `ctx.sex` / `ctx.age_years` from the `EvalContext`.
- Missing features in a `ThresholdClause.expr` make the clause EVALUATE TO FALSE
  (not raise). We record the missing features in the trace under `missing`.
  This way, a rule-set run on a partial feature vector degrades gracefully.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from ..dsl.expr import ExprError, parse_expr
from ..dsl.schema import (
    AgeThreshold,
    AllOfClause,
    AnyOfClause,
    Clause,
    Disease,
    NotClause,
    PointScoreClause,
    SexThreshold,
    ThresholdClause,
    ThresholdValue,
    Variant,
)


@dataclass(frozen=True)
class EvalContext:
    """Per-ECG context for rule evaluation.

    `features` is augmented on access with auto-derived `sex_M_flag` and
    `sex_F_flag` (from `sex`) so DSL expressions can reference them directly.
    """
    features: Mapping[str, float]
    sex: str | None = None  # "M" / "F"
    age_years: int | None = None

    def effective_features(self) -> dict[str, float]:
        out = dict(self.features)
        out.setdefault("sex_M_flag", 1.0 if self.sex == "M" else 0.0)
        out.setdefault("sex_F_flag", 1.0 if self.sex == "F" else 0.0)
        if self.age_years is not None:
            out.setdefault("age_years", float(self.age_years))
        return out


# --- Trace tree --------------------------------------------------------------

@dataclass
class TraceNode:
    """A single node in the per-variant explanation tree."""
    kind: str
    fired: bool
    detail: str
    children: list["TraceNode"] = field(default_factory=list)
    missing_features: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "fired": self.fired,
            "detail": self.detail,
            "note": self.note,
            "missing_features": self.missing_features,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class VariantResult:
    variant_name: str
    fired: bool
    trace: TraceNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant_name,
            "fired": self.fired,
            "trace": self.trace.to_dict(),
        }


@dataclass
class DiseaseResult:
    disease: str
    fired: bool  # disease-level OR across variants, gated by exclusions
    excluded_by: list[str] = field(default_factory=list)
    variant_results: list[VariantResult] = field(default_factory=list)

    def fired_variants(self) -> list[str]:
        return [v.variant_name for v in self.variant_results if v.fired]

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "fired": self.fired,
            "excluded_by": self.excluded_by,
            "fired_variants": self.fired_variants(),
            "variants": [v.to_dict() for v in self.variant_results],
        }


# --- Value resolution --------------------------------------------------------

def _resolve_threshold(value: ThresholdValue, ctx: EvalContext) -> tuple[float | None, str]:
    """Resolve a threshold value to a scalar, returning (scalar, human-readable tag)."""
    if isinstance(value, (int, float)):
        return float(value), _fmt(float(value))
    if isinstance(value, SexThreshold):
        if ctx.sex == "M":
            return value.M, f"{_fmt(value.M)} [M]"
        if ctx.sex == "F":
            return value.F, f"{_fmt(value.F)} [F]"
        if value.default is not None:
            return value.default, f"{_fmt(value.default)} [default, sex unknown]"
        return None, "sex-specific threshold but sex is unknown"
    if isinstance(value, AgeThreshold):
        if ctx.age_years is None:
            return None, "age-specific threshold but age is unknown"
        for band in value.bands:
            if ctx.age_years <= band.max_age:
                return band.value, f"{_fmt(band.value)} [age<={band.max_age}]"
        return None, f"no age band matches age={ctx.age_years}"
    raise TypeError(f"Unsupported threshold value type: {type(value).__name__}")


def _fmt(x: float) -> str:
    if x == int(x):
        return str(int(x))
    return f"{x:.3f}".rstrip("0").rstrip(".")


_OPS = {
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


# --- Clause evaluation -------------------------------------------------------

def _eval_threshold(clause: ThresholdClause, ctx: EvalContext) -> TraceNode:
    node = parse_expr(clause.expr)
    env = ctx.effective_features()
    missing = [f for f in node.features() if f not in env]
    thr, thr_tag = _resolve_threshold(clause.value, ctx)

    if thr is None or missing:
        return TraceNode(
            kind="threshold",
            fired=False,
            detail=f"{clause.expr} {clause.op} {thr_tag}  [not evaluable]",
            missing_features=missing,
            note=clause.note,
        )

    try:
        actual = node.eval(env)
        pretty = node.pretty(env)
    except ExprError as e:
        return TraceNode(
            kind="threshold",
            fired=False,
            detail=f"{clause.expr}  [eval error: {e}]",
            missing_features=missing,
            note=clause.note,
        )

    fired = _OPS[clause.op](actual, thr)
    detail = f"{pretty} = {_fmt(actual)} {clause.op} {thr_tag}  -> {fired}"
    return TraceNode(
        kind="threshold",
        fired=fired,
        detail=detail,
        note=clause.note,
    )


def _eval_clause(clause: Clause, ctx: EvalContext) -> TraceNode:
    if isinstance(clause, ThresholdClause):
        return _eval_threshold(clause, ctx)

    if isinstance(clause, AllOfClause):
        children = [_eval_clause(c, ctx) for c in clause.clauses]
        fired = all(c.fired for c in children)
        return TraceNode(
            kind="all_of",
            fired=fired,
            detail=f"all_of ({sum(c.fired for c in children)}/{len(children)} matched)",
            children=children,
            note=clause.note,
        )

    if isinstance(clause, AnyOfClause):
        children = [_eval_clause(c, ctx) for c in clause.clauses]
        fired = any(c.fired for c in children)
        return TraceNode(
            kind="any_of",
            fired=fired,
            detail=f"any_of ({sum(c.fired for c in children)}/{len(children)} matched)",
            children=children,
            note=clause.note,
        )

    if isinstance(clause, NotClause):
        inner = _eval_clause(clause.clause, ctx)
        return TraceNode(
            kind="not_",
            fired=not inner.fired,
            detail=f"not_  (inner={inner.fired})",
            children=[inner],
            note=clause.note,
        )

    if isinstance(clause, PointScoreClause):
        total = 0
        children: list[TraceNode] = []
        for item in clause.items:
            sub = _eval_clause(item.when, ctx)
            awarded = item.points if sub.fired else 0
            total += awarded
            children.append(TraceNode(
                kind="point_score_item",
                fired=sub.fired,
                detail=f"awarded {awarded}/{item.points} pts",
                children=[sub],
                note=item.note,
            ))
        fired = total >= clause.threshold_positive
        return TraceNode(
            kind="point_score",
            fired=fired,
            detail=f"score={total} (threshold={clause.threshold_positive})",
            children=children,
            note=clause.note,
        )

    raise TypeError(f"Unsupported clause type: {type(clause).__name__}")


# --- Public API --------------------------------------------------------------

def evaluate_variant(variant: Variant, ctx: EvalContext) -> VariantResult:
    trace = _eval_clause(variant.clause, ctx)
    return VariantResult(variant_name=variant.name, fired=trace.fired, trace=trace)


def evaluate_disease(disease: Disease, ctx: EvalContext) -> DiseaseResult:
    # Exclusion gate: if any exclusion flag is present AND truthy, nothing fires.
    excluded_by = []
    env = ctx.effective_features()
    for excl in disease.exclusions:
        feat_name = excl if excl.endswith("_flag") else f"{excl.lower()}_flag"
        val = env.get(feat_name)
        if val is not None and val >= 1:
            excluded_by.append(excl)

    variant_results = [evaluate_variant(v, ctx) for v in disease.variants]
    disease_fired = (not excluded_by) and any(v.fired for v in variant_results)
    return DiseaseResult(
        disease=disease.disease,
        fired=disease_fired,
        excluded_by=excluded_by,
        variant_results=variant_results,
    )


def evaluate_all(
    diseases: dict[str, Disease],
    ctx: EvalContext,
) -> dict[str, DiseaseResult]:
    return {name: evaluate_disease(d, ctx) for name, d in diseases.items()}


__all__ = [
    "EvalContext",
    "TraceNode",
    "VariantResult",
    "DiseaseResult",
    "evaluate_variant",
    "evaluate_disease",
    "evaluate_all",
]
