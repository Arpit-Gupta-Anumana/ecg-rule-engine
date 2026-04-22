"""Pydantic schema for the Rule DSL used to encode the 12SL Physician's Guide.

Top-level YAML shape (one file per disease):

    disease: LVH
    manual_source: "12SL Physicians Guide v24, p.147-152"
    exclusions: [LBBB, WPW, VentricularPacing]
    sex_specific: true
    variants:
      - name: Sokolow_Lyon
        manual_page: 147
        clause:
          kind: threshold
          expr: "max(S_V1_mV, S_V2_mV) + max(R_V5_mV, R_V6_mV)"
          op: ">="
          value: 3.5
      - name: Cornell_voltage
        manual_page: 148
        clause:
          kind: threshold
          expr: "R_aVL_mV + S_V3_mV"
          op: ">"
          value: { M: 2.8, F: 2.0 }
      - name: Romhilt_Estes
        manual_page: 149
        clause:
          kind: point_score
          threshold_positive: 5
          items:
            - points: 3
              when: { kind: threshold, expr: "max(abs(R_aVL_mV), abs(S_V1_mV))", op: ">=", value: 2.0 }
            - points: 3
              when: { kind: threshold, expr: "QRS_ms", op: ">=", value: 90 }

A `clause` is a recursive sum-type: `all_of`, `any_of`, `not_`, `threshold`, or
`point_score`. Each variant has exactly one top-level `clause`. The `exclusions`
field applies to ALL variants of the disease (if any exclusion flag is true, no
variant fires).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .expr import ExprError, parse_expr, validate_features

# ----------------------------------------------------------------------------
# Leaf value types
# ----------------------------------------------------------------------------

Op = Literal[">", ">=", "<", "<=", "==", "!="]


class SexThreshold(BaseModel):
    """A threshold that differs by sex. `M` and `F` are required; `default` optional."""
    model_config = ConfigDict(extra="forbid")
    M: float
    F: float
    default: float | None = None


class AgeThreshold(BaseModel):
    """A piecewise-constant threshold by age band, evaluated top-to-bottom.

    Example:
        age_bands:
          - { max_age: 40, value: 3.0 }
          - { max_age: 999, value: 2.5 }
    """
    model_config = ConfigDict(extra="forbid")
    bands: list["AgeBand"]


class AgeBand(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_age: int
    value: float


ThresholdValue = Union[float, int, SexThreshold, AgeThreshold]


# ----------------------------------------------------------------------------
# Clause types (recursive sum)
# ----------------------------------------------------------------------------

class ThresholdClause(BaseModel):
    """A single comparison: `expr` compared to `value` using `op`."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["threshold"] = "threshold"
    expr: str
    op: Op
    value: ThresholdValue
    note: str | None = None

    @field_validator("expr")
    @classmethod
    def _validate_expr(cls, v: str) -> str:
        try:
            node = parse_expr(v)
        except ExprError as e:
            raise ValueError(f"Invalid expression {v!r}: {e}") from e
        unknown = validate_features(node)
        if unknown:
            raise ValueError(f"Expression {v!r} references unknown features: {unknown}")
        return v


class AllOfClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["all_of"] = "all_of"
    clauses: list["Clause"] = Field(min_length=1)
    note: str | None = None


class AnyOfClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["any_of"] = "any_of"
    clauses: list["Clause"] = Field(min_length=1)
    note: str | None = None


class NotClause(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["not_"] = "not_"
    clause: "Clause"
    note: str | None = None


class PointScoreItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    points: int
    when: "Clause"
    note: str | None = None


class PointScoreClause(BaseModel):
    """Sum `points` of every item whose `when` clause fires; true iff sum >= threshold."""
    model_config = ConfigDict(extra="forbid")
    kind: Literal["point_score"] = "point_score"
    threshold_positive: int
    items: list[PointScoreItem] = Field(min_length=1)
    note: str | None = None


Clause = Annotated[
    Union[ThresholdClause, AllOfClause, AnyOfClause, NotClause, PointScoreClause],
    Field(discriminator="kind"),
]


# Resolve forward refs
AllOfClause.model_rebuild()
AnyOfClause.model_rebuild()
NotClause.model_rebuild()
PointScoreItem.model_rebuild()
PointScoreClause.model_rebuild()


# ----------------------------------------------------------------------------
# Variant + Disease
# ----------------------------------------------------------------------------

class Variant(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    manual_page: int | None = None
    clause: Clause
    description: str | None = None
    # Some named criteria are explicitly sex-specific in the manual; this flag is
    # informational — the actual sex dependence lives inside SexThreshold values.
    sex_specific: bool = False

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("variant name must be non-empty")
        if any(c.isspace() for c in v):
            raise ValueError(f"variant name {v!r} must not contain whitespace")
        return v


class Disease(BaseModel):
    """A disease ruleset: exclusions + one or more named variants."""
    model_config = ConfigDict(extra="forbid")
    disease: str
    manual_source: str
    description: str | None = None
    # Free-text explanation of places where this YAML intentionally departs from
    # the literal manual wording (e.g. using widely-accepted literature
    # thresholds where the manual only gestures at a criterion, encoding
    # sensor-proxy features where the manual presumes morphology detectors,
    # etc). Must be reviewed and signed off by a clinician before promotion.
    manual_departure_notes: str | None = None
    exclusions: list[str] = Field(default_factory=list)
    sex_specific: bool = False
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_variant_names(self) -> "Disease":
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            dup = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate variant names in {self.disease}: {sorted(dup)}")
        return self


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def walk_clauses(clause: Clause):
    """Yield every Clause in a rule tree (pre-order)."""
    yield clause
    if isinstance(clause, (AllOfClause, AnyOfClause)):
        for c in clause.clauses:
            yield from walk_clauses(c)
    elif isinstance(clause, NotClause):
        yield from walk_clauses(clause.clause)
    elif isinstance(clause, PointScoreClause):
        for item in clause.items:
            yield from walk_clauses(item.when)


def disease_features(disease: Disease) -> set[str]:
    """Collect every feature name referenced by any clause in a disease."""
    feats: set[str] = set()
    for v in disease.variants:
        for c in walk_clauses(v.clause):
            if isinstance(c, ThresholdClause):
                feats |= parse_expr(c.expr).features()
    return feats


__all__ = [
    "Op",
    "SexThreshold", "AgeThreshold", "AgeBand", "ThresholdValue",
    "ThresholdClause", "AllOfClause", "AnyOfClause", "NotClause",
    "PointScoreClause", "PointScoreItem", "Clause",
    "Variant", "Disease",
    "walk_clauses", "disease_features",
]
