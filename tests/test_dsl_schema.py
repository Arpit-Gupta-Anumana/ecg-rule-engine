"""Tests for the Rule DSL pydantic schema + YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from ecg_rule_engine.dsl.loader import RuleLoadError, load_disease_yaml, load_rules_dir
from ecg_rule_engine.dsl.schema import (
    AllOfClause,
    AnyOfClause,
    Disease,
    PointScoreClause,
    SexThreshold,
    ThresholdClause,
    disease_features,
)


MIN_VALID = {
    "disease": "Test",
    "manual_source": "Test manual p.1",
    "variants": [
        {
            "name": "simple",
            "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 120},
        }
    ],
}


def test_minimal_disease_loads():
    d = Disease.model_validate(MIN_VALID)
    assert d.disease == "Test"
    assert len(d.variants) == 1
    assert isinstance(d.variants[0].clause, ThresholdClause)


def test_extra_fields_rejected():
    bad = {**MIN_VALID, "bogus_top_level": 42}
    with pytest.raises(ValidationError):
        Disease.model_validate(bad)


def test_unknown_feature_rejected():
    bad = {
        "disease": "X",
        "manual_source": "p.1",
        "variants": [
            {
                "name": "v1",
                "clause": {"kind": "threshold", "expr": "bogus_feature_xyz", "op": ">", "value": 1},
            }
        ],
    }
    with pytest.raises(ValidationError):
        Disease.model_validate(bad)


def test_sex_threshold():
    d = Disease.model_validate({
        "disease": "LVH",
        "manual_source": "p.148",
        "sex_specific": True,
        "variants": [{
            "name": "Cornell",
            "clause": {
                "kind": "threshold",
                "expr": "R_aVL_mV + S_V3_mV",
                "op": ">",
                "value": {"M": 2.8, "F": 2.0},
            },
        }],
    })
    tc = d.variants[0].clause
    assert isinstance(tc, ThresholdClause)
    assert isinstance(tc.value, SexThreshold)
    assert tc.value.M == 2.8


def test_nested_all_of_any_of():
    d = Disease.model_validate({
        "disease": "X",
        "manual_source": "p.1",
        "variants": [{
            "name": "nested",
            "clause": {
                "kind": "all_of",
                "clauses": [
                    {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 120},
                    {
                        "kind": "any_of",
                        "clauses": [
                            {"kind": "threshold", "expr": "R_V1_mV", "op": ">", "value": 1.5},
                            {"kind": "threshold", "expr": "R_V2_mV", "op": ">", "value": 1.5},
                        ],
                    },
                ],
            },
        }],
    })
    top = d.variants[0].clause
    assert isinstance(top, AllOfClause)
    assert isinstance(top.clauses[1], AnyOfClause)


def test_point_score():
    d = Disease.model_validate({
        "disease": "LVH",
        "manual_source": "p.149",
        "variants": [{
            "name": "Romhilt_Estes",
            "clause": {
                "kind": "point_score",
                "threshold_positive": 5,
                "items": [
                    {"points": 3, "when": {"kind": "threshold", "expr": "R_aVL_mV", "op": ">=", "value": 1.1}},
                    {"points": 3, "when": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 90}},
                ],
            },
        }],
    })
    ps = d.variants[0].clause
    assert isinstance(ps, PointScoreClause)
    assert ps.threshold_positive == 5
    assert len(ps.items) == 2


def test_duplicate_variant_names_rejected():
    bad = {
        "disease": "X",
        "manual_source": "p.1",
        "variants": [
            {"name": "same", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">", "value": 1}},
            {"name": "same", "clause": {"kind": "threshold", "expr": "QT_ms", "op": ">", "value": 1}},
        ],
    }
    with pytest.raises(ValidationError):
        Disease.model_validate(bad)


def test_disease_features_collects_all():
    d = Disease.model_validate({
        "disease": "X",
        "manual_source": "p.1",
        "variants": [
            {"name": "v1", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">", "value": 120}},
            {"name": "v2", "clause": {"kind": "threshold", "expr": "R_aVL_mV + S_V3_mV", "op": ">", "value": 2.8}},
        ],
    })
    feats = disease_features(d)
    assert feats == {"QRS_ms", "R_aVL_mV", "S_V3_mV"}


def test_yaml_loader_roundtrip(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(MIN_VALID))
    d = load_disease_yaml(p)
    assert d.disease == "Test"


def test_yaml_loader_bad_yaml(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(": : not yaml :\n  [\n")
    with pytest.raises(RuleLoadError):
        load_disease_yaml(p)


def test_rules_dir_dedup(tmp_path: Path):
    (tmp_path / "a.yaml").write_text(yaml.safe_dump(MIN_VALID))
    (tmp_path / "b.yaml").write_text(yaml.safe_dump(MIN_VALID))  # same disease name
    with pytest.raises(RuleLoadError):
        load_rules_dir(tmp_path)
