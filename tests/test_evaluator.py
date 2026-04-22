"""Tests for the deterministic rule evaluator (all DSL node kinds + exclusions)."""

from __future__ import annotations

import pytest

from ecg_rule_engine.dsl.schema import Disease
from ecg_rule_engine.engine.evaluator import (
    EvalContext,
    evaluate_disease,
    evaluate_variant,
)
from ecg_rule_engine.engine.trace import format_disease_result


def _disease(dct):
    return Disease.model_validate(dct)


def test_threshold_fires_when_above():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{"name": "v", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 120}}],
    })
    r = evaluate_disease(d, EvalContext(features={"QRS_ms": 130}))
    assert r.fired
    assert r.fired_variants() == ["v"]


def test_threshold_does_not_fire_when_below():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{"name": "v", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 120}}],
    })
    r = evaluate_disease(d, EvalContext(features={"QRS_ms": 100}))
    assert not r.fired


def test_missing_feature_degrades_gracefully():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{"name": "v", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">", "value": 120}}],
    })
    r = evaluate_disease(d, EvalContext(features={}))
    assert not r.fired
    trace = r.variant_results[0].trace
    assert "QRS_ms" in trace.missing_features


def test_sex_specific_threshold_male():
    d = _disease({
        "disease": "LVH", "manual_source": "p.148",
        "sex_specific": True,
        "variants": [{
            "name": "Cornell_voltage",
            "clause": {
                "kind": "threshold",
                "expr": "R_aVL_mV + S_V3_mV",
                "op": ">",
                "value": {"M": 2.8, "F": 2.0},
            },
        }],
    })
    # 2.9 mV > 2.8 male threshold -> fires
    r_male = evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 1.2, "S_V3_mV": 1.7}, sex="M")
    )
    assert r_male.fired
    # same values vs female threshold 2.0 -> fires
    r_fem = evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 1.2, "S_V3_mV": 1.7}, sex="F")
    )
    assert r_fem.fired
    # male low values -> not fires
    r_low = evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 0.8, "S_V3_mV": 0.5}, sex="M")
    )
    assert not r_low.fired


def test_sex_unknown_with_no_default_does_not_fire():
    d = _disease({
        "disease": "LVH", "manual_source": "p.148",
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
    r = evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 5, "S_V3_mV": 5})  # sex=None
    )
    assert not r.fired


def test_all_of_and_any_of():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{
            "name": "v",
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
    # QRS >=120 + R_V1 >1.5 -> fires
    assert evaluate_disease(
        d, EvalContext(features={"QRS_ms": 130, "R_V1_mV": 2.0, "R_V2_mV": 0.5})
    ).fired
    # QRS ok but both R_V1 / R_V2 small -> does not fire (any_of fails)
    assert not evaluate_disease(
        d, EvalContext(features={"QRS_ms": 130, "R_V1_mV": 0.5, "R_V2_mV": 0.5})
    ).fired
    # QRS below -> all_of fails
    assert not evaluate_disease(
        d, EvalContext(features={"QRS_ms": 100, "R_V1_mV": 2.0, "R_V2_mV": 0.5})
    ).fired


def test_not_clause():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{
            "name": "v",
            "clause": {
                "kind": "not_",
                "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">", "value": 120},
            },
        }],
    })
    assert evaluate_disease(d, EvalContext(features={"QRS_ms": 80})).fired
    assert not evaluate_disease(d, EvalContext(features={"QRS_ms": 140})).fired


def test_point_score_threshold():
    d = _disease({
        "disease": "LVH_RE", "manual_source": "p.149",
        "variants": [{
            "name": "Romhilt_Estes",
            "clause": {
                "kind": "point_score",
                "threshold_positive": 5,
                "items": [
                    {"points": 3, "when": {"kind": "threshold", "expr": "R_aVL_mV", "op": ">=", "value": 1.1}},
                    {"points": 3, "when": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 90}},
                    {"points": 1, "when": {"kind": "threshold", "expr": "QRS_axis_deg", "op": "<", "value": -30}},
                ],
            },
        }],
    })
    # 3 + 3 = 6 -> fires
    assert evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 1.5, "QRS_ms": 100, "QRS_axis_deg": 10})
    ).fired
    # 3 + 1 = 4 -> does not fire
    assert not evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 1.5, "QRS_ms": 80, "QRS_axis_deg": -45})
    ).fired


def test_exclusion_gate_blocks_all_variants():
    d = _disease({
        "disease": "LVH", "manual_source": "p.148",
        "exclusions": ["LBBB", "WPW"],
        "variants": [{
            "name": "Cornell",
            "clause": {
                "kind": "threshold", "expr": "R_aVL_mV + S_V3_mV", "op": ">", "value": 2.0,
            },
        }],
    })
    # rule would fire, but LBBB flag blocks it
    r = evaluate_disease(
        d, EvalContext(features={"R_aVL_mV": 2, "S_V3_mV": 2, "lbbb_flag": 1}, sex="F")
    )
    assert not r.fired
    assert "LBBB" in r.excluded_by
    # still individual variants record that they matched on voltage
    assert r.variant_results[0].fired is True


def test_format_disease_result_readable():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{
            "name": "v",
            "manual_page": 7,
            "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">=", "value": 120},
        }],
    })
    r = evaluate_disease(d, EvalContext(features={"QRS_ms": 130}))
    txt = format_disease_result(r, d)
    assert "FIRED" in txt
    assert "QRS_ms" in txt
    assert "p.7" in txt


def test_evaluate_variant_direct():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{"name": "v", "clause": {"kind": "threshold", "expr": "QRS_ms", "op": ">", "value": 100}}],
    })
    vr = evaluate_variant(d.variants[0], EvalContext(features={"QRS_ms": 120}))
    assert vr.fired
    assert vr.variant_name == "v"


def test_age_threshold():
    d = _disease({
        "disease": "X", "manual_source": "p.1",
        "variants": [{
            "name": "v",
            "clause": {
                "kind": "threshold",
                "expr": "QRS_ms",
                "op": ">=",
                "value": {"bands": [
                    {"max_age": 40, "value": 110},
                    {"max_age": 999, "value": 120},
                ]},
            },
        }],
    })
    # young patient, 115 >= 110 -> fires
    assert evaluate_disease(d, EvalContext(features={"QRS_ms": 115}, age_years=30)).fired
    # older patient, same 115 < 120 -> does not fire
    assert not evaluate_disease(d, EvalContext(features={"QRS_ms": 115}, age_years=60)).fired
