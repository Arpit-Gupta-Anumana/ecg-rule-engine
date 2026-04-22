"""Tests for the mini-expression language used by ThresholdClause.expr."""

from __future__ import annotations

import pytest

from ecg_rule_engine.dsl.expr import (
    ALLOWED_FUNCS,
    ExprError,
    parse_expr,
    validate_features,
)


def test_numeric_literal():
    n = parse_expr("3.5")
    assert n.eval({}) == 3.5


def test_variable_lookup():
    n = parse_expr("R_aVL_mV")
    assert n.eval({"R_aVL_mV": 2.1}) == 2.1


def test_missing_variable_raises():
    n = parse_expr("R_aVL_mV")
    with pytest.raises(ExprError):
        n.eval({})


def test_binop_arithmetic():
    assert parse_expr("R_aVL_mV + S_V3_mV").eval({"R_aVL_mV": 1.5, "S_V3_mV": 2.0}) == 3.5
    assert parse_expr("(QRS_ms - 80) / 10").eval({"QRS_ms": 120}) == 4.0


def test_division_by_zero():
    with pytest.raises(ExprError):
        parse_expr("1 / QRS_ms").eval({"QRS_ms": 0})


def test_max_min_abs_sum_avg():
    env = {"R_V5_mV": 2.0, "R_V6_mV": 3.0, "S_V1_mV": 2.5}
    assert parse_expr("max(R_V5_mV, R_V6_mV)").eval(env) == 3.0
    assert parse_expr("min(R_V5_mV, R_V6_mV)").eval(env) == 2.0
    assert parse_expr("abs(-5)").eval({}) == 5
    assert parse_expr("sum(R_V5_mV, R_V6_mV, S_V1_mV)").eval(env) == 7.5
    assert parse_expr("avg(R_V5_mV, R_V6_mV)").eval(env) == 2.5


def test_sokolow_lyon_expression():
    env = {"S_V1_mV": 2.0, "S_V2_mV": 2.2, "R_V5_mV": 1.8, "R_V6_mV": 2.1}
    expr = "max(S_V1_mV, S_V2_mV) + max(R_V5_mV, R_V6_mV)"
    assert parse_expr(expr).eval(env) == pytest.approx(4.3)


def test_unknown_function_rejected():
    with pytest.raises(ExprError):
        parse_expr("foo(1, 2)")


def test_unknown_feature_validation():
    node = parse_expr("R_aVL_mV + fake_feature_xyz")
    unknown = validate_features(node)
    assert unknown == ["fake_feature_xyz"]


def test_empty_expression():
    with pytest.raises(ExprError):
        parse_expr("")


def test_allowed_funcs_set_is_stable():
    assert ALLOWED_FUNCS == {"max", "min", "abs", "sum", "avg"}


def test_pretty_substitutes_values():
    node = parse_expr("R_aVL_mV + S_V3_mV")
    p = node.pretty({"R_aVL_mV": 2.9, "S_V3_mV": 1.8})
    assert "2.9" in p and "1.8" in p and "+" in p
