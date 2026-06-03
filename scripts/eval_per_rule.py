#!/usr/bin/env python3
"""Per-rule evaluation against the nfer parquet GT.

For each (abnormality in parquet, rule mapped to it) pair:

  1. Parse the rule's clauses, collect the EXACT features it touches.
  2. Filter the merged (3-CSV) cohort to ECGs where those features are present.
  3. Evaluate just that rule on the per-rule cohort.
  4. Score against the matching GT column from the abnormalities parquet.

Result: one CSV row per (abnormality, rule) with N_eval, GT+, TP/FP/FN, F1, plus
the required-feature list and how many ECGs were dropped for missing features.

Usage
-----
python3 scripts/eval_per_rule.py \
    --lead-csv  /data/sharedhdd/arpit/arpit/ecg_lead_measurement.csv \
    --global-csv /data/sharedhdd/arpit/arpit/ecg_measurement.csv \
    --interval-csv /data/sharedhdd/arpit/arpit/ecg_waveform_measurement.csv \
    --keys PERSON_ID,EVENT_DTM \
    --gt-csv /data/NFERECG/shared/supreeth.gupta/abnormality_extraction/abnormalities_cohort.parquet \
    --gt-keys NFER_PID,NFER_DTM \
    --amp-scale 0.001 \
    --out reports/nfer_per_rule_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ecg_rule_engine.dsl.expr import parse_expr
from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.dsl.schema import (
    AllOfClause, AnyOfClause, NotClause, PointScoreClause, ThresholdClause,
)
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

from run_nfer_csvs import (
    ABNORMALITY_TO_PRED,
    DISEASE_TO_FLAG,
    VARIANT_TO_FLAG,
    _norm,
    _truthy,
    load_gt_table,
    merge_csvs,
    row_to_features,
    safe_div,
)

# Features the engine derives or auto-fills (NOT required from the CSV).
DERIVED_OR_OPTIONAL: set[str] = {
    # 2nd-pass derived flags (from rule output, not measurements):
    *DISEASE_TO_FLAG.values(),
    *VARIANT_TO_FLAG.values(),
    # demographics / auto-derived in evaluator:
    "sex_M_flag", "sex_F_flag",
    # rate ↔ interval fallbacks
    "PP_ms",  # derived from atrial_rate_bpm if missing
}

RULES_DIR = ROOT / "rules"


# ── clause walking ─────────────────────────────────────────────────────────

def features_in_clause(clause) -> set[str]:
    """Walk a clause tree, return every CSV-derivable feature it touches."""
    out: set[str] = set()
    if isinstance(clause, ThresholdClause):
        out |= parse_expr(clause.expr).features()
    elif isinstance(clause, AllOfClause) or isinstance(clause, AnyOfClause):
        for c in clause.clauses:
            out |= features_in_clause(c)
    elif isinstance(clause, NotClause):
        out |= features_in_clause(clause.clause)
    elif isinstance(clause, PointScoreClause):
        for item in clause.items:
            out |= features_in_clause(item.when)
    return out


def required_features_for_spec(spec: str, diseases: dict) -> set[str]:
    """`Disease` or `Disease.Variant` → set of CSV features it needs."""
    if "." in spec:
        disease_name, variant_name = spec.split(".", 1)
    else:
        disease_name, variant_name = spec, None

    disease = diseases.get(disease_name)
    if disease is None:
        return set()

    feats: set[str] = set()
    for v in disease.variants:
        if variant_name is None or v.name == variant_name:
            feats |= features_in_clause(v.clause)
    # strip derived / auto features
    return {f for f in feats if f not in DERIVED_OR_OPTIONAL}


# ── per-rule eval ──────────────────────────────────────────────────────────

def evaluate_one_rule(
    spec: str,
    diseases: dict,
    merged: pd.DataFrame,
    amp_scale: float,
    keys: list[str],
) -> tuple[pd.DataFrame, set[str], int]:
    """Returns (per-ECG predictions for this spec, required features, n_skipped).

    Only ECGs where ALL required features are present (after row_to_features)
    are evaluated; others are excluded from this rule's cohort.
    """
    required = required_features_for_spec(spec, diseases)
    if not required:
        return pd.DataFrame(), required, 0

    if "." in spec:
        disease_name, variant_name = spec.split(".", 1)
    else:
        disease_name, variant_name = spec, None

    disease = diseases.get(disease_name)
    rows: list[dict] = []
    skipped = 0
    for _, row in merged.iterrows():
        feats = row_to_features(row, amp_scale)
        if not required.issubset(feats.keys()):
            skipped += 1
            continue
        age = None
        av = feats.get("age_years")
        if av is None and "PERSON_AGE" in row.index and pd.notna(row.get("PERSON_AGE")):
            try:
                age = int(float(row.get("PERSON_AGE")))
            except (TypeError, ValueError):
                age = None
        ctx = EvalContext(features=dict(feats), sex=None, age_years=age)
        res = evaluate_disease(disease, ctx)
        fired = 0
        if variant_name is None:
            fired = int(res.fired)
        else:
            for vr in res.variant_results:
                if vr.variant_name == variant_name:
                    fired = int(vr.fired)
                    break
        rows.append({**{k: row.get(k) for k in keys}, "pred": fired})

    return pd.DataFrame(rows), required, skipped


def score_pair(
    abnormality: str,
    spec: str,
    preds: pd.DataFrame,
    gt: pd.DataFrame,
    pred_keys: list[str],
    col_lookup: dict[str, str],
    required: set[str],
    skipped: int,
) -> dict:
    gtc = col_lookup.get(_norm(abnormality))
    if gtc is None:
        return {
            "abnormality": abnormality, "rule": spec,
            "status": "NO GT COLUMN",
            "required_features": ", ".join(sorted(required)),
            "n_skipped_missing_features": skipped,
        }
    if preds.empty:
        return {
            "abnormality": abnormality, "rule": spec,
            "status": "NO ECGs WITH ALL FEATURES",
            "required_features": ", ".join(sorted(required)),
            "n_skipped_missing_features": skipped,
        }

    merged = preds.merge(gt[pred_keys + [gtc]], on=pred_keys, how="inner")
    if merged.empty:
        return {
            "abnormality": abnormality, "rule": spec,
            "status": "NO GT JOIN",
            "required_features": ", ".join(sorted(required)),
            "n_skipped_missing_features": skipped,
        }

    gt_pos = merged[gtc].apply(_truthy)
    pred_pos = merged["pred"] >= 1
    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    tn = int((~pred_pos & ~gt_pos).sum())
    return {
        "abnormality": abnormality, "rule": spec,
        "required_features": ", ".join(sorted(required)),
        "n_required": len(required),
        "n_eval": len(merged),
        "n_skipped_missing_features": skipped,
        "GT_pos": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sensitivity": round(safe_div(tp, tp + fn), 4),
        "specificity": round(safe_div(tn, tn + fp), 4),
        "PPV": round(safe_div(tp, tp + fp), 4),
        "F1": round(safe_div(2 * tp, 2 * tp + fp + fn), 4),
        "prevalence": round(safe_div(tp + fn, len(merged)), 4),
        "status": "ok",
    }


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-csv", type=Path, required=True)
    ap.add_argument("--global-csv", type=Path, required=True)
    ap.add_argument("--interval-csv", type=Path, default=None)
    ap.add_argument("--keys", type=str, default="PERSON_ID,EVENT_DTM")
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM")
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "nfer_per_rule_metrics.csv")
    args = ap.parse_args()

    pred_keys = [k.strip() for k in args.keys.split(",")]
    gt_keys = [k.strip() for k in args.gt_keys.split(",")]

    print("Loading rules...")
    diseases: dict = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        try:
            d = load_disease_yaml(yf)
            diseases[d.disease] = d
        except Exception as e:
            print(f"  WARN: {yf.name}: {e}")

    print("Loading CSVs...")
    lead_df = pd.read_csv(args.lead_csv)
    glob_df = pd.read_csv(args.global_csv)
    ivl_df = pd.read_csv(args.interval_csv) if args.interval_csv else None

    merged = merge_csvs(lead_df, glob_df, ivl_df, pred_keys, args.amp_scale,
                        complete_only=False)
    print(f"Merged cohort: {len(merged)} ECGs")

    print("Loading GT parquet...")
    gt = load_gt_table(args.gt_csv)
    col_lookup = {_norm(c): c for c in gt.columns}
    rename = {gk: pk for gk, pk in zip(gt_keys, pred_keys)}
    gt = gt.rename(columns=rename)
    for k in pred_keys:
        gt[k] = gt[k].astype(str)
        merged[k] = merged[k].astype(str)

    # Iterate (abnormality, rule) pairs
    results: list[dict] = []
    pairs = [(ab, spec) for ab, specs in ABNORMALITY_TO_PRED.items() for spec in specs]
    print(f"Evaluating {len(pairs)} (abnormality, rule) pairs...")
    for abnormality, spec in tqdm(pairs, unit="rule"):
        preds, required, skipped = evaluate_one_rule(
            spec, diseases, merged, args.amp_scale, pred_keys
        )
        results.append(score_pair(
            abnormality, spec, preds, gt, pred_keys, col_lookup, required, skipped
        ))

    df = pd.DataFrame(results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")

    ok = df[df["status"] == "ok"].sort_values("F1", ascending=False, na_position="last")
    if not ok.empty:
        print("\n=== Per-rule metrics (sorted by F1) ===")
        cols = ["abnormality", "rule", "n_eval", "n_skipped_missing_features",
                "GT_pos", "TP", "FP", "FN", "sensitivity", "PPV", "F1",
                "n_required"]
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 60)
        print(ok[cols].to_string(index=False))

    skipped = df[df["status"] != "ok"]
    if not skipped.empty:
        print("\n=== Skipped pairs ===")
        print(skipped[["abnormality", "rule", "status",
                       "required_features"]].to_string(index=False))


if __name__ == "__main__":
    main()
