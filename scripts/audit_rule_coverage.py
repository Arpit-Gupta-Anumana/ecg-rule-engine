#!/usr/bin/env python3
"""Audit which abnormalities are silently skipped vs really evaluated.

For each parquet abnormality and every ECG, figure out:

  * does row_to_features() produce *all* features required by the rules
    mapped to that abnormality?
  * if not, which features are missing?
  * how does this break down across GT-positive vs GT-negative ECGs?

This surfaces "AtrialFlutter-class" bugs: rules that look like they
returned 0 but actually never ran because some required feature wasn't in
the CSV (or wasn't being mapped into the engine vocabulary).

Outputs
-------
--out-summary    one row per abnormality, with GT counts split by
                 evaluable vs skipped, plus top missing features
--out-missing    long-format (abnormality, feature) → how many GT+ ECGs
                 were dropped because of that single feature

Example
-------
python3 scripts/audit_rule_coverage.py \\
    --lead-csv     /data/sharedhdd/arpit/arpit/ecg_lead_measurement.csv \\
    --global-csv   /data/sharedhdd/arpit/arpit/ecg_measurement.csv \\
    --interval-csv /data/sharedhdd/arpit/arpit/ecg_waveform_measurement.csv \\
    --keys         PERSON_ID,EVENT_DTM \\
    --gt-csv       /data/NFERECG/shared/supreeth.gupta/abnormality_extraction/abnormalities_cohort.parquet \\
    --gt-keys      NFER_PID,NFER_DTM \\
    --amp-scale    0.001 \\
    --out-summary  reports/rule_coverage_summary.csv \\
    --out-missing  reports/rule_coverage_missing_features.csv
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from ecg_rule_engine.dsl.loader import load_disease_yaml

from run_nfer_csvs import (
    ABNORMALITY_TO_PRED,
    _norm,
    _truthy,
    load_gt_table,
    merge_csvs,
    row_to_features,
)
from eval_per_rule import required_features_for_spec

RULES_DIR = ROOT / "rules"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-csv", type=Path, required=True)
    ap.add_argument("--global-csv", type=Path, required=True)
    ap.add_argument("--interval-csv", type=Path, default=None)
    ap.add_argument("--keys", type=str, default="PERSON_ID,EVENT_DTM")
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM")
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--out-summary", type=Path,
                    default=ROOT / "reports" / "rule_coverage_summary.csv")
    ap.add_argument("--out-missing", type=Path,
                    default=ROOT / "reports" / "rule_coverage_missing_features.csv")
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

    # required features per abnormality = union over its mapped specs
    abn_required: dict[str, set[str]] = {}
    for abn, specs in ABNORMALITY_TO_PRED.items():
        s: set[str] = set()
        for spec in specs:
            s |= required_features_for_spec(spec, diseases)
        abn_required[abn] = s

    print("Loading CSVs...")
    lead_df = pd.read_csv(args.lead_csv)
    glob_df = pd.read_csv(args.global_csv)
    ivl_df = pd.read_csv(args.interval_csv) if args.interval_csv else None
    merged = merge_csvs(lead_df, glob_df, ivl_df, pred_keys, args.amp_scale,
                        complete_only=False)
    for k in pred_keys:
        merged[k] = merged[k].astype(str)
    print(f"Merged cohort: {len(merged)} ECGs")

    print("Loading GT...")
    gt = load_gt_table(args.gt_csv)
    gt = gt.rename(columns={gk: pk for gk, pk in zip(gt_keys, pred_keys)})
    for k in pred_keys:
        gt[k] = gt[k].astype(str)
    col_lookup = {_norm(c): c for c in gt.columns}

    # Pre-compute per-ECG feature presence ONCE.
    print("Computing per-ECG feature availability...")
    feats_present: list[dict] = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), unit="ecg"):
        f = row_to_features(row, args.amp_scale)
        rec = {k: row[k] for k in pred_keys}
        rec["__features__"] = set(f.keys())
        feats_present.append(rec)
    fp = pd.DataFrame(feats_present)

    # Join with GT
    joined = fp.merge(gt, on=pred_keys, how="inner")
    print(f"Joined cohort with GT: {len(joined)} ECGs")

    # ── per-abnormality audit ─────────────────────────────────────────
    summary_rows: list[dict] = []
    missing_rows: list[dict] = []
    for abn, req in abn_required.items():
        gt_col = col_lookup.get(_norm(abn))
        if gt_col is None:
            summary_rows.append({"abnormality": abn,
                                 "status": "NO GT COLUMN",
                                 "required_features": ", ".join(sorted(req))})
            continue
        if not req:
            summary_rows.append({"abnormality": abn,
                                 "status": "NO RULE MAPPING / NO REQUIRED FEATURES",
                                 "required_features": ""})
            continue

        gt_pos_mask = joined[gt_col].apply(_truthy)
        # missing-features mask
        def missing_for(feat_set: set[str], required: set[str] = req) -> set[str]:
            return required - feat_set

        joined["__missing__"] = joined["__features__"].apply(missing_for)
        joined["__evaluable__"] = joined["__missing__"].apply(lambda s: len(s) == 0)

        n = len(joined)
        n_eval = int(joined["__evaluable__"].sum())
        gt_pos = int(gt_pos_mask.sum())
        gt_pos_eval = int((gt_pos_mask & joined["__evaluable__"]).sum())
        gt_pos_skipped = gt_pos - gt_pos_eval

        # top missing features among GT-positive skipped ECGs
        miss_counter: Counter[str] = Counter()
        for s in joined.loc[gt_pos_mask & ~joined["__evaluable__"], "__missing__"]:
            miss_counter.update(s)

        for feat, cnt in miss_counter.most_common():
            missing_rows.append({
                "abnormality": abn,
                "feature": feat,
                "n_gt_pos_dropped": cnt,
                "frac_gt_pos_dropped": round(cnt / gt_pos, 4) if gt_pos else 0,
            })

        top_missing = "; ".join(f"{f}({c})" for f, c in miss_counter.most_common(5))

        summary_rows.append({
            "abnormality": abn,
            "required_features": ", ".join(sorted(req)),
            "n_total": n,
            "n_evaluable": n_eval,
            "n_skipped_missing_features": n - n_eval,
            "GT_pos": gt_pos,
            "GT_pos_evaluable": gt_pos_eval,
            "GT_pos_skipped": gt_pos_skipped,
            "frac_GT_pos_skipped": round(gt_pos_skipped / gt_pos, 4) if gt_pos else 0,
            "top_missing_features_in_gt_pos_skipped": top_missing,
            "status": "ok",
        })

    summary = pd.DataFrame(summary_rows)
    missing = pd.DataFrame(missing_rows)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_missing.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    missing.to_csv(args.out_missing, index=False)
    print(f"\nWrote summary -> {args.out_summary}")
    print(f"Wrote feature-level breakdown -> {args.out_missing}")

    # ── console: show the worst offenders ────────────────────────────
    if not summary.empty:
        ok = summary[summary["status"] == "ok"].copy()
        if not ok.empty:
            ok = ok.sort_values("GT_pos_skipped", ascending=False)
            cols = ["abnormality", "GT_pos", "GT_pos_evaluable",
                    "GT_pos_skipped", "frac_GT_pos_skipped",
                    "top_missing_features_in_gt_pos_skipped"]
            pd.set_option("display.width", 220)
            pd.set_option("display.max_colwidth", 80)
            print("\n=== Worst offenders: GT-positive ECGs the engine never sees ===")
            print(ok[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
