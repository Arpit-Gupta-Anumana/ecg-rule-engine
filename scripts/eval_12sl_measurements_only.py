#!/usr/bin/env python3
"""Evaluate rule engine using ONLY 12SL measurement columns (no statement-derived flags).

Inputs:  ptb_testing_data.csv → ~122 mapped measurement features per ECG
         NO lbbb_flag, rbbb_flag, wpw_flag, paced_flag, rhythm_*_flag, etc.
         (missing exclusion flags are treated as inactive — exclusions do not fire)

Ground truth: 12SL `statements` column (unchanged — for prevalence + metrics only)

Outputs:
  reports/12sl_input_feature_list.csv   — all measurement inputs + CSV source column
  reports/12sl_gt_prevalence.csv        — GT positive count/prevalence per disease
  reports/12sl_metrics_measurements_only.csv — confusion metrics (no flags run)
  reports/12sl_variant_fires_measurements_only.csv

Usage:
  python3 scripts/eval_12sl_measurements_only.py
  python3 scripts/eval_12sl_measurements_only.py --csv /path/to/ptb_testing_data.csv
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

# Reuse mapping + GT tables from the original eval
from eval_12sl_features import (  # noqa: E402
    DISEASE_GT,
    LEADS,
    NO_GT,
    map_row_to_features,
    parse_statements,
)

DEFAULT_CSV = Path("/Users/arpit.gupta/Downloads/ptb_testing_data.csv")
RULES_DIR = ROOT / "rules"
REPORTS = ROOT / "reports"

# Canonical measurement feature IDs (122) — keys produced by map_row_to_features
MEASUREMENT_FEATURE_IDS: list[str] = sorted(
    {
        "ventricular_rate_bpm",
        "atrial_rate_bpm",
        "RR_ms",
        "PP_ms",
        "QRS_ms",
        "PR_ms",
        "QT_ms",
        "QTc_Bazett_ms",
        "QTc_Fridericia_ms",
        "QTc_Framingham_ms",
        "P_ms",
        "QRS_axis_deg",
        "P_axis_deg",
        "T_axis_deg",
        *[f"{w}_{lead}_mV" for w in ("R", "Q", "S", "P", "T", "STJ", "STM") for lead in LEADS],
        *[f"Q_{lead}_ms" for lead in LEADS],
        *[f"QRS_{lead}_ms" for lead in LEADS],
    }
)

# CSV column that feeds each canonical ID (inverse of map_row_to_features)
CSV_SOURCE: dict[str, str] = {
    "QRS_ms": "QRS_Dur_Global",
    "PR_ms": "PR_Int_Global",
    "QT_ms": "QT_Int_Global",
    "QTc_Bazett_ms": "QT_IntBazett_Global",
    "QTc_Fridericia_ms": "QT_IntFridericia_Global",
    "QTc_Framingham_ms": "QT_IntFramingham_Global",
    "ventricular_rate_bpm": "HR_Ventr_Global",
    "atrial_rate_bpm": "HR_Atrial_Global",
    "RR_ms": "RR_Mean_Global",
    "P_ms": "P_Dur_Global",
    "PP_ms": "derived: 60000 / HR_Atrial_Global",
    "QRS_axis_deg": "R_AxisFrontal_Global",
    "P_axis_deg": "P_AxisFront_Global",
    "T_axis_deg": "T_AxisFront_Global",
}
for lead in LEADS:
    CSV_SOURCE[f"R_{lead}_mV"] = f"R_Amp_{lead}"
    CSV_SOURCE[f"Q_{lead}_mV"] = f"Q_Amp_{lead} (abs)"
    CSV_SOURCE[f"S_{lead}_mV"] = f"S_Amp_{lead} (abs)"
    CSV_SOURCE[f"Q_{lead}_ms"] = f"Q_Dur_{lead}"
    CSV_SOURCE[f"P_{lead}_mV"] = f"P_Amp_{lead}"
    CSV_SOURCE[f"T_{lead}_mV"] = f"T_Amp_{lead}"
    CSV_SOURCE[f"STJ_{lead}_mV"] = f"ST_Amp_{lead}"
    CSV_SOURCE[f"STM_{lead}_mV"] = f"ST_Amp116_{lead} (/1000)"
    CSV_SOURCE[f"QRS_{lead}_ms"] = f"R_Dur_{lead}+Q_Dur_{lead}+S_Dur_{lead}"


def load_diseases() -> dict:
    diseases = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        try:
            d = load_disease_yaml(yf)
            diseases[d.disease] = d
        except Exception as e:
            print(f"  WARN: skipping {yf.name}: {e}")
    return diseases


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def write_input_feature_list(out: Path) -> None:
    rows = []
    for i, fid in enumerate(MEASUREMENT_FEATURE_IDS, start=1):
        rows.append({
            "index": i,
            "feature_id": fid,
            "csv_source": CSV_SOURCE.get(fid, ""),
            "category": (
                "global_interval" if fid.endswith("_ms") and "_" not in fid.replace("_ms", "")
                else "global_rate" if "rate" in fid or fid in ("RR_ms", "PP_ms")
                else "global_axis" if fid.endswith("_deg")
                else "per_lead_amplitude" if fid.endswith("_mV")
                else "per_lead_duration"
            ),
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {len(rows)} input features → {out}")


def compute_gt_prevalence(df: pd.DataFrame, diseases: dict) -> pd.DataFrame:
    """Per-disease: how many ECGs have GT positive from 12SL statements."""
    n = len(df)
    stmt_sets = [parse_statements(str(row.get("statements", "[]"))) for _, row in df.iterrows()]

    rows = []
    for dn in sorted(diseases.keys()):
        gt_codes = DISEASE_GT.get(dn, set())
        has_gt = bool(gt_codes) and dn not in NO_GT
        if has_gt:
            pos = sum(1 for s in stmt_sets if gt_codes & s)
            neg = n - pos
            prev = pos / n if n else 0.0
        else:
            pos = neg = 0
            prev = float("nan")
        rows.append({
            "disease": dn,
            "has_12sl_gt_mapping": has_gt,
            "gt_12sl_codes": ",".join(sorted(gt_codes)) if gt_codes else "(no GT)",
            "n_ecgs": n,
            "gt_positive": pos if has_gt else "",
            "gt_negative": neg if has_gt else "",
            "prevalence_pct": round(100 * prev, 3) if has_gt else "",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="12SL eval: measurements only, no statement flags")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    diseases = load_diseases()
    print(f"Loaded {len(diseases)} rules")

    write_input_feature_list(args.out / "12sl_input_feature_list.csv")

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} records from {args.csv}")

    prev_df = compute_gt_prevalence(df, diseases)
    prev_df.to_csv(args.out / "12sl_gt_prevalence.csv", index=False)
    print(f"Wrote GT prevalence → {args.out / '12sl_gt_prevalence.csv'}")

    confusion = {dn: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for dn in diseases}
    fire_counts = {dn: 0 for dn in diseases}
    variant_fires: dict[tuple[str, str], int] = {}
    for d in diseases.values():
        for v in d.variants:
            variant_fires[(d.disease, v.name)] = 0

    skipped = 0
    evaluated = 0

    for _, row in tqdm(df.iterrows(), total=len(df), unit="ecg"):
        feats = map_row_to_features(row)
        if "QRS_ms" not in feats:
            skipped += 1
            continue
        evaluated += 1

        # Measurements only — explicitly NO statement-derived flags
        stmts = parse_statements(str(row.get("statements", "[]")))
        ctx = EvalContext(features=feats, sex=None, age_years=None)

        for dn, d in diseases.items():
            res = evaluate_disease(d, ctx)
            fired = res.fired
            if fired:
                fire_counts[dn] += 1
            for v in res.variant_results:
                if v.fired:
                    variant_fires[(dn, v.variant_name)] += 1

            gt_codes = DISEASE_GT.get(dn, set())
            if gt_codes and dn not in NO_GT:
                gt_pos = bool(gt_codes & stmts)
                if fired and gt_pos:
                    confusion[dn]["tp"] += 1
                elif fired and not gt_pos:
                    confusion[dn]["fp"] += 1
                elif not fired and gt_pos:
                    confusion[dn]["fn"] += 1
                else:
                    confusion[dn]["tn"] += 1

    n = evaluated
    print(f"\nEvaluated {n} records (skipped {skipped}) — measurements only, no flags")

    rows_out = []
    for dn in diseases:
        c = confusion[dn]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        gt_codes = DISEASE_GT.get(dn, set())
        has_gt = bool(gt_codes) and dn not in NO_GT
        positives = tp + fn
        negatives = tn + fp
        total = positives + negatives

        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        npv = safe_div(tn, tn + fn)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        acc = safe_div(tp + tn, total) if total else 0.0

        if has_gt and positives > 0:
            roc_auc = 0.5 * (1 + sens - (1 - spec))
            prev = positives / total if total else 0.0
            pr_auc = 0.5 * sens * (1 + ppv) + 0.5 * (1 - sens) * (ppv + prev)
        else:
            roc_auc = float("nan")
            pr_auc = float("nan")

        rows_out.append({
            "disease": dn,
            "gt_codes": ",".join(sorted(gt_codes)) if has_gt else "(no GT)",
            "positives_GT": positives,
            "negatives_GT": negatives,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": sens,
            "specificity": spec,
            "PPV": ppv,
            "NPV": npv,
            "F1": f1,
            "accuracy": acc,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc,
            "prevalence": safe_div(positives, n) if has_gt else 0,
            "fire_rate": safe_div(fire_counts[dn], n),
            "fires": fire_counts[dn],
            "eval_mode": "measurements_only_no_flags",
        })

    res_df = pd.DataFrame(rows_out)
    res_df["has_gt"] = res_df["gt_codes"] != "(no GT)"
    res_df = res_df.sort_values(["has_gt", "F1", "fire_rate"], ascending=[False, False, False])
    res_df = res_df.drop(columns=["has_gt"])
    res_df.to_csv(args.out / "12sl_metrics_measurements_only.csv", index=False)

    var_rows = [
        {"disease": dn, "variant": vn, "fires": fires, "fire_rate": safe_div(fires, n)}
        for (dn, vn), fires in variant_fires.items()
    ]
    pd.DataFrame(var_rows).sort_values(["disease", "fires"], ascending=[True, False]).to_csv(
        args.out / "12sl_variant_fires_measurements_only.csv", index=False
    )

    print(f"Wrote metrics → {args.out / '12sl_metrics_measurements_only.csv'}")

    gt_df = res_df[res_df["gt_codes"] != "(no GT)"].head(15)
    print("\n=== Top 15 by F1 (measurements only, no flags) ===")
    for _, r in gt_df.iterrows():
        print(
            f"  {r['disease']:28s}  F1={r['F1']*100:5.1f}%  "
            f"Sens={r['sensitivity']*100:5.1f}%  Spec={r['specificity']*100:5.1f}%  "
            f"GT prev={r['prevalence']*100:5.2f}%  fires={int(r['fires'])}"
        )


if __name__ == "__main__":
    main()
