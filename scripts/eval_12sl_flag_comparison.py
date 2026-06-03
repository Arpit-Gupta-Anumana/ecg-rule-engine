#!/usr/bin/env python3
"""Compare rule-engine eval: measurements only vs derived flags vs statement flags.

Derived flags = outputs from pass-1 rules on the SAME ECG (e.g. SinusRhythm fired
→ rhythm_sinus_flag=1). NOT read from 12SL statements input.

Statement flags = read from 12SL `statements` column (oracle / label leakage for
rhythm/conduction — what the old eval_12sl_features.py did).

Outputs:
  reports/12sl_metrics_comparison.csv      — F1/Sens/Spec per disease × mode
  reports/12sl_derived_flag_coverage.csv   — how often each derived flag fires
  reports/12sl_flag_summary.md             — short human summary

Usage:
  python3 scripts/eval_12sl_flag_comparison.py
  python3 scripts/eval_12sl_flag_comparison.py --limit 2000
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

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import DiseaseResult, EvalContext, evaluate_disease

from eval_12sl_features import (
    DISEASE_GT,
    NO_GT,
    map_row_to_features,
    parse_statements,
)

DEFAULT_CSV = Path("/Users/arpit.gupta/Downloads/ptb_testing_data.csv")
RULES_DIR = ROOT / "rules"
REPORTS = ROOT / "reports"

# 13 registry flags (excl sex_M/F, age_years)
ALL_FLAGS = [
    "rhythm_afib_flag",
    "rhythm_aflutter_flag",
    "rhythm_sinus_flag",
    "paced_flag",
    "wpw_flag",
    "lbbb_flag",
    "rbbb_flag",
    "ilbbb_flag",
    "irbbb_flag",
    "lafb_flag",
    "lpfb_flag",
    "ivcb_flag",
    "rvh_flag",
]

# Disease-level fire → flag
DISEASE_TO_FLAG: dict[str, str] = {
    "SinusRhythm": "rhythm_sinus_flag",
    "AFib": "rhythm_afib_flag",
    "AtrialFlutter": "rhythm_aflutter_flag",
    "Pacing": "paced_flag",
    "WPW": "wpw_flag",
    "LBBB": "lbbb_flag",
    "RBBB": "rbbb_flag",
    "RVH": "rvh_flag",
    "NonspecificIVCB": "ivcb_flag",
}

# Variant-level fire → flag
VARIANT_TO_FLAG: dict[tuple[str, str], str] = {
    ("IncompleteBundleBlocks", "ILBBB"): "ilbbb_flag",
    ("IncompleteBundleBlocks", "IRBBB"): "irbbb_flag",
    ("Hemiblocks", "LAFB"): "lafb_flag",
    ("Hemiblocks", "LPFB"): "lpfb_flag",
    ("Hemiblocks", "IVCD_nonspecific"): "ivcb_flag",
}


def load_diseases() -> dict:
    out = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        try:
            d = load_disease_yaml(yf)
            out[d.disease] = d
        except Exception as e:
            print(f"  WARN skip {yf.name}: {e}")
    return out


def statement_flags(stmts: set[str]) -> dict[str, float]:
    """Flags from 12SL statements (input labels — not derived from rules)."""
    return {
        "lbbb_flag": float("LBBB" in stmts),
        "rbbb_flag": float("RBBB" in stmts),
        "ilbbb_flag": float("ILBBB" in stmts),
        "irbbb_flag": float("IRBBB" in stmts),
        "lafb_flag": float("AFB" in stmts),
        "lpfb_flag": float("PFB" in stmts),
        "rvh_flag": float("RVH" in stmts or "RVE+" in stmts),
        "ivcb_flag": float(
            "IVCB" in stmts or "IVCD" in stmts or "ILBBB" in stmts or "IRBBB" in stmts
        ),
        "wpw_flag": float("WPW" in stmts or "WPWA" in stmts or "WPWB" in stmts),
        "paced_flag": float(
            "APR" in stmts or "VPR" in stmts or "ASVPR" in stmts or "AVDPR" in stmts
        ),
        "rhythm_afib_flag": float("AFIB" in stmts),
        "rhythm_aflutter_flag": float("FLUT" in stmts),
        "rhythm_sinus_flag": float(bool({"NSR", "SRTH", "SBRAD", "STACH", "MSBRAD"} & stmts)),
    }


def derived_flags_from_results(results: dict[str, DiseaseResult]) -> dict[str, float]:
    """Flags from pass-1 rule outputs on same ECG (measurements only)."""
    flags: dict[str, float] = {}
    for disease, flag_name in DISEASE_TO_FLAG.items():
        res = results.get(disease)
        if res and res.fired:
            flags[flag_name] = 1.0

    for (disease, variant), flag_name in VARIANT_TO_FLAG.items():
        res = results.get(disease)
        if not res:
            continue
        for vr in res.variant_results:
            if vr.variant_name == variant and vr.fired:
                flags[flag_name] = 1.0
                break

    return flags


def evaluate_all(diseases: dict, ctx: EvalContext) -> dict[str, DiseaseResult]:
    return {dn: evaluate_disease(d, ctx) for dn, d in diseases.items()}


def update_confusion(
    confusion: dict,
    mode: str,
    diseases: dict,
    results: dict[str, DiseaseResult],
    stmts: set[str],
):
    for dn in diseases:
        fired = results[dn].fired
        gt_codes = DISEASE_GT.get(dn, set())
        if not gt_codes or dn in NO_GT:
            continue
        gt_pos = bool(gt_codes & stmts)
        c = confusion[mode][dn]
        if fired and gt_pos:
            c["tp"] += 1
        elif fired and not gt_pos:
            c["fp"] += 1
        elif not fired and gt_pos:
            c["fn"] += 1
        else:
            c["tn"] += 1


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metrics_from_confusion(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    ppv = safe_div(tp, tp + fp)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    return {"sensitivity": sens, "specificity": spec, "PPV": ppv, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)

    diseases = load_diseases()
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    modes = ["measurements_only", "derived_flags", "statement_flags"]
    confusion = {m: {dn: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for dn in diseases} for m in modes}
    flag_counts = {f: 0 for f in ALL_FLAGS}
    n_eval = 0

    for _, row in tqdm(df.iterrows(), total=len(df), unit="ecg"):
        feats = map_row_to_features(row)
        if "QRS_ms" not in feats:
            continue
        n_eval += 1
        stmts = parse_statements(str(row.get("statements", "[]")))

        # --- Pass 1: measurements only ---
        ctx_meas = EvalContext(features=dict(feats), sex=None, age_years=None)
        res_meas = evaluate_all(diseases, ctx_meas)
        update_confusion(confusion, "measurements_only", diseases, res_meas, stmts)

        # --- Derived flags from pass-1 outputs ---
        derived = derived_flags_from_results(res_meas)
        for f in ALL_FLAGS:
            if derived.get(f, 0) >= 1:
                flag_counts[f] += 1
        feats_derived = {**feats, **derived}
        ctx_derived = EvalContext(features=feats_derived, sex=None, age_years=None)
        res_derived = evaluate_all(diseases, ctx_derived)
        update_confusion(confusion, "derived_flags", diseases, res_derived, stmts)

        # --- Statement flags (from 12SL input) ---
        feats_stmt = {**feats, **statement_flags(stmts)}
        ctx_stmt = EvalContext(features=feats_stmt, sex=None, age_years=None)
        res_stmt = evaluate_all(diseases, ctx_stmt)
        update_confusion(confusion, "statement_flags", diseases, res_stmt, stmts)

    print(f"\nEvaluated {n_eval} ECGs")

    # Derived flag coverage
    cov_rows = []
    for f in ALL_FLAGS:
        src_parts = [dn for dn, fn in DISEASE_TO_FLAG.items() if fn == f]
        for (dn, vn), fn in VARIANT_TO_FLAG.items():
            if fn == f:
                src_parts.append(f"{dn}.{vn}")
        cov_rows.append({
            "flag": f,
            "derived_from_rule_output": ", ".join(src_parts) if src_parts else "?",
            "ecgs_flag_on": flag_counts[f],
            "prevalence_pct": round(100 * flag_counts[f] / n_eval, 3) if n_eval else 0,
            "statement_flag_prevalence_note": "see 12sl_gt / statements (different source)",
        })
    pd.DataFrame(cov_rows).to_csv(REPORTS / "12sl_derived_flag_coverage.csv", index=False)

    # Comparison table
    rows = []
    for dn in diseases:
        gt_codes = DISEASE_GT.get(dn, set())
        has_gt = bool(gt_codes) and dn not in NO_GT
        row = {"disease": dn, "has_gt": has_gt}
        for mode in modes:
            c = confusion[mode][dn]
            m = metrics_from_confusion(c["tp"], c["fp"], c["fn"], c["tn"])
            row[f"F1_{mode}"] = m["F1"] if has_gt else float("nan")
            row[f"Sens_{mode}"] = m["sensitivity"] if has_gt else float("nan")
            row[f"Spec_{mode}"] = m["specificity"] if has_gt else float("nan")
            row[f"TP_{mode}"] = m["TP"] if has_gt else ""
            row[f"FP_{mode}"] = m["FP"] if has_gt else ""
            row[f"FN_{mode}"] = m["FN"] if has_gt else ""
        if has_gt:
            row["F1_delta_derived_vs_meas"] = row["F1_derived_flags"] - row["F1_measurements_only"]
            row["F1_delta_stmt_vs_meas"] = row["F1_statement_flags"] - row["F1_measurements_only"]
        rows.append(row)

    cmp_df = pd.DataFrame(rows)
    cmp_df = cmp_df.sort_values("has_gt", ascending=False)
    cmp_df.to_csv(REPORTS / "12sl_metrics_comparison.csv", index=False)

    # Summary markdown
    derivable = sum(1 for r in cov_rows if r["ecgs_flag_on"] > 0)
    lines = [
        "# 12SL eval: measurements vs derived flags vs statement flags",
        "",
        f"ECGs evaluated: **{n_eval}**",
        "",
        "## Flag sources",
        "",
        "| Mode | Flag source |",
        "|------|-------------|",
        "| `measurements_only` | 122 measurement features only |",
        "| `derived_flags` | Pass-1 rule fires → flags (same ECG, no statements) |",
        "| `statement_flags` | 12SL `statements` column (input labels) |",
        "",
        f"## Derived flags that ever fire ({derivable} / {len(ALL_FLAGS)})",
        "",
        "| Flag | Rule output source | % ECGs ON |",
        "|------|-------------------|-----------|",
    ]
    for r in sorted(cov_rows, key=lambda x: -x["ecgs_flag_on"]):
        if r["ecgs_flag_on"] > 0:
            lines.append(
                f"| `{r['flag']}` | {r['derived_from_rule_output']} | {r['prevalence_pct']}% |"
            )
    never = [r["flag"] for r in cov_rows if r["ecgs_flag_on"] == 0]
    if never:
        lines.extend(["", "**Never derived from measurements-only rules:**", ""])
        for f in never:
            lines.append(f"- `{f}`")
    lines.extend(["", "## Top F1 improvements (derived vs measurements)", ""])
    has_gt_df = cmp_df[cmp_df["has_gt"]]
    if len(has_gt_df):
        top = has_gt_df.nlargest(10, "F1_delta_derived_vs_meas")[
            ["disease", "F1_measurements_only", "F1_derived_flags", "F1_statement_flags", "F1_delta_derived_vs_meas"]
        ]
        lines.append("```")
        lines.append(top.to_string(index=False))
        lines.append("```")
    (REPORTS / "12sl_flag_summary.md").write_text("\n".join(lines))

    print(f"Wrote {REPORTS / '12sl_metrics_comparison.csv'}")
    print(f"Wrote {REPORTS / '12sl_derived_flag_coverage.csv'}")
    print(f"Wrote {REPORTS / '12sl_flag_summary.md'}")

    # Console: flags coverage
    print("\n=== Derived flags (from rule outputs, not statements) ===")
    for r in cov_rows:
        if r["ecgs_flag_on"] > 0:
            print(f"  {r['flag']:22s}  {r['prevalence_pct']:6.2f}%  ← {r['derived_from_rule_output']}")
    print(f"\n  Active: {derivable}/{len(ALL_FLAGS)} flags")


if __name__ == "__main__":
    main()
