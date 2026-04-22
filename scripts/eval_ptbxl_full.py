"""Full PTB-XL benchmark: all rules × all 21k records.

Outputs per disease (where SCP ground-truth exists):
  TP, FP, FN, TN, Sensitivity, Specificity, PPV, NPV,
  F1, ROC-AUC*, PR-AUC*, Prevalence, Fire rate, Positive count, Negative count

*AUC note: our rule engine outputs binary (fire/not), so ROC-AUC and PR-AUC
are computed from two discrete operating points only (always 0 or 1). They are
included for completeness but will equal the trapezoidal approximation of the
(0, FPR)→(1, TPR) step; for a proper smooth AUC a probabilistic scorer would
be needed.

Usage:
  python scripts/eval_ptbxl_full.py          # all records, all folds
  python scripts/eval_ptbxl_full.py --fold 10  # test fold only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path("/Users/arpit.gupta/Desktop/ecg_rule_engine")

import sys
sys.path.insert(0, str(ROOT / "src"))

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

PARQUET = ROOT / "data" / "ptbxl_features.parquet"
REPORTS = ROOT / "reports"
RULES_DIR = ROOT / "rules"

# ── SCP ground-truth mapping ────────────────────────────────────────────
# Maps disease YAML key → set of SCP codes that count as positive GT.
# An empty set means "no GT available" — we report fire rate only.
DISEASE_SCP_GT: dict[str, set[str]] = {
    "LBBB":                   {"CLBBB"},
    "RBBB":                   {"CRBBB"},
    "IncompleteBundleBlocks": {"IRBBB", "ILBBB"},
    "LVH":                    {"LVH", "VCLVH"},
    "RVH":                    {"RVH"},
    "AFib":                   {"AFIB"},
    "AtrialFlutter":          {"AFLT"},
    "WPW":                    {"WPW"},
    "ProlongedQT":            {"LNGQT"},
    "Hemiblocks":             {"LAFB", "LPFB", "IVCD"},
    "LowVoltageQRS":          {"LVOLT"},
    "AtrialEnlargement":      {"LAO/LAE", "RAO/RAE"},
    "QWaveMI":                {"IMI", "AMI", "ASMI", "ALMI", "ILMI",
                               "LMI", "PMI", "IPMI", "IPLMI"},
    "STElevationInjury":      {"INJAS", "INJAL", "INJIL", "INJIN", "INJLA", "STE_"},
    "STDepressionIschemia":   {"STD_", "ISC_", "ISCAS", "ISCAL", "ISCIL",
                               "ISCIN", "ISCAN", "ISCLA"},
    "TWaveIschemia":          {"INVT", "NDT", "NT_", "TAB_", "LOWT"},
    "SinusRhythm":            {"SR", "SBRAD", "STACH"},
    "ProlongedQT":            {"LNGQT"},
    "NonspecificIVCB":        {"IVCD"},
    "BiventricularHypertrophy": {"BIVH"},
    "PulmonaryDiseasePattern": set(),
    "QRSAxis":                 set(),
    "Brugada":                 set(),
    "PericarditisOrEarlyRepol": set(),
    "AbnormalQRSTAngle":       set(),
    "AcuteMISTEMI":            set(),
    "NonspecificSTElevation":  set(),
    "NonspecificSTDepression": set(),
    "NonspecificTWave":        set(),
    "STElevationMechanismUnknown": set(),
    "PoorRWaveProgression":    set(),
    "PRInterval":              {"1AVB"},
    "AVBlock":                 {"3AVB", "2AVB"},
    "SinusRhythm":            {"SR", "SBRAD", "STACH"},
    "EctopicAtrialRhythm":    set(),
    "JunctionalRhythm":       set(),
    "UndeterminedRhythm":     set(),
    "Ectopy":                  {"PVC", "PAC", "SVPB", "VPB"},
    "SinusArrhythmia":         {"SARRH"},
    "Pacing":                  {"PACE"},
    "ElectrodeReversals":      set(),
    # Pediatric rules — PTB-XL has very few peds so most will show 0/0
    "PedWPW": set(), "PedDextrocardia": set(), "PedAtrialEnlargement": set(),
    "PedQRSAxis": set(), "PedLowVoltage": set(), "PedBrugada": set(),
    "PedConduction": set(), "PedRVH": set(), "PedLVH": set(),
    "PedMI": set(), "PedSTElevation": set(), "PedTWave": set(),
    "PedProlongedQT": set(), "PedEarlyRepolPericarditis": set(),
    "PedSTDepression": set(), "PedBiventricularHypertrophy": set(),
    "PedNonspecificBlock": set(),
}


def load_diseases() -> dict:
    diseases = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        d = load_disease_yaml(yf)
        diseases[d.disease] = d
    return diseases


def build_ctx(row: pd.Series) -> EvalContext:
    feats: dict[str, float] = {}
    for k, v in row.items():
        if pd.isna(v):
            continue
        ks = str(k)
        if ks.startswith("ge_"):
            feats[ks[3:]] = float(v)
        elif ks in ("ecg_id", "patient_id", "sex", "age_years", "strat_fold",
                    "heart_axis", "device", "report", "scp_codes_json",
                    "scp_codes_str", "_error"):
            continue
        elif isinstance(v, (int, float)):
            feats[ks] = float(v)
    sex = row.get("sex") if pd.notna(row.get("sex")) else None
    age = int(row["age_years"]) if pd.notna(row.get("age_years")) else None
    return EvalContext(features=feats, sex=sex, age_years=age)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_auc_binary(tp: int, fp: int, fn: int, tn: int) -> tuple[float, float]:
    """Compute ROC AUC and PR AUC from a single binary operating point."""
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    fpr = 1 - spec
    ppv = safe_div(tp, tp + fp)
    prev = safe_div(tp + fn, tp + fp + fn + tn)

    # ROC AUC from two points: (0,0), (fpr, sens), (1,1)
    # = area of trapezoid
    roc_auc = 0.5 * (1 + sens - fpr) if (tp + fn > 0 and tn + fp > 0) else float('nan')

    # PR AUC from two points: (0, ppv_at_zero_recall), (recall=sens, ppv)
    # Simple: area under step from (0, precision) to (recall, precision) + baseline
    # For a single threshold: PR AUC ≈ average precision = ppv * sens + (1-sens) * prev
    # More accurately: interpolated area
    if tp + fn == 0:
        pr_auc = float('nan')
    else:
        # Two points on PR curve: (0, 1.0) [no predictions, perfect precision trivially]
        # and (sens, ppv). Plus the baseline (1.0, prev).
        # Standard: PR-AUC for binary = sens * ppv + (1 - sens) * prev (approx)
        # Actually simplest correct: just report ppv * sens as the single-point AP
        pr_auc = ppv * sens  # average precision at single threshold

    return roc_auc, pr_auc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORTS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diseases = load_diseases()
    print(f"Loaded {len(diseases)} rules, "
          f"{sum(len(d.variants) for d in diseases.values())} variants.")

    df = pd.read_parquet(args.parquet)
    if args.fold is not None:
        df = df[df["strat_fold"] == args.fold]
    if args.limit:
        df = df.head(args.limit)
    df = df[df.get("_error").isna()] if "_error" in df.columns else df
    print(f"Evaluating on {len(df)} records.")

    # Pre-allocate per-disease confusion counters
    confusion: dict[str, dict[str, int]] = {
        dn: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for dn in diseases
    }
    fire_counts: dict[str, int] = {dn: 0 for dn in diseases}
    variant_fires: dict[tuple[str, str], int] = {}
    for d in diseases.values():
        for v in d.variants:
            variant_fires[(d.disease, v.name)] = 0

    skipped = 0
    evaluated = 0

    for _, row in tqdm(df.iterrows(), total=len(df), unit="ecg"):
        if pd.isna(row.get("QRS_ms")):
            skipped += 1
            continue
        evaluated += 1
        ctx = build_ctx(row)
        scp_codes = set(json.loads(row.get("scp_codes_json") or "{}").keys())

        for dn, d in diseases.items():
            res = evaluate_disease(d, ctx)
            fired = res.fired

            if fired:
                fire_counts[dn] += 1
            for v in res.variant_results:
                if v.fired:
                    variant_fires[(dn, v.variant_name)] += 1

            gt_codes = DISEASE_SCP_GT.get(dn, set())
            if gt_codes:
                gt_pos = bool(gt_codes & scp_codes)
                if fired and gt_pos:
                    confusion[dn]["tp"] += 1
                elif fired and not gt_pos:
                    confusion[dn]["fp"] += 1
                elif not fired and gt_pos:
                    confusion[dn]["fn"] += 1
                else:
                    confusion[dn]["tn"] += 1

    n = evaluated
    print(f"\nEvaluated {n} records (skipped {skipped} with no features).")

    # ── Build results table ─────────────────────────────────────────────
    rows = []
    for dn in diseases:
        c = confusion[dn]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        gt_codes = DISEASE_SCP_GT.get(dn, set())
        has_gt = bool(gt_codes)
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
            roc_auc, pr_auc = compute_auc_binary(tp, fp, fn, tn)
        else:
            roc_auc = pr_auc = float('nan')

        rows.append({
            "disease": dn,
            "scp_gt": ",".join(sorted(gt_codes)) or "(no GT)",
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
            "prevalence": safe_div(positives, n),
            "fire_rate": safe_div(fire_counts[dn], n),
            "fires": fire_counts[dn],
        })

    res_df = pd.DataFrame(rows)
    # Sort: diseases with GT first (by F1 desc), then no-GT (by fire rate)
    res_df["has_gt"] = res_df["scp_gt"] != "(no GT)"
    res_df = res_df.sort_values(
        ["has_gt", "F1", "fire_rate"],
        ascending=[False, False, False],
    ).drop(columns=["has_gt"])

    # ── Write CSV ───────────────────────────────────────────────────────
    res_df.to_csv(args.out / "ptbxl_full_metrics.csv", index=False)

    # ── Write variant fires CSV ─────────────────────────────────────────
    var_rows = []
    for (dn, vn), fires in variant_fires.items():
        var_rows.append({
            "disease": dn, "variant": vn,
            "fires": fires, "fire_rate": safe_div(fires, n),
        })
    var_df = pd.DataFrame(var_rows).sort_values(
        ["disease", "fires"], ascending=[True, False]
    )
    var_df.to_csv(args.out / "ptbxl_full_variant_fires.csv", index=False)

    # ── Write Markdown summary ──────────────────────────────────────────
    md = [
        "# PTB-XL Full Benchmark — All Rules",
        "",
        f"- Records evaluated: **{n}** (skipped {skipped})",
        f"- Rules: **{len(diseases)}** diseases, "
        f"**{sum(len(d.variants) for d in diseases.values())}** variants",
        f"- Fold filter: {args.fold if args.fold is not None else 'ALL (1-10)'}",
        "",
        "## Diseases WITH SCP Ground Truth",
        "",
        "| Disease | Pos | Neg | TP | FP | FN | TN | Sens | Spec | PPV | NPV | F1 | ROC AUC | PR AUC | Prev |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in res_df.iterrows():
        if r["scp_gt"] == "(no GT)":
            continue
        md.append(
            f"| {r['disease']} | {r['positives_GT']} | {r['negatives_GT']} | "
            f"{r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | "
            f"{r['sensitivity']*100:.1f}% | {r['specificity']*100:.1f}% | "
            f"{r['PPV']*100:.1f}% | {r['NPV']*100:.1f}% | "
            f"{r['F1']*100:.1f}% | "
            f"{r['ROC_AUC']:.3f} | {r['PR_AUC']:.3f} | "
            f"{r['prevalence']*100:.1f}% |"
        )

    md += [
        "",
        "## Diseases WITHOUT SCP Ground Truth (fire rate only)",
        "",
        "| Disease | Fires | Fire rate |",
        "|---|---:|---:|",
    ]
    for _, r in res_df.iterrows():
        if r["scp_gt"] != "(no GT)":
            continue
        md.append(f"| {r['disease']} | {int(r['fires'])} | {r['fire_rate']*100:.2f}% |")

    md += [
        "",
        "## Variant-level fire rates (top 20 by fires)",
        "",
        "| Disease | Variant | Fires | Fire rate |",
        "|---|---|---:|---:|",
    ]
    top_var = var_df.nlargest(20, "fires")
    for _, r in top_var.iterrows():
        md.append(
            f"| {r['disease']} | {r['variant']} | {int(r['fires'])} | "
            f"{r['fire_rate']*100:.2f}% |"
        )

    (args.out / "ptbxl_full_summary.md").write_text("\n".join(md))

    # ── Print to stdout ─────────────────────────────────────────────────
    print("\n=== DISEASES WITH GT (sorted by F1) ===\n")
    gt_df = res_df[res_df["scp_gt"] != "(no GT)"]
    cols = ["disease", "positives_GT", "TP", "FP", "FN", "TN",
            "sensitivity", "specificity", "PPV", "NPV", "F1", "ROC_AUC", "PR_AUC"]
    fmt = gt_df[cols].copy()
    for c in ["sensitivity", "specificity", "PPV", "NPV", "F1"]:
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    for c in ["ROC_AUC", "PR_AUC"]:
        fmt[c] = fmt[c].round(3)
    print(fmt.to_string(index=False))

    print(f"\n\nWrote reports to {args.out}/")


if __name__ == "__main__":
    main()
