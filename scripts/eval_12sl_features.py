"""Evaluate all rules against GE 12SL-computed features from PTB-XL+ dataset.

Input:  /Users/arpit.gupta/Documents/PTB/ptb_testing_data.csv
        - 21,799 records with 788 columns of GE 12SL measurements + statements
Ground truth: the `statements` column (GE 12SL's own diagnostic output)

Usage:
    python scripts/eval_12sl_features.py
    python scripts/eval_12sl_features.py --limit 500
"""
from __future__ import annotations

import argparse
import ast
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

CSV_PATH = Path("/Users/arpit.gupta/Documents/PTB/ptb_testing_data.csv")
RULES_DIR = ROOT / "rules"
REPORTS = ROOT / "reports"

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Ground truth: 12SL statement acronyms → rule engine disease names ──────
# These are the GE 12SL's OWN output codes (from 12slv23ToSNOMED.csv)
DISEASE_GT: dict[str, set[str]] = {
    "LBBB":                {"LBBB"},
    "RBBB":                {"RBBB"},
    "IncompleteBundleBlocks": {"IRBBB", "ILBBB"},
    "LVH":                 {"LVH", "LVH2", "LVH3", "QRSV"},
    "RVH":                 {"RVH", "RVE+"},
    "AFib":                {"AFIB"},
    "AtrialFlutter":       {"FLUT"},
    "WPW":                 {"WPW", "WPWA", "WPWB"},
    "ProlongedQT":         {"LNGQT"},
    "Hemiblocks":          {"AFB", "PFB", "IVCB", "IVCD"},
    "LowVoltageQRS":       {"LOWV"},
    "AtrialEnlargement":   {"LAE", "RAE", "BAE"},
    "QWaveMI":             {"SMI", "AMI", "LMI", "IMI", "IPMI", "POSTMI",
                            "ASMI", "ALMI", "QESPMI"},
    "STElevationInjury":   {"AINJ", "LINJ", "IINJ", "ALINJ", "ILINJ",
                            "IIOHAI", "AIOHAI", "LIOHAI", "ALIHAI", "ILIHAI"},
    "STDepressionIschemia": {"NST", "STDEP2", "SSBINJ", "ASBINJ", "LSBINJ",
                             "ISBINJ", "MSTDIL", "MSTDAS", "MSTDAL", "JST"},
    "TWaveIschemia":       {"NT", "NSTT", "AT", "MAT", "LT", "MLT", "IT",
                            "MIT", "ILT", "MILT", "ALT", "MALT"},
    "SinusRhythm":         {"NSR", "SRTH", "SBRAD", "STACH", "MSBRAD"},
    "PRInterval":          {"FAV", "SPR"},
    "AVBlock":             {"CHB", "SAV", "MBZI", "MBZII", "VAVB", "W2T1",
                            "W3T1", "W4T1", "AVDIS"},
    "Pacing":              {"APR", "VPR", "ASVPR", "AVDPR"},
    "NonspecificIVCB":     {"IVCB", "IVCD"},
    "BiventricularHypertrophy": {"BIVH"},
    "PulmonaryDiseasePattern": {"PULD"},
    "Ectopy":              {"PVC", "PAC", "PSVC", "ABER"},
    "SinusArrhythmia":     {"SAR", "MSAR"},
    "EctopicAtrialRhythm": {"EAR", "EABRAD", "EATACH"},
    "JunctionalRhythm":    {"JR", "JTACH", "JUNBRAD", "JUNCT-R"},
    "UndeterminedRhythm":  {"UR"},
    "AcuteMISTEMI":        {"STEMI"},
    "NonspecificSTElevation": {"SERYR1", "SERYR2"},
    "NonspecificTWave":    {"NT", "NSTT", "QRST"},
    "AbnormalQRSTAngle":   {"QRST"},
    "ElectrodeReversals":  {"ARM"},
}

# Diseases with no 12SL GT mapping (fire rate only)
NO_GT = {
    "QRSAxis", "Brugada", "PericarditisOrEarlyRepol",
    "STElevationMechanismUnknown", "PoorRWaveProgression",
    "NonspecificSTDepression",
    # Pediatric rules — 12SL doesn't output pediatric-specific codes
    "PedWPW", "PedDextrocardia", "PedAtrialEnlargement",
    "PedQRSAxis", "PedLowVoltage", "PedBrugada",
    "PedConduction", "PedRVH", "PedLVH",
    "PedMI", "PedSTElevation", "PedTWave",
    "PedProlongedQT", "PedEarlyRepolPericarditis",
    "PedSTDepression", "PedBiventricularHypertrophy",
    "PedNonspecificBlock",
}


def map_row_to_features(row: pd.Series) -> dict[str, float]:
    """Map CSV columns to rule-engine feature names."""
    f: dict[str, float] = {}

    def _safe(val) -> float | None:
        if pd.isna(val):
            return None
        return float(val)

    # Global intervals / rates
    v = _safe(row.get("QRS_Dur_Global"))
    if v is not None:
        f["QRS_ms"] = v
    v = _safe(row.get("PR_Int_Global"))
    if v is not None:
        f["PR_ms"] = v
    v = _safe(row.get("QT_Int_Global"))
    if v is not None:
        f["QT_ms"] = v
    v = _safe(row.get("QT_IntBazett_Global"))
    if v is not None:
        f["QTc_Bazett_ms"] = v
    v = _safe(row.get("QT_IntFridericia_Global"))
    if v is not None:
        f["QTc_Fridericia_ms"] = v
    v = _safe(row.get("QT_IntFramingham_Global"))
    if v is not None:
        f["QTc_Framingham_ms"] = v
    v = _safe(row.get("HR_Ventr_Global"))
    if v is not None:
        f["ventricular_rate_bpm"] = v
    v = _safe(row.get("HR_Atrial_Global"))
    if v is not None:
        f["atrial_rate_bpm"] = v
    v = _safe(row.get("RR_Mean_Global"))
    if v is not None:
        f["RR_ms"] = v
    v = _safe(row.get("P_Dur_Global"))
    if v is not None:
        f["P_ms"] = v

    # PP interval: not directly in CSV, approximate from atrial rate
    atr = _safe(row.get("HR_Atrial_Global"))
    if atr is not None and atr > 0:
        f["PP_ms"] = 60000.0 / atr

    # Axes
    v = _safe(row.get("R_AxisFrontal_Global"))
    if v is not None:
        f["QRS_axis_deg"] = v
    v = _safe(row.get("P_AxisFront_Global"))
    if v is not None:
        f["P_axis_deg"] = v
    v = _safe(row.get("T_AxisFront_Global"))
    if v is not None:
        f["T_axis_deg"] = v

    # Per-lead features
    for lead in LEADS:
        # R amplitude (positive in CSV, positive in our rules)
        v = _safe(row.get(f"R_Amp_{lead}"))
        if v is not None:
            f[f"R_{lead}_mV"] = max(v, 0.0)

        # Q amplitude (negative in CSV, our rules expect positive magnitude)
        v = _safe(row.get(f"Q_Amp_{lead}"))
        if v is not None:
            f[f"Q_{lead}_mV"] = abs(v)

        # S amplitude (negative in CSV, our rules expect positive magnitude)
        v = _safe(row.get(f"S_Amp_{lead}"))
        if v is not None:
            f[f"S_{lead}_mV"] = abs(v)

        # Q duration
        v = _safe(row.get(f"Q_Dur_{lead}"))
        if v is not None:
            f[f"Q_{lead}_ms"] = v

        # P amplitude (signed — can be positive or negative)
        v = _safe(row.get(f"P_Amp_{lead}"))
        if v is not None:
            f[f"P_{lead}_mV"] = v

        # T amplitude (signed)
        v = _safe(row.get(f"T_Amp_{lead}"))
        if v is not None:
            f[f"T_{lead}_mV"] = v

        # ST-J amplitude (at J-point)
        v = _safe(row.get(f"ST_Amp_{lead}"))
        if v is not None:
            f[f"STJ_{lead}_mV"] = v

        # ST mid-segment: use ST_Amp116 (1/16 into ST segment) as proxy
        # The CSV stores this in microvolts-ish (integer), convert
        v = _safe(row.get(f"ST_Amp116_{lead}"))
        if v is not None:
            f[f"STM_{lead}_mV"] = v / 1000.0  # µV → mV

        # Per-lead QRS duration (use global as fallback; CSV doesn't have per-lead QRS dur)
        # Actually we have R_Dur + Q_Dur + S_Dur which sums to QRS per lead
        r_dur = _safe(row.get(f"R_Dur_{lead}")) or 0.0
        q_dur = _safe(row.get(f"Q_Dur_{lead}")) or 0.0
        s_dur = _safe(row.get(f"S_Dur_{lead}")) or 0.0
        qrs_lead = r_dur + q_dur + s_dur
        if qrs_lead > 0:
            f[f"QRS_{lead}_ms"] = qrs_lead

    return f


def parse_statements(raw: str) -> set[str]:
    """Parse the statements column which looks like: ['NSR', 'NML']"""
    try:
        lst = ast.literal_eval(raw)
        result = set()
        for item in lst:
            # Handle compound codes like 'ASMI;AU' — split on semicolons
            for part in str(item).split(";"):
                result.add(part.strip())
        return result
    except Exception:
        return set()


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_PATH)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORTS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    diseases = load_diseases()
    print(f"Loaded {len(diseases)} rules, "
          f"{sum(len(d.variants) for d in diseases.values())} variants.")

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} records from {args.csv.name}")

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
        feats = map_row_to_features(row)
        if "QRS_ms" not in feats:
            skipped += 1
            continue
        evaluated += 1

        stmts = parse_statements(str(row.get("statements", "[]")))

        # No SCP-derived flags — we don't cheat.
        # For exclusion flags that reference other conditions, derive from
        # the 12SL statements (this is legitimate: the manual says "if LBBB
        # is stated, skip X" — the 12SL stated it, so we use that).
        feats["lbbb_flag"] = float("LBBB" in stmts)
        feats["rbbb_flag"] = float("RBBB" in stmts)
        feats["ilbbb_flag"] = float("ILBBB" in stmts)
        feats["irbbb_flag"] = float("IRBBB" in stmts)
        feats["lafb_flag"] = float("AFB" in stmts)
        feats["lpfb_flag"] = float("PFB" in stmts)
        feats["rvh_flag"] = float("RVH" in stmts or "RVE+" in stmts)
        feats["ivcb_flag"] = float(
            "IVCB" in stmts or "IVCD" in stmts
            or "ILBBB" in stmts or "IRBBB" in stmts
        )
        feats["wpw_flag"] = float(
            "WPW" in stmts or "WPWA" in stmts or "WPWB" in stmts
        )
        feats["paced_flag"] = float(
            "APR" in stmts or "VPR" in stmts
            or "ASVPR" in stmts or "AVDPR" in stmts
        )
        feats["rhythm_afib_flag"] = float("AFIB" in stmts)
        feats["rhythm_aflutter_flag"] = float("FLUT" in stmts)
        feats["rhythm_sinus_flag"] = float(
            bool({"NSR", "SRTH", "SBRAD", "STACH", "MSBRAD"} & stmts)
        )

        # Demographics: not in CSV, set to None so EvalContext handles it
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
    print(f"\nEvaluated {n} records (skipped {skipped}).")

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
            pr_auc = prev * sens + (1 - prev) * (1 - spec)
            pr_auc = prev + (sens - prev) * ppv if (sens + ppv) > 0 else prev
            # Single-threshold PR-AUC: interpolate between (0, precision@0)
            # and (recall, precision) and (1, prevalence).
            # Standard trapezoid: area = 0.5*(0 + sens)*(ppv + ppv) / 2 is
            # not quite right. Use the Davis-Goadrich lower bound for a
            # single operating point which equals the F1-weighted area:
            #   PR-AUC ≈ ppv * sens  (the rectangle under the point)
            # plus the triangle from (0, 1) to (sens, ppv) to (1, prev).
            # For a single binary threshold, the most honest estimate is
            # the weighted average of precision at the two recall anchors:
            #   (0, 1) → (sens, ppv) → (1, prevalence)
            # giving trapezoid = 0.5*sens*(1+ppv) + 0.5*(1-sens)*(ppv+prev)
            if total > 0:
                pr_auc = 0.5 * sens * (1 + ppv) + 0.5 * (1 - sens) * (ppv + prev)
            else:
                pr_auc = float('nan')
        else:
            roc_auc = float('nan')
            pr_auc = float('nan')

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
        })

    res_df = pd.DataFrame(rows_out)
    res_df["has_gt"] = res_df["gt_codes"] != "(no GT)"
    res_df = res_df.sort_values(
        ["has_gt", "F1", "fire_rate"],
        ascending=[False, False, False],
    ).drop(columns=["has_gt"])

    res_df.to_csv(args.out / "12sl_full_metrics.csv", index=False)

    var_rows = []
    for (dn, vn), fires in variant_fires.items():
        var_rows.append({
            "disease": dn, "variant": vn,
            "fires": fires, "fire_rate": safe_div(fires, n),
        })
    var_df = pd.DataFrame(var_rows).sort_values(
        ["disease", "fires"], ascending=[True, False]
    )
    var_df.to_csv(args.out / "12sl_variant_fires.csv", index=False)

    # Print summary
    print("\n=== DISEASES WITH GT (sorted by F1) ===\n")
    gt_df = res_df[res_df["gt_codes"] != "(no GT)"]
    cols = ["disease", "positives_GT", "TP", "FP", "FN", "TN",
            "sensitivity", "specificity", "PPV", "NPV", "F1", "ROC_AUC", "PR_AUC"]
    fmt = gt_df[cols].copy()
    for c in ["sensitivity", "specificity", "PPV", "NPV", "F1"]:
        fmt[c] = (fmt[c] * 100).round(1).astype(str) + "%"
    fmt["ROC_AUC"] = fmt["ROC_AUC"].round(3)
    fmt["PR_AUC"] = fmt["PR_AUC"].round(3)
    print(fmt.to_string(index=False))

    print("\n\n=== DISEASES WITHOUT GT (fire rate only) ===\n")
    no_gt_df = res_df[res_df["gt_codes"] == "(no GT)"]
    print(no_gt_df[["disease", "fires", "fire_rate"]].to_string(index=False))

    print(f"\n\nReports written to {args.out}/")


if __name__ == "__main__":
    main()
