"""Benchmark all rule-engine diseases/variants on the cached PTB-XL features.

Reads data/ptbxl_features.parquet (produced by cache_features_ptbxl.py), runs
every rule in rules/*.yaml against every record, and compares to SCP-coded
ground truth.

Outputs:
  reports/ptbxl_run_summary.md     - human-readable summary
  reports/ptbxl_run_disease.csv    - one row per (ecg_id, disease) with fired?
  reports/ptbxl_run_variant.csv    - one row per (ecg_id, variant)   with fired?
  reports/ptbxl_run_confusion.csv  - TP/FP/FN/TN per disease, with GT=SCP codes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease


ROOT = Path("/Users/arpit.gupta/Desktop/ecg_rule_engine")
PARQUET = ROOT / "data" / "ptbxl_features.parquet"
REPORTS = ROOT / "reports"
RULES_DIR = ROOT / "rules"


# Disease (YAML name) -> SCP code set whose presence means "ground truth positive"
# for this disease. Empty set means "no direct SCP mapping" - we still fire the
# rule and report fire rate, just no confusion matrix.
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
    # Rules without a direct SCP mapping: report fire rate only.
    "QRSAxis":                 set(),
    "Brugada":                 set(),
    "PericarditisOrEarlyRepol": set(),
}


def load_diseases() -> dict:
    diseases = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        d = load_disease_yaml(yf)
        diseases[d.disease] = d
    return diseases


def build_ctx(row: pd.Series) -> EvalContext:
    """Build an EvalContext from one row of the cached parquet.

    The parquet has:
      - feature columns (R_V1_mV, ...), QRS_ms etc
      - ge_* flag columns (ge_rhythm_afib_flag, ...)
      - demographics: sex, age_years
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", type=Path, default=PARQUET)
    ap.add_argument("--fold", type=int, default=None,
                    help="restrict to one strat_fold (e.g. 10 for test)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=REPORTS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    diseases = load_diseases()
    print(f"Loaded {len(diseases)} rules covering "
          f"{sum(len(d.variants) for d in diseases.values())} variants.")

    df = pd.read_parquet(args.parquet)
    if args.fold is not None:
        df = df[df["strat_fold"] == args.fold]
    if args.limit:
        df = df.head(args.limit)
    df = df[df.get("_error").isna()] if "_error" in df.columns else df
    print(f"Evaluating on {len(df)} records.")

    disease_rows, variant_rows = [], []
    confusion = {dn: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for dn in diseases}
    variant_fires = {}
    for d in diseases.values():
        for v in d.variants:
            variant_fires[(d.disease, v.name)] = {"fires": 0, "fp_on_norm": 0}

    # Handle records where the extractor failed - they have no features
    # (all NaN) and should be skipped in the confusion matrix so they don't
    # show as TN trivially.
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df), unit="ecg"):
        # A record without measured QRS_ms almost certainly has no extracted
        # features at all; skip it.
        if pd.isna(row.get("QRS_ms")):
            skipped += 1
            continue
        ctx = build_ctx(row)
        scp_codes = set(json.loads(row.get("scp_codes_json") or "{}").keys())
        is_norm = "NORM" in scp_codes

        for dn, d in diseases.items():
            res = evaluate_disease(d, ctx)
            fired_variants = [v.variant_name for v in res.variant_results if v.fired]
            disease_rows.append({
                "ecg_id": int(row["ecg_id"]),
                "disease": dn,
                "fired": int(res.fired),
                "fired_variants": "|".join(fired_variants),
            })
            for v in res.variant_results:
                variant_rows.append({
                    "ecg_id": int(row["ecg_id"]),
                    "disease": dn,
                    "variant": v.variant_name,
                    "fired": int(v.fired),
                })
                if v.fired:
                    key = (dn, v.variant_name)
                    variant_fires[key]["fires"] += 1
                    if is_norm:
                        variant_fires[key]["fp_on_norm"] += 1

            gt_codes = DISEASE_SCP_GT.get(dn, set())
            if gt_codes:
                gt_pos = bool(gt_codes & scp_codes)
                if res.fired and gt_pos:
                    confusion[dn]["tp"] += 1
                elif res.fired and not gt_pos:
                    confusion[dn]["fp"] += 1
                elif not res.fired and gt_pos:
                    confusion[dn]["fn"] += 1
                else:
                    confusion[dn]["tn"] += 1

    n = len(df) - skipped
    print(f"\nSkipped {skipped} records that had no extracted features.")

    # --- Write CSVs ---
    pd.DataFrame(disease_rows).to_csv(args.out / "ptbxl_run_disease.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(args.out / "ptbxl_run_variant.csv", index=False)

    conf_df = []
    for dn, c in confusion.items():
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        ppv  = tp / max(1, tp + fp)
        npv  = tn / max(1, tn + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        conf_df.append({
            "disease": dn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "prevalence": (tp + fn) / max(1, n),
            "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
            "f1": f1, "fire_rate": (tp + fp) / max(1, n),
            "scp_gt_codes": ",".join(sorted(DISEASE_SCP_GT.get(dn, []))) or "(none)",
        })
    conf_df = pd.DataFrame(conf_df).sort_values("f1", ascending=False)
    conf_df.to_csv(args.out / "ptbxl_run_confusion.csv", index=False)

    var_df = []
    for (dn, vn), vc in variant_fires.items():
        var_df.append({
            "disease": dn, "variant": vn,
            "fires": vc["fires"], "fp_on_norm": vc["fp_on_norm"],
            "fire_rate": vc["fires"] / max(1, n),
            "fp_rate_on_norm": vc["fp_on_norm"] / max(1, n),
        })
    var_df = pd.DataFrame(var_df).sort_values(["disease", "fires"], ascending=[True, False])
    var_df.to_csv(args.out / "ptbxl_run_variant_fires.csv", index=False)

    # --- Human-readable summary ---
    md_lines = [
        f"# PTB-XL rule-engine run",
        "",
        f"- records evaluated: **{n}**  (skipped {skipped})",
        f"- rules: **{len(diseases)}** diseases, **{sum(len(d.variants) for d in diseases.values())}** variants",
        f"- fold filter: {args.fold if args.fold is not None else 'ALL (1-10)'}",
        "",
        "## Confusion matrix vs SCP-coded GT",
        "",
        "| Disease | SCP GT codes | TP | FP | FN | TN | Prev | Sens | Spec | PPV | F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in conf_df.iterrows():
        md_lines.append(
            f"| {r['disease']} | `{r['scp_gt_codes']}` | {int(r['tp'])} | {int(r['fp'])} | "
            f"{int(r['fn'])} | {int(r['tn'])} | {r['prevalence']*100:.1f}% | "
            f"{r['sensitivity']*100:.1f}% | {r['specificity']*100:.1f}% | "
            f"{r['ppv']*100:.1f}% | {r['f1']*100:.1f}% |"
        )
    md_lines += [
        "",
        "## Variant fire counts (top 10 per disease)",
        "",
        "| Disease | Variant | Fires | FP-on-NORM | Fire rate |",
        "|---|---|---:|---:|---:|",
    ]
    for _, r in var_df.iterrows():
        md_lines.append(
            f"| {r['disease']} | {r['variant']} | {int(r['fires'])} | "
            f"{int(r['fp_on_norm'])} | {r['fire_rate']*100:.2f}% |"
        )
    (args.out / "ptbxl_run_summary.md").write_text("\n".join(md_lines))
    print(f"\nWrote reports to {args.out}/")
    # echo the confusion table to stdout
    print("\n=== Confusion (sorted by F1) ===")
    print(conf_df[["disease", "tp", "fp", "fn", "tn", "prevalence",
                    "sensitivity", "specificity", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
