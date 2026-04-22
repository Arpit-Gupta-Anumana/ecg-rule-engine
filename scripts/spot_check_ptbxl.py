"""Spot-check the full rule engine against PTB-XL SCP-coded labels.

Runs a sample of PTB-XL records through a set of our rules and computes a
confusion matrix against the SCP codes (which PTB-XL provides as clinician-
validated ground truth - folds 9 and 10 are especially reliable).

Unlike the GE-text spot-checks, this uses the SCP-code SET directly as the GT
signal, which is far cleaner than substring-matching a free-form interpretation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
from ecg_rule_engine.features.ptbxl import iter_records


ROOT = "/Users/arpit.gupta/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Rule name -> (yaml path, set of SCP codes that should make this rule fire)
RULES: dict[str, tuple[str, set[str]]] = {
    "LBBB":                   ("rules/lbbb.yaml",                 {"CLBBB"}),
    "RBBB":                   ("rules/rbbb.yaml",                 {"CRBBB"}),
    "IncompleteBundleBlocks": ("rules/incomplete_blocks.yaml",    {"IRBBB", "ILBBB"}),
    "LVH":                    ("rules/lvh.yaml",                  {"LVH", "VCLVH"}),
    "RVH":                    ("rules/rvh.yaml",                  {"RVH"}),
    "AFib":                   ("rules/afib.yaml",                 {"AFIB"}),
    "AtrialFlutter":          ("rules/atrial_flutter.yaml",       {"AFLT"}),
    "WPW":                    ("rules/wpw.yaml",                  {"WPW"}),
    "ProlongedQT":            ("rules/prolonged_qt.yaml",         {"LNGQT"}),
    "Hemiblocks":             ("rules/hemiblocks.yaml",           {"LAFB", "LPFB", "IVCD"}),
    "LowVoltageQRS":          ("rules/low_voltage_qrs.yaml",      {"LVOLT"}),
    "AtrialEnlargement":      ("rules/atrial_enlargement.yaml",   {"LAO/LAE", "RAO/RAE"}),
    "QRS_Axis":               ("rules/qrs_axis.yaml",             set()),  # axis is not an SCP code
    "Q_Wave_MI":              ("rules/q_wave_mi.yaml",            {"IMI", "AMI", "ASMI", "ALMI",
                                                                    "ILMI", "LMI", "PMI",
                                                                    "IPMI", "IPLMI"}),
    "ST_Elevation_Injury":    ("rules/st_elevation_injury.yaml",  {"INJAS", "INJAL", "INJIL",
                                                                    "INJIN", "INJLA", "STE_"}),
    "ST_Depression_Ischemia": ("rules/st_depression_ischemia.yaml", {"STD_", "ISC_", "ISCAS",
                                                                     "ISCAL", "ISCIL", "ISCIN",
                                                                     "ISCAN", "ISCLA"}),
    "T_Wave_Ischemia":        ("rules/t_wave_ischemia.yaml",      {"INVT", "NDT", "NT_", "TAB_"}),
}


def _fmt_pct(num: int, denom: int) -> str:
    return f"{100*num/max(1,denom):5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="number of records")
    ap.add_argument("--fold", type=int, default=10,
                    help="stratification fold (10=test, highest label quality)")
    ap.add_argument("--show-disagreements", type=int, default=10)
    args = ap.parse_args()

    rules_root = Path("/Users/arpit.gupta/Desktop/ecg_rule_engine")
    diseases = {name: load_disease_yaml(rules_root / p) for name, (p, _) in RULES.items()}
    scp_sets = {name: s for name, (_, s) in RULES.items()}

    counts = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in diseases}
    disagreements: dict[str, list[tuple[str, str, str, set[str]]]] = {n: [] for n in diseases}

    processed = 0
    feat_extract_failed = 0
    for rec in iter_records(ROOT, folds=[args.fold], n=args.n, random_state=42):
        processed += 1
        if not rec.features:
            feat_extract_failed += 1
            continue
        merged = dict(rec.features)
        merged.update({k: float(v) for k, v in rec.ge_flags.items()})
        ctx = EvalContext(features=merged, sex=rec.sex, age_years=rec.age_years)

        codes_present = set(rec.scp_codes_raw.keys())
        for name, d in diseases.items():
            res = evaluate_disease(d, ctx)
            gt_codes = scp_sets[name]
            gt_pos = bool(gt_codes & codes_present) if gt_codes else None
            if gt_pos is None:
                continue  # skip rules that don't map to an SCP label
            fired_variants = ",".join(v.variant_name for v in res.variant_results if v.fired)
            if res.fired and gt_pos:
                counts[name]["tp"] += 1
            elif res.fired and not gt_pos:
                counts[name]["fp"] += 1
                disagreements[name].append((str(rec.ptbxl_ecg_id), "FP",
                                            fired_variants, codes_present))
            elif not res.fired and gt_pos:
                counts[name]["fn"] += 1
                disagreements[name].append((str(rec.ptbxl_ecg_id), "FN",
                                            fired_variants or "-", codes_present))
            else:
                counts[name]["tn"] += 1

    print(f"\n=== {processed} PTB-XL ECGs (fold {args.fold}), "
          f"{feat_extract_failed} feature-extract failures ===")
    print(f"{'Rule':30s}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}   Sens   Spec   PPV")
    for name, c in counts.items():
        if not scp_sets[name]:
            continue
        sens = _fmt_pct(c['tp'], c['tp'] + c['fn'])
        spec = _fmt_pct(c['tn'], c['tn'] + c['fp'])
        ppv  = _fmt_pct(c['tp'], c['tp'] + c['fp'])
        print(f"{name:30s}  {c['tp']:>4} {c['fp']:>4} {c['fn']:>4} {c['tn']:>4}   "
              f"{sens}  {spec}  {ppv}")

    if args.show_disagreements > 0:
        print(f"\n=== Up to {args.show_disagreements} disagreements per rule ===")
        for name, dis in disagreements.items():
            if not dis:
                continue
            print(f"\n[{name}]")
            for ecg_id, kind, fired, codes in dis[:args.show_disagreements]:
                tagged = sorted(codes & scp_sets[name]) or "-"
                print(f"  ecg_id={ecg_id:>6s}  {kind}  fired={fired:<30s}  "
                      f"scp_gt={tagged}  all_codes={sorted(codes)}")


if __name__ == "__main__":
    main()
