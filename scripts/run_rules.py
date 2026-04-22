#!/usr/bin/env python3
"""Universal rule evaluation script for any CSV dataset.

Usage:
    # With default GE 12SL column names (built-in mapping):
    python scripts/run_rules.py --csv your_data.csv

    # With a custom column mapping JSON:
    python scripts/run_rules.py --csv your_data.csv --mapping your_mapping.json

    # Specify age/sex columns explicitly:
    python scripts/run_rules.py --csv your_data.csv --age-col patient_age --sex-col gender

    # Only run specific diseases:
    python scripts/run_rules.py --csv your_data.csv --diseases LBBB,RBBB,LVH

    # Limit to N records:
    python scripts/run_rules.py --csv your_data.csv --limit 500

    # Output predictions CSV (one row per ECG, one column per disease):
    python scripts/run_rules.py --csv your_data.csv --out-preds predictions.csv

Column Mapping:
    The mapping JSON maps our canonical feature IDs to your CSV column names.
    See feature_dictionary.json for all 138 feature IDs with descriptions.

    Example mapping file (my_mapping.json):
    {
        "QRS_ms":              "QRS_Duration",
        "PR_ms":               "PR_Interval",
        "ventricular_rate_bpm": "HeartRate",
        "R_V1_mV":             "R_amp_V1",
        "S_V1_mV":             {"column": "S_amp_V1", "transform": "abs"},
        "STM_V1_mV":           {"column": "ST_mid_V1", "transform": "divide_1000"},
        "age_years":           "patient_age",
        "sex_col":             "gender",
        "sex_male_value":      "M"
    }

    Transform options for each feature:
        "abs"          - take absolute value (for Q/S waves stored as negative)
        "divide_1000"  - divide by 1000 (for microvolts → millivolts)
        "negate"       - multiply by -1

    If no mapping file is provided, the built-in GE 12SL mapping is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

RULES_DIR = ROOT / "rules"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Built-in GE 12SL column mapping ─────────────────────────────────────
BUILTIN_MAPPING: dict[str, dict | str] = {
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
    "QRS_axis_deg": "R_AxisFrontal_Global",
    "P_axis_deg": "P_AxisFront_Global",
    "T_axis_deg": "T_AxisFront_Global",
}
for lead in LEADS:
    BUILTIN_MAPPING[f"R_{lead}_mV"] = f"R_Amp_{lead}"
    BUILTIN_MAPPING[f"Q_{lead}_mV"] = {"column": f"Q_Amp_{lead}", "transform": "abs"}
    BUILTIN_MAPPING[f"S_{lead}_mV"] = {"column": f"S_Amp_{lead}", "transform": "abs"}
    BUILTIN_MAPPING[f"P_{lead}_mV"] = f"P_Amp_{lead}"
    BUILTIN_MAPPING[f"T_{lead}_mV"] = f"T_Amp_{lead}"
    BUILTIN_MAPPING[f"STJ_{lead}_mV"] = f"ST_Amp_{lead}"
    BUILTIN_MAPPING[f"STM_{lead}_mV"] = {"column": f"ST_Amp116_{lead}", "transform": "divide_1000"}
    BUILTIN_MAPPING[f"Q_{lead}_ms"] = f"Q_Dur_{lead}"


def _safe(val) -> float | None:
    if pd.isna(val):
        return None
    return float(val)


def _apply_transform(val: float | None, transform: str | None) -> float | None:
    if val is None or transform is None:
        return val
    if transform == "abs":
        return abs(val)
    if transform == "divide_1000":
        return val / 1000.0
    if transform == "negate":
        return -val
    return val


def map_row(row: pd.Series, mapping: dict, age_col: str | None, sex_col: str | None,
            sex_male_value: str = "M") -> tuple[dict[str, float], str | None, int | None]:
    """Map a CSV row to the canonical feature dict using the given mapping."""
    f: dict[str, float] = {}

    for feat_id, spec in mapping.items():
        if feat_id in ("sex_col", "sex_male_value"):
            continue

        if isinstance(spec, str):
            col_name = spec
            transform = None
        elif isinstance(spec, dict):
            col_name = spec["column"]
            transform = spec.get("transform")
        else:
            continue

        if col_name not in row.index:
            continue

        val = _safe(row.get(col_name))
        val = _apply_transform(val, transform)
        if val is not None:
            f[feat_id] = val

    # Derived: PP_ms from atrial rate
    if "PP_ms" not in f and "atrial_rate_bpm" in f and f["atrial_rate_bpm"] > 0:
        f["PP_ms"] = 60000.0 / f["atrial_rate_bpm"]

    # Per-lead QRS duration (sum of Q+R+S durations if not directly mapped)
    for lead in LEADS:
        key = f"QRS_{lead}_ms"
        if key not in f:
            r_dur = _safe(row.get(f"R_Dur_{lead}")) or 0.0
            q_dur = _safe(row.get(f"Q_Dur_{lead}")) or 0.0
            s_dur = _safe(row.get(f"S_Dur_{lead}")) or 0.0
            total = r_dur + q_dur + s_dur
            if total > 0:
                f[key] = total

    # Demographics
    age = None
    sex = None
    if age_col and age_col in row.index:
        age_val = _safe(row.get(age_col))
        if age_val is not None:
            age = int(age_val)
    elif "age_years" in f:
        age = int(f["age_years"])

    if sex_col and sex_col in row.index:
        raw_sex = str(row.get(sex_col, "")).strip().upper()
        sex_m_val = (mapping.get("sex_male_value") or sex_male_value).strip().upper()
        if raw_sex == sex_m_val:
            sex = "M"
        elif raw_sex:
            sex = "F"

    return f, sex, age


def load_diseases(disease_filter: set[str] | None = None) -> dict:
    diseases = {}
    for yf in sorted(RULES_DIR.glob("*.yaml")):
        try:
            d = load_disease_yaml(yf)
            if disease_filter and d.disease not in disease_filter:
                continue
            diseases[d.disease] = d
        except Exception as e:
            print(f"  WARN: skipping {yf.name}: {e}", file=sys.stderr)
    return diseases


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run ECG rule engine on any CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--csv", type=Path, required=True, help="Path to input CSV file")
    ap.add_argument("--mapping", type=Path, default=None,
                    help="Path to column mapping JSON (see feature_dictionary.json for IDs)")
    ap.add_argument("--age-col", type=str, default=None, help="CSV column name for patient age")
    ap.add_argument("--sex-col", type=str, default=None, help="CSV column name for patient sex")
    ap.add_argument("--sex-male-value", type=str, default="M",
                    help="Value in sex column that indicates male (default: 'M')")
    ap.add_argument("--diseases", type=str, default=None,
                    help="Comma-separated list of diseases to evaluate (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N rows")
    ap.add_argument("--out-preds", type=Path, default=None,
                    help="Output CSV path for predictions (one column per disease)")
    ap.add_argument("--out-traces", type=Path, default=None,
                    help="Output JSON path for fire traces (detailed explanations)")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress bar")
    args = ap.parse_args()

    # Load mapping
    if args.mapping:
        with open(args.mapping) as f:
            mapping = json.load(f)
        print(f"Loaded custom mapping with {len(mapping)} entries from {args.mapping}")
    else:
        mapping = BUILTIN_MAPPING
        print("Using built-in GE 12SL column mapping")

    age_col = args.age_col or mapping.get("age_years")
    sex_col = args.sex_col or mapping.get("sex_col")
    sex_male_value = args.sex_male_value

    if isinstance(age_col, dict):
        age_col = age_col.get("column", age_col)
    if isinstance(sex_col, dict):
        sex_col = sex_col.get("column", sex_col)

    # Load diseases
    disease_filter = None
    if args.diseases:
        disease_filter = set(args.diseases.split(","))
    diseases = load_diseases(disease_filter)
    print(f"Loaded {len(diseases)} rule files, "
          f"{sum(len(d.variants) for d in diseases.values())} variants")

    # Load CSV
    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)
    print(f"Loaded {len(df)} records from {args.csv.name}")

    # Check how many features we can map
    sample_row = df.iloc[0]
    test_feats, _, _ = map_row(sample_row, mapping, age_col, sex_col, sex_male_value)
    print(f"Mapped {len(test_feats)} features from first row (of 138 possible)")

    # Evaluate
    pred_rows: list[dict[str, int]] = []
    trace_rows: list[dict] = []
    disease_names = sorted(diseases.keys())
    skipped = 0

    iterator = df.iterrows()
    if not args.quiet:
        iterator = tqdm(iterator, total=len(df), unit="ecg")

    for idx, row in iterator:
        feats, sex, age = map_row(row, mapping, age_col, sex_col, sex_male_value)

        if "QRS_ms" not in feats:
            skipped += 1
            pred_rows.append({dn: -1 for dn in disease_names})
            continue

        ctx = EvalContext(features=feats, sex=sex, age_years=age)
        preds: dict[str, int] = {}
        traces: dict[str, dict] = {}

        for dn in disease_names:
            res = evaluate_disease(diseases[dn], ctx)
            preds[dn] = int(res.fired)
            if args.out_traces:
                traces[dn] = res.to_dict()

        pred_rows.append(preds)
        if args.out_traces:
            trace_rows.append({"row_index": idx, "results": traces})

    print(f"\nProcessed {len(df)} records ({skipped} skipped due to missing QRS_ms)")

    # Summary
    pred_df = pd.DataFrame(pred_rows)
    print("\n=== Prediction Summary ===\n")
    for dn in disease_names:
        valid = pred_df[pred_df[dn] >= 0][dn]
        fires = valid.sum()
        total = len(valid)
        rate = fires / total if total > 0 else 0
        print(f"  {dn:35s}  fires: {fires:>5d} / {total:>5d}  ({rate:6.2%})")

    # Save predictions
    if args.out_preds:
        out_df = pred_df.copy()
        out_df.insert(0, "row_index", range(len(out_df)))
        out_df.to_csv(args.out_preds, index=False)
        print(f"\nPredictions saved to {args.out_preds}")

    if args.out_traces and trace_rows:
        with open(args.out_traces, "w") as f:
            json.dump(trace_rows, f, indent=2, default=str)
        print(f"Fire traces saved to {args.out_traces}")


if __name__ == "__main__":
    main()
