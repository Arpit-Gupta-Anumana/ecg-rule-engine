#!/usr/bin/env python3
"""
Apply Post-Prediction Suppression
==================================
Takes model predictions (abnormality labels) and continuous measurements,
then applies the suppression rules defined in suppression_rules.yaml
to produce a cleaned, conflict-free set of diagnoses.

Usage
-----
1. Interactive / inline mode (pass everything on CLI):

    python apply_suppression.py \
        --labels "Afib,Left Bundle Branch Block,ST_ST Elevation" \
        --heart_rate 82 --pr_interval 160 --qrs_duration 140 --axis -20

2. CSV batch mode (one row per ECG):

    python apply_suppression.py \
        --csv model_predictions.csv \
        --label-cols "Afib,LBBB,RBBB,LVH,..."  \
        --hr-col heart_rate --pr-col pr_interval \
        --qrs-col qrs_duration --axis-col axis \
        -o suppressed_output.csv

3. JSON mode:

    python apply_suppression.py --json input.json
    
    where input.json is:
    {
        "predictions": {"Afib": 1, "Left Bundle Branch Block": 1, ...},
        "measurements": {"heart_rate": 82, "pr_interval": 160, ...}
    }
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from suppression.engine import apply_suppression


ALL_LABELS = [
    "Bifascicular block",
    "Bigeminy",
    "Left Bundle Branch Block",
    "Right Atrial Enlargement",
    "Biatrial Enlargement",
    "Left Anterior Fascicular Block",
    "WPW",
    "Left Posterior Fascicular Block",
    "Right Bundle Branch Block",
    "Right Ventricular Hypertrophy",
    "Trigeminy",
    "Premature Ventricular Complex",
    "Early Repolarization Pattern",
    "Premature Atrial Complex",
    "Left Ventricular Hypertrophy",
    "Left Atrial Enlargement",
    "Brugada Syndrome Pattern",
    "Premature Junctional Complexes",
    "T Wave Inversion",
    "ST_NegativeSet",
    "ST_ST Depression",
    "ST_ST Elevation",
    "AV_BLOCK_NegativeSet",
    "AV_BLOCK_1st Degree AV Block",
    "AV_BLOCK_2nd Degree AV Block Mobitz I",
    "AV_BLOCK_3rd Degree AV Block",
    "AV_BLOCK_3:1 a-v conduction",
    "AV_BLOCK_2:1 a-v conduction",
    "AV_BLOCK_5:1 a-v conduction",
    "AV_BLOCK_4:1 a-v conduction",
    "Acute MI Posterior",
    "Acute MI Lateral",
    "Acute MI Septal",
    "Acute MI Inferior",
    "Acute MI Anterior",
    "Old MI",
    "Afib",
    "SVT",
    "Sinus",
    "Paced",
    "Flutter",
    "Junctional",
    "Wide QRS",
    "Ectropic Atrial",
    "NICD",
]


def parse_labels(raw: str) -> dict[str, int]:
    """Comma-separated label names → prediction dict with 1s."""
    preds = {l: 0 for l in ALL_LABELS}
    for token in raw.split(","):
        token = token.strip()
        if token:
            preds[token] = 1
    return preds


def run_single(predictions, measurements, *, scores=None, screening=False, verbose=True):
    """Run suppression on one record and print results."""
    result = apply_suppression(
        predictions, measurements, scores=scores, screening_mode=screening,
    )

    if verbose:
        print("=" * 72)
        print("SUPPRESSION TRACE")
        print("=" * 72)
        for line in result["trace"]:
            print(line)
        print()

        print("─" * 72)
        print("INPUT LABELS  :", [k for k, v in predictions.items() if v])
        print("MEASUREMENTS  :", {k: v for k, v in measurements.items() if v is not None})
        print("─" * 72)
        print("FINAL LABELS  :", result["final_labels"])
        print("SUPPRESSED    :", list(result["suppressed"].keys()))
        print("ASSEMBLED     :", result["assembled"])
        print("─" * 72)
        if result["suppressed"]:
            print("\nSuppression details:")
            for label, reason in result["suppressed"].items():
                print(f"  • {label}")
                print(f"    Reason: {reason}")
        print()

    return result


def run_csv(args):
    """Batch mode: process a CSV of predictions."""
    label_cols = [c.strip() for c in args.label_cols.split(",")]
    meas_map = {}
    if args.hr_col:
        meas_map["heart_rate"] = args.hr_col
    if args.pr_col:
        meas_map["pr_interval"] = args.pr_col
    if args.qrs_col:
        meas_map["qrs_duration"] = args.qrs_col
    if args.axis_col:
        meas_map["axis"] = args.axis_col
    if args.qt_col:
        meas_map["qt_interval"] = args.qt_col
    if args.qtc_col:
        meas_map["qtc"] = args.qtc_col

    rows_out = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            preds = {}
            for col in label_cols:
                val = row.get(col, "0")
                preds[col] = 1 if val and float(val) >= 0.5 else 0

            meas = {}
            for canon, csv_col in meas_map.items():
                raw = row.get(csv_col)
                meas[canon] = float(raw) if raw and raw.strip() else None

            result = apply_suppression(preds, meas, screening_mode=args.screening)

            out_row = dict(row)
            for label in label_cols:
                out_row[f"{label}_suppressed"] = 1 if label in result["active_raw"] else 0
            out_row["final_labels"] = "; ".join(result["final_labels"])
            out_row["suppressed_labels"] = "; ".join(result["suppressed"].keys())
            rows_out.append(out_row)

            if (i + 1) % 500 == 0:
                print(f"  processed {i + 1} rows...")

    out_path = args.output or "suppressed_output.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nWrote {len(rows_out)} rows → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Apply post-prediction suppression rules to ECG diagnostic labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--labels", type=str,
                       help='Comma-separated active labels, e.g. "Afib,LBBB,ST_ST Elevation"')
    mode.add_argument("--csv", type=str,
                       help="Path to CSV with model predictions (one row per ECG)")
    mode.add_argument("--json", type=str,
                       help="Path to JSON file with predictions + measurements")

    parser.add_argument("--heart_rate", type=float, default=None)
    parser.add_argument("--pr_interval", type=float, default=None)
    parser.add_argument("--qrs_duration", type=float, default=None)
    parser.add_argument("--axis", type=float, default=None)
    parser.add_argument("--qt_interval", type=float, default=None)
    parser.add_argument("--qtc", type=float, default=None)
    parser.add_argument("--screening", action="store_true",
                       help="Enable screening mode (high-specificity, suppresses borderline labels)")

    # CSV-mode options
    parser.add_argument("--label-cols", type=str, default=None,
                       help="Comma-separated column names that hold binary predictions")
    parser.add_argument("--hr-col", type=str, default=None)
    parser.add_argument("--pr-col", type=str, default=None)
    parser.add_argument("--qrs-col", type=str, default=None)
    parser.add_argument("--axis-col", type=str, default=None)
    parser.add_argument("--qt-col", type=str, default=None)
    parser.add_argument("--qtc-col", type=str, default=None)
    parser.add_argument("-o", "--output", type=str, default=None)

    args = parser.parse_args()

    if args.labels:
        predictions = parse_labels(args.labels)
        measurements = {
            "heart_rate": args.heart_rate,
            "pr_interval": args.pr_interval,
            "qrs_duration": args.qrs_duration,
            "axis": args.axis,
            "qt_interval": args.qt_interval,
            "qtc": args.qtc,
        }
        run_single(predictions, measurements, screening=args.screening)

    elif args.json:
        with open(args.json) as f:
            data = json.load(f)
        preds = data.get("predictions", {})
        meas = data.get("measurements", {})
        scores_data = data.get("scores", None)
        run_single(preds, meas, scores=scores_data, screening=args.screening)

    elif args.csv:
        if not args.label_cols:
            parser.error("--label-cols is required in CSV mode")
        run_csv(args)


if __name__ == "__main__":
    main()
