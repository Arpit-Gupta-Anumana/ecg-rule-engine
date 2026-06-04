#!/usr/bin/env python3
"""Inspect every GT-positive case for a given abnormality.

For each ECG where the GT parquet says the abnormality is present, print
(and write to CSV) the engine's prediction plus the raw values it used
from the nfer measurement CSVs.

Examples
--------
# Atrial Flutter — show atrial_rate, age, ventricular_rate, engine call:
python3 scripts/inspect_abnormality.py \\
    --abnormality "Atrial Flutter" \\
    --predictions-in reports/nfer_per_rule_predictions.csv \\
    --global-csv     /data/sharedhdd/arpit/arpit/ecg_measurement.csv \\
    --gt-csv         /data/NFERECG/shared/supreeth.gupta/abnormality_extraction/abnormalities_cohort.parquet \\
    --gt-keys        NFER_PID,NFER_DTM \\
    --features       ATRIAL_RATE,VENTRICULAR_RATE,PERSON_AGE \\
    --out            reports/atrial_flutter_gt_positive.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from run_nfer_csvs import _norm, _truthy, load_gt_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--abnormality", required=True,
                    help='Parquet label, e.g. "Atrial Flutter"')
    ap.add_argument("--predictions-in", type=Path, required=True,
                    help="Per-ECG predictions CSV from eval_per_rule.py")
    ap.add_argument("--global-csv", type=Path, default=None,
                    help="ecg_measurement.csv (for raw feature columns)")
    ap.add_argument("--lead-csv", type=Path, default=None)
    ap.add_argument("--interval-csv", type=Path, default=None)
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--keys", type=str, default="PERSON_ID,EVENT_DTM")
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM")
    ap.add_argument("--features", type=str, default="ATRIAL_RATE",
                    help="Comma-separated raw feature columns to pull from "
                         "--global-csv (or --lead-csv/--interval-csv).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional output CSV (otherwise just prints).")
    ap.add_argument("--head", type=int, default=200,
                    help="Max rows to print to stdout.")
    args = ap.parse_args()

    pred_keys = [k.strip() for k in args.keys.split(",")]
    gt_keys = [k.strip() for k in args.gt_keys.split(",")]
    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]

    # ── load predictions ────────────────────────────────────────────
    print(f"Loading predictions from {args.predictions_in}...")
    pred = pd.read_csv(args.predictions_in)
    for k in pred_keys:
        pred[k] = pred[k].astype(str)
    if args.abnormality not in pred.columns:
        raise SystemExit(
            f"Abnormality column {args.abnormality!r} not in predictions CSV."
        )
    supp_col = f"{args.abnormality}__supp"
    keep_cols = pred_keys + [args.abnormality]
    if supp_col in pred.columns:
        keep_cols.append(supp_col)
    pred = pred[keep_cols].rename(columns={
        args.abnormality: "engine_pred",
        supp_col: "engine_pred_post_supp",
    })

    # ── load GT ────────────────────────────────────────────────────
    print(f"Loading GT from {args.gt_csv}...")
    gt = load_gt_table(args.gt_csv)
    gt = gt.rename(columns={gk: pk for gk, pk in zip(gt_keys, pred_keys)})
    for k in pred_keys:
        gt[k] = gt[k].astype(str)
    col_lookup = {_norm(c): c for c in gt.columns}
    gt_col = col_lookup.get(_norm(args.abnormality))
    if gt_col is None:
        raise SystemExit(f"No GT column matches {args.abnormality!r}")
    gt = gt[pred_keys + [gt_col]].rename(columns={gt_col: "gt"})
    gt["gt"] = gt["gt"].apply(lambda v: int(_truthy(v)))

    # ── join ───────────────────────────────────────────────────────
    df = pred.merge(gt, on=pred_keys, how="inner")
    print(f"Joined predictions ↔ GT on {pred_keys}: {len(df)} ECGs")
    pos = df[df["gt"] == 1].copy()
    print(f"GT-positive {args.abnormality!r}: {len(pos)} ECGs in this cohort")
    if pos.empty:
        return

    # ── pull raw feature columns from source CSVs ──────────────────
    raw_frames = []
    for path, label in [(args.global_csv, "global"),
                         (args.lead_csv, "lead"),
                         (args.interval_csv, "interval")]:
        if path is None:
            continue
        src = pd.read_csv(path)
        for k in pred_keys:
            if k not in src.columns:
                print(f"  WARN {label}: missing key {k!r}, skipping merge")
                break
        else:
            for k in pred_keys:
                src[k] = src[k].astype(str)
            have = [c for c in feat_cols if c in src.columns]
            if not have:
                print(f"  {label}: no requested feature columns present")
                continue
            print(f"  {label}: pulling {have}")
            grp = src[pred_keys + have].drop_duplicates(subset=pred_keys)
            raw_frames.append(grp)

    for rf in raw_frames:
        pos = pos.merge(rf, on=pred_keys, how="left")

    # ── summarise engine outcome ───────────────────────────────────
    pos["engine_pred"] = pos["engine_pred"].fillna(0).astype(int)
    if "engine_pred_post_supp" in pos.columns:
        pos["engine_pred_post_supp"] = (
            pos["engine_pred_post_supp"].fillna(0).astype(int)
        )
    n = len(pos)
    tp_raw = int((pos["engine_pred"] == 1).sum())
    print(f"\nEngine called {args.abnormality!r}: {tp_raw}/{n} of GT-positive "
          f"({tp_raw / n:.1%})")
    if "engine_pred_post_supp" in pos.columns:
        tp_supp = int((pos["engine_pred_post_supp"] == 1).sum())
        print(f"After suppression:             {tp_supp}/{n} "
              f"({tp_supp / n:.1%})")

    # ── show & save ───────────────────────────────────────────────
    show_cols = pred_keys + ["engine_pred"]
    if "engine_pred_post_supp" in pos.columns:
        show_cols.append("engine_pred_post_supp")
    show_cols += [c for c in feat_cols if c in pos.columns]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", args.head)
    print(f"\n=== First {min(args.head, n)} GT-positive ECGs ===")
    print(pos[show_cols].head(args.head).to_string(index=False))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pos[show_cols].to_csv(args.out, index=False)
        print(f"\nWrote {len(pos)} rows -> {args.out}")


if __name__ == "__main__":
    main()
