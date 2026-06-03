#!/usr/bin/env python3
"""Merge three nfer ECG CSVs, map to engine features, run the rule engine.

The three CSVs (column names as seen on the target server):

  1. PER-LEAD  (long format, one row per LEAD_NAME per ECG):
       LEAD_NAME, P_WAVE_PEAK_AMPLITUDE, Q_WAVE_PEAK_AMPLITUDE,
       R_WAVE_PEAK_AMPLITUDE, S_WAVE_PEAK_AMPLITUDE, T_WAVE_PEAK_AMPLITUDE,
       STJ_POINT_OR_QRS_END_AMPLITUDE, ST_SEGMENT_MIDDLE_AMPLITUDE,
       Q_WAVE_DURATION, R_WAVE_DURATION, S_WAVE_DURATION, ...

  2. GLOBAL  (one row per ECG, intervals/rates/axes):
       ATRIAL_RATE, P_AXIS, R_AXIS, T_AXIS, P_ONSET, P_OFFSET, PR_INTERVAL,
       QRS_DURATION, QT_INTERVAL, QTC_BAZETT, QTC_FREDERICA, QTC_FRAMINGHAM,
       VENTRICULAR_RATE, ...

  3. INTERVAL  (one row per ECG, overlaps GLOBAL, adds AVG_RR_INTERVAL):
       P_ONSET, P_OFFSET, QRS_ONSET, QRS_OFFSET, T_ONSET, T_OFFSET,
       QRS_DURATION, QT_INTERVAL, QTC_BAZETT, PR_INTERVAL, VENTRICULAR_RATE,
       AVG_RR_INTERVAL, ...

All three carry PERSON_ID, EVENT_DTM, ECG_ID, PERSON_AGE. The per-lead table
is pivoted to wide (one row per ECG) and joined to the two global tables.

IMPORTANT — units:
  Rule thresholds are in mV (amplitudes) and ms (durations/intervals). GE/nfer
  amplitudes are frequently stored in microvolts. If yours are in uV, pass
  --amp-scale 0.001 to convert. Verify a known ECG (e.g. R in V5 ~ 1-2 mV).

Usage:
  python3 scripts/run_nfer_csvs.py \
      --lead-csv per_lead.csv \
      --global-csv global.csv \
      --interval-csv interval.csv \
      --out-preds reports/nfer_predictions.csv \
      --out-traces reports/nfer_traces.json \
      --amp-scale 1.0

  # auto-detect which file is which (per-lead = has LEAD_NAME):
  python3 scripts/run_nfer_csvs.py --csv a.csv b.csv c.csv --out-preds preds.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

RULES_DIR = ROOT / "rules"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Default merge keys (user requested PERSON_ID + EVENT_DTM). ECG_ID is more
# robust if present in all three — override with --keys.
DEFAULT_KEYS = ["PERSON_ID", "EVENT_DTM"]

# ── nfer global/interval column → engine feature ───────────────────────────
GLOBAL_MAP: dict[str, str] = {
    "QRS_ms": "QRS_DURATION",
    "PR_ms": "PR_INTERVAL",
    "QT_ms": "QT_INTERVAL",
    "QTc_Bazett_ms": "QTC_BAZETT",
    "QTc_Fridericia_ms": "QTC_FREDERICA",
    "QTc_Framingham_ms": "QTC_FRAMINGHAM",
    "ventricular_rate_bpm": "VENTRICULAR_RATE",
    "atrial_rate_bpm": "ATRIAL_RATE",
    "QRS_axis_deg": "R_AXIS",
    "P_axis_deg": "P_AXIS",
    "T_axis_deg": "T_AXIS",
    "RR_ms": "AVG_RR_INTERVAL",
}

# ── per-lead column → engine feature prefix, with transform ────────────────
# transform: "clamp0" (max(.,0)), "abs", "signed"
LEAD_AMP_MAP: dict[str, tuple[str, str]] = {
    "R": ("R_WAVE_PEAK_AMPLITUDE", "clamp0"),
    "Q": ("Q_WAVE_PEAK_AMPLITUDE", "abs"),
    "S": ("S_WAVE_PEAK_AMPLITUDE", "abs"),
    "P": ("P_WAVE_PEAK_AMPLITUDE", "signed"),
    "T": ("T_WAVE_PEAK_AMPLITUDE", "signed"),
    "STJ": ("STJ_POINT_OR_QRS_END_AMPLITUDE", "signed"),
    "STM": ("ST_SEGMENT_MIDDLE_AMPLITUDE", "signed"),
}

# ── optional 2nd-pass: rule output → derived exclusion flag ────────────────
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
VARIANT_TO_FLAG: dict[tuple[str, str], str] = {
    ("IncompleteBundleBlocks", "ILBBB"): "ilbbb_flag",
    ("IncompleteBundleBlocks", "IRBBB"): "irbbb_flag",
    ("Hemiblocks", "LAFB"): "lafb_flag",
    ("Hemiblocks", "LPFB"): "lpfb_flag",
    ("Hemiblocks", "IVCD_nonspecific"): "ivcb_flag",
}

# ── engine disease → ground-truth column(s) in the 4th (GT) file ───────────
# GT is a wide one-hot table; a disease is POSITIVE if ANY mapped column is
# truthy. Edit freely (or override with --gt-map JSON). Diseases not listed
# here have no GT column and are skipped in the metrics table.
DISEASE_TO_GT: dict[str, list[str]] = {
    "SinusRhythm": ["Sinus Rhythm"],
    "AFib": ["Atrial Fibrillation"],
    "AtrialFlutter": ["Atrial Flutter"],
    "WPW": ["Wolff-Parkinson-White Syndrome"],
    "LBBB": ["Left Bundle Branch Block"],
    "RBBB": ["Right Bundle Branch Block"],
    "LVH": ["Left Ventricular Hypertrophy", "Left Ventricular Strain Pattern"],
    "RVH": ["Right Ventricular Hypertrophy"],
    "LowVoltageQRS": ["Low Voltage QRS"],
    "NonspecificIVCB": ["Non-Specific Intraventricular Conduction Delay"],
    "Pacing": ["Paced Rhythm"],
    "EctopicAtrialRhythm": ["Ectopic Atrial Rhythm"],
    "JunctionalRhythm": ["Junctional Rhythm"],
    "Brugada": ["Brugada Syndrome Pattern"],
    "PericarditisOrEarlyRepol": ["Early Repolarization Pattern"],
    "PRInterval": ["1st Degree AV Block"],
    "AVBlock": [
        "2nd Degree AV Block", "2nd Degree AV Block Mobitz I",
        "2nd Degree AV Block Mobitz II", "3rd Degree AV Block",
        "2:1 a-v conduction", "3:1 a-v conduction",
        "4:1 a-v conduction", "5:1 a-v conduction",
    ],
    "Hemiblocks": [
        "Left Anterior Fascicular Block", "Left Posterior Fascicular Block",
        "Bifascicular block",
    ],
    "AtrialEnlargement": [
        "Left Atrial Enlargement", "Right Atrial Enlargement",
        "Biatrial Enlargement",
    ],
    "QWaveMI": [
        "Inferior Myocardial Infarction", "Anterolateral Infarct (closest)",
        "Acute Anterior Myocardial Infarction",
        "Acute Inferior Myocardial Infarction",
    ],
    "AcuteMISTEMI": [
        "Acute Anterior Myocardial Infarction",
        "Acute Inferior Myocardial Infarction",
    ],
    "STElevationInjury": ["ST Elevation"],
    "NonspecificSTElevation": ["ST Elevation"],
    "STDepressionIschemia": ["ST Depression"],
    "NonspecificSTDepression": ["ST Depression"],
    "TWaveIschemia": ["T Wave Inversion"],
    "NonspecificTWave": ["T Wave Inversion"],
    "Ectopy": [
        "Premature Atrial Complex", "Premature Junctional Complexes",
        "Premature Ventricular Complex", "Bigeminy", "Trigeminy",
    ],
    "UndeterminedRhythm": ["Wide QRS Rhythm", "Supraventricular Tachycardia (SVT)"],
}


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _truthy(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, (int, float)):
        return float(val) >= 1
    s = str(val).strip().lower()
    return s in {"1", "1.0", "true", "yes", "y", "t", "positive", "present"}


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_gt_metrics(pred_df: pd.DataFrame, gt_path: Path, pred_keys: list[str],
                       gt_keys: list[str], disease_map: dict[str, list[str]],
                       out_metrics: Path) -> None:
    gt = pd.read_csv(gt_path)
    # normalized column lookup so minor spacing/case differences still match
    col_lookup = {_norm(c): c for c in gt.columns}

    # align GT keys onto prediction key names (positional)
    rename = {gk: pk for gk, pk in zip(gt_keys, pred_keys)}
    gt = gt.rename(columns=rename)
    for k in pred_keys:
        if k not in gt.columns:
            print(f"  GT ERROR: key '{k}' not in GT file after rename; "
                  f"have {list(gt.columns)[:8]}...")
            return
        gt[k] = gt[k].astype(str)

    pred = pred_df.copy()
    for k in pred_keys:
        pred[k] = pred[k].astype(str)

    merged = pred.merge(gt, on=pred_keys, how="inner", suffixes=("", "_gt"))
    print(f"\nGT match: {len(merged)} ECGs joined "
          f"(pred={len(pred)}, gt={len(gt)}) on {pred_keys}<-{gt_keys}")
    if len(merged) == 0:
        print("  No rows joined — check key names/values (--gt-keys).")
        return

    rows = []
    for disease, gt_cols in disease_map.items():
        if disease not in pred.columns:
            continue
        present = [col_lookup[_norm(c)] for c in gt_cols if _norm(c) in col_lookup]
        missing = [c for c in gt_cols if _norm(c) not in col_lookup]
        if not present:
            rows.append({"disease": disease, "gt_columns": "; ".join(gt_cols),
                         "status": "NO GT COLUMN FOUND"})
            continue

        gt_pos = merged[present].apply(lambda r: any(_truthy(v) for v in r), axis=1)
        valid = merged[merged[disease] >= 0]
        gt_pos_v = gt_pos.loc[valid.index]
        pred_pos = valid[disease] >= 1

        tp = int((pred_pos & gt_pos_v).sum())
        fp = int((pred_pos & ~gt_pos_v).sum())
        fn = int((~pred_pos & gt_pos_v).sum())
        tn = int((~pred_pos & ~gt_pos_v).sum())
        sens = safe_div(tp, tp + fn)
        spec = safe_div(tn, tn + fp)
        ppv = safe_div(tp, tp + fp)
        f1 = safe_div(2 * tp, 2 * tp + fp + fn)
        rows.append({
            "disease": disease,
            "gt_columns": "; ".join(present) + (f"  [MISSING: {missing}]" if missing else ""),
            "GT_pos": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(sens, 4), "specificity": round(spec, 4),
            "PPV": round(ppv, 4), "F1": round(f1, 4),
            "prevalence": round(safe_div(tp + fn, len(valid)), 4),
            "status": "ok",
        })

    mdf = pd.DataFrame(rows)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out_metrics, index=False)
    print(f"GT metrics  -> {out_metrics}")

    ok = mdf[mdf["status"] == "ok"].sort_values("F1", ascending=False)
    if len(ok):
        print("\n=== GT metrics (sorted by F1) ===")
        print(ok[["disease", "GT_pos", "TP", "FP", "FN",
                  "sensitivity", "specificity", "PPV", "F1"]].to_string(index=False))
    no_gt = mdf[mdf["status"] != "ok"]["disease"].tolist()
    if no_gt:
        print(f"\nNo GT column for: {', '.join(no_gt)}")


def _f(val) -> float | None:
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def normalize_lead(raw) -> str | None:
    """Map raw LEAD_NAME values to canonical engine lead names."""
    if pd.isna(raw):
        return None
    s = str(raw).strip().upper().replace("LEAD", "").strip()
    table = {
        "I": "I", "II": "II", "III": "III",
        "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
        "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
    }
    return table.get(s)


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


def pivot_lead_csv(df: pd.DataFrame, keys: list[str], amp_scale: float) -> pd.DataFrame:
    """Long per-lead table → wide (one row per ECG) with engine column names."""
    df = df.copy()
    df["_lead"] = df["LEAD_NAME"].map(normalize_lead)
    df = df[df["_lead"].notna()]

    wide_rows: dict[tuple, dict] = {}
    for _, r in df.iterrows():
        key = tuple(r.get(k) for k in keys)
        lead = r["_lead"]
        bucket = wide_rows.setdefault(key, {k: r.get(k) for k in keys})

        for prefix, (col, transform) in LEAD_AMP_MAP.items():
            v = _f(r.get(col))
            if v is None:
                continue
            if transform == "clamp0":
                v = max(v, 0.0)
            elif transform == "abs":
                v = abs(v)
            bucket[f"{prefix}_{lead}_mV"] = v * amp_scale

        qd = _f(r.get("Q_WAVE_DURATION"))
        if qd is not None:
            bucket[f"Q_{lead}_ms"] = qd
        rd = _f(r.get("R_WAVE_DURATION")) or 0.0
        sd = _f(r.get("S_WAVE_DURATION")) or 0.0
        qrs_lead = (qd or 0.0) + rd + sd
        if qrs_lead > 0:
            bucket[f"QRS_{lead}_ms"] = qrs_lead

    return pd.DataFrame(list(wide_rows.values()))


def merge_csvs(lead: pd.DataFrame, glob: pd.DataFrame,
               interval: pd.DataFrame | None, keys: list[str],
               amp_scale: float) -> pd.DataFrame:
    lead_wide = pivot_lead_csv(lead, keys, amp_scale)
    print(f"  per-lead pivoted: {len(lead_wide)} ECGs, {lead_wide.shape[1]} cols")

    # Coerce key dtypes to string for a stable join.
    def _prep(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for k in keys:
            d[k] = d[k].astype(str)
        return d

    merged = _prep(glob)
    if interval is not None:
        ivl = _prep(interval)
        # only bring columns not already present (avoid suffix collisions)
        extra = [c for c in ivl.columns if c not in merged.columns or c in keys]
        merged = merged.merge(ivl[extra], on=keys, how="outer", suffixes=("", "_ivl"))
    merged = merged.merge(_prep(lead_wide), on=keys, how="outer", suffixes=("", "_lead"))
    print(f"  merged: {len(merged)} ECGs, {merged.shape[1]} cols")
    return merged


def row_to_features(row: pd.Series, amp_scale: float) -> dict[str, float]:
    """Build engine feature dict from a merged row."""
    f: dict[str, float] = {}

    for feat_id, col in GLOBAL_MAP.items():
        v = _f(row.get(col))
        if v is not None:
            f[feat_id] = v

    # P duration derived from onset/offset if not present
    if "P_ms" not in f:
        on, off = _f(row.get("P_ONSET")), _f(row.get("P_OFFSET"))
        if on is not None and off is not None and off > on:
            f["P_ms"] = off - on

    # RR / PP fallbacks
    if "RR_ms" not in f and f.get("ventricular_rate_bpm", 0) > 0:
        f["RR_ms"] = 60000.0 / f["ventricular_rate_bpm"]
    if "PP_ms" not in f and f.get("atrial_rate_bpm", 0) > 0:
        f["PP_ms"] = 60000.0 / f["atrial_rate_bpm"]

    # Per-lead features already on the merged row (created during pivot)
    for lead in LEADS:
        for prefix in ("R", "Q", "S", "P", "T", "STJ", "STM"):
            key = f"{prefix}_{lead}_mV"
            v = _f(row.get(key))
            if v is not None:
                f[key] = v
        for suffix in ("ms",):
            for base in (f"Q_{lead}_{suffix}", f"QRS_{lead}_{suffix}"):
                v = _f(row.get(base))
                if v is not None:
                    f[base] = v

    return f


def derived_flags(results: dict) -> dict[str, float]:
    flags: dict[str, float] = {}
    for disease, flag in DISEASE_TO_FLAG.items():
        res = results.get(disease)
        if res and res.fired:
            flags[flag] = 1.0
    for (disease, variant), flag in VARIANT_TO_FLAG.items():
        res = results.get(disease)
        if res:
            for vr in res.variant_results:
                if vr.variant_name == variant and vr.fired:
                    flags[flag] = 1.0
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lead-csv", type=Path, help="Per-lead CSV (has LEAD_NAME)")
    ap.add_argument("--global-csv", type=Path, help="Global measurements CSV")
    ap.add_argument("--interval-csv", type=Path, default=None, help="Interval CSV (optional)")
    ap.add_argument("--csv", type=Path, nargs="+", default=None,
                    help="Provide the 3 CSVs and auto-detect roles")
    ap.add_argument("--keys", type=str, default=",".join(DEFAULT_KEYS),
                    help="Merge keys, comma-separated (default PERSON_ID,EVENT_DTM)")
    ap.add_argument("--amp-scale", type=float, default=1.0,
                    help="Multiply all amplitudes by this (use 0.001 for uV->mV)")
    ap.add_argument("--no-derived-flags", action="store_true",
                    help="Disable 2nd pass that derives exclusion flags from rule output")
    ap.add_argument("--diseases", type=str, default=None,
                    help="Comma-separated subset of diseases")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-preds", type=Path, default=ROOT / "reports" / "nfer_predictions.csv")
    ap.add_argument("--out-merged", type=Path, default=None,
                    help="Optional: write the merged wide table for inspection")
    ap.add_argument("--out-traces", type=Path, default=None)
    ap.add_argument("--gt-csv", type=Path, default=None,
                    help="4th file: wide one-hot ground-truth table")
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM",
                    help="GT key columns (aligned positionally to --keys)")
    ap.add_argument("--gt-map", type=Path, default=None,
                    help="Optional JSON overriding the disease->GT-column(s) map")
    ap.add_argument("--out-metrics", type=Path,
                    default=ROOT / "reports" / "nfer_gt_metrics.csv")
    args = ap.parse_args()

    keys = [k.strip() for k in args.keys.split(",")]

    # Resolve which CSV is which
    lead_path, glob_path, ivl_path = args.lead_csv, args.global_csv, args.interval_csv
    if args.csv:
        lead_path = glob_path = ivl_path = None
        globals_found = []
        for p in args.csv:
            head = pd.read_csv(p, nrows=1)
            if "LEAD_NAME" in head.columns:
                lead_path = p
            else:
                globals_found.append(p)
        # the one with AVG_RR_INTERVAL is the interval table
        for p in globals_found:
            head = pd.read_csv(p, nrows=1)
            if "AVG_RR_INTERVAL" in head.columns and ivl_path is None and glob_path is not None:
                ivl_path = p
            elif glob_path is None:
                glob_path = p
            else:
                ivl_path = p

    if not lead_path or not glob_path:
        ap.error("Need a per-lead CSV (LEAD_NAME) and a global CSV. "
                 "Pass --lead-csv/--global-csv or 3 files via --csv.")

    print(f"lead-csv     : {lead_path}")
    print(f"global-csv   : {glob_path}")
    print(f"interval-csv : {ivl_path}")
    print(f"merge keys   : {keys}")
    print(f"amp-scale    : {args.amp_scale}")

    lead_df = pd.read_csv(lead_path)
    glob_df = pd.read_csv(glob_path)
    ivl_df = pd.read_csv(ivl_path) if ivl_path else None

    merged = merge_csvs(lead_df, glob_df, ivl_df, keys, args.amp_scale)
    if args.limit:
        merged = merged.head(args.limit)
    if args.out_merged:
        merged.to_csv(args.out_merged, index=False)
        print(f"  wrote merged table -> {args.out_merged}")

    disease_filter = set(args.diseases.split(",")) if args.diseases else None
    diseases = load_diseases(disease_filter)
    print(f"Loaded {len(diseases)} rules, "
          f"{sum(len(d.variants) for d in diseases.values())} variants")

    disease_names = sorted(diseases.keys())
    pred_rows: list[dict] = []
    trace_rows: list[dict] = []
    skipped = 0
    mapped_counts: list[int] = []

    for idx, row in tqdm(merged.iterrows(), total=len(merged), unit="ecg"):
        feats = row_to_features(row, args.amp_scale)
        mapped_counts.append(len(feats))
        if "QRS_ms" not in feats:
            skipped += 1
            pred_rows.append({**{k: row.get(k) for k in keys},
                              **{dn: -1 for dn in disease_names}})
            continue

        age = None
        av = _f(row.get("PERSON_AGE"))
        if av is not None:
            age = int(av)

        # Pass 1 — measurements (+ age), no flags
        ctx = EvalContext(features=dict(feats), sex=None, age_years=age)
        results = {dn: evaluate_disease(diseases[dn], ctx) for dn in disease_names}

        # Pass 2 — re-evaluate with flags derived from pass-1 output
        if not args.no_derived_flags:
            flags = derived_flags(results)
            if flags:
                ctx2 = EvalContext(features={**feats, **flags}, sex=None, age_years=age)
                results = {dn: evaluate_disease(diseases[dn], ctx2) for dn in disease_names}

        pred_rows.append({**{k: row.get(k) for k in keys},
                          **{dn: int(results[dn].fired) for dn in disease_names}})
        if args.out_traces:
            trace_rows.append({
                **{k: row.get(k) for k in keys},
                "results": {dn: results[dn].to_dict() for dn in disease_names},
            })

    print(f"\nProcessed {len(merged)} ECGs ({skipped} skipped: no QRS_ms)")
    if mapped_counts:
        import statistics
        print(f"Features mapped per ECG: median={int(statistics.median(mapped_counts))}, "
              f"max={max(mapped_counts)} (of 122 measurements)")

    pred_df = pd.DataFrame(pred_rows)
    print("\n=== Fire-rate summary ===")
    for dn in disease_names:
        valid = pred_df[pred_df[dn] >= 0][dn]
        fires, total = int(valid.sum()), len(valid)
        rate = fires / total if total else 0
        print(f"  {dn:32s} {fires:>6d} / {total:<6d} ({rate:6.2%})")

    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(args.out_preds, index=False)
    print(f"\nPredictions -> {args.out_preds}")

    if args.out_traces and trace_rows:
        args.out_traces.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_traces, "w") as fh:
            json.dump(trace_rows, fh, indent=2, default=str)
        print(f"Traces      -> {args.out_traces}")

    # ── Ground-truth comparison (4th file) ─────────────────────────────────
    if args.gt_csv:
        disease_map = DISEASE_TO_GT
        if args.gt_map:
            with open(args.gt_map) as fh:
                disease_map = json.load(fh)
        gt_keys = [k.strip() for k in args.gt_keys.split(",")]
        compute_gt_metrics(pred_df, args.gt_csv, keys, gt_keys,
                           disease_map, args.out_metrics)


if __name__ == "__main__":
    main()
