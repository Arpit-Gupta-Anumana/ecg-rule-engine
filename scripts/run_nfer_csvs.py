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

# All 122 measurement inputs the rule engine can use (excl. flags, sex, age).
REQUIRED_FEATURE_IDS: frozenset[str] = frozenset(
    {
        "ventricular_rate_bpm", "atrial_rate_bpm", "RR_ms", "PP_ms",
        "QRS_ms", "PR_ms", "QT_ms", "QTc_Bazett_ms", "QTc_Fridericia_ms",
        "QTc_Framingham_ms", "P_ms", "QRS_axis_deg", "P_axis_deg", "T_axis_deg",
        *[f"{w}_{lead}_mV" for w in ("R", "Q", "S", "P", "T", "STJ", "STM") for lead in LEADS],
        *[f"Q_{lead}_ms" for lead in LEADS],
        *[f"QRS_{lead}_ms" for lead in LEADS],
    }
)
assert len(REQUIRED_FEATURE_IDS) == 122

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

# Non-label columns in the abnormalities cohort file
GT_META_COLS = frozenset({
    "NFER_PID", "NFER_DTM", "ECG_INTERPRET_FULL_TEXT", "LLM_OUTPUT", "parsed", "merged",
})

# Your GT abnormality column → rule-engine prediction(s).
# Each value is "Disease" (any variant) or "Disease.Variant" (specific variant).
# Prediction is positive if ANY listed rule fires. Override with --gt-map JSON.
ABNORMALITY_TO_PRED: dict[str, list[str]] = {
    "Sinus Rhythm": ["SinusRhythm"],
    "1st Degree AV Block": ["PRInterval.FirstDegreeAVBlock"],
    "2:1 a-v conduction": ["AVBlock"],
    "2nd Degree AV Block": ["AVBlock"],
    "2nd Degree AV Block Mobitz I": ["AVBlock"],
    "2nd Degree AV Block Mobitz II": ["AVBlock"],
    "3:1 a-v conduction": ["AVBlock"],
    "3rd Degree AV Block": ["AVBlock.CompleteHeartBlock"],
    "4:1 a-v conduction": ["AVBlock"],
    "5:1 a-v conduction": ["AVBlock"],
    "Atrial Fibrillation": ["AFib"],
    "Atrial Flutter": ["AtrialFlutter"],
    "Junctional Rhythm": ["JunctionalRhythm"],
    "Ectopic Atrial Rhythm": ["EctopicAtrialRhythm"],
    "Paced Rhythm": ["Pacing"],
    "Wide QRS Rhythm": ["UndeterminedRhythm.WideQRSRhythm"],
    "Supraventricular Tachycardia (SVT)": ["UndeterminedRhythm.SVT"],
    "Wolff-Parkinson-White Syndrome": ["WPW"],
    "Left Bundle Branch Block": ["LBBB"],
    "Right Bundle Branch Block": ["RBBB"],
    "Left Anterior Fascicular Block": ["Hemiblocks.LAFB"],
    "Left Posterior Fascicular Block": ["Hemiblocks.LPFB"],
    "Bifascicular block": ["Hemiblocks.LAFB", "Hemiblocks.LPFB"],
    "Non-Specific Intraventricular Conduction Delay": ["NonspecificIVCB"],
    "Left Ventricular Hypertrophy": ["LVH"],
    "Left Ventricular Strain Pattern": ["LVH"],
    "Right Ventricular Hypertrophy": ["RVH"],
    "Low Voltage QRS": ["LowVoltageQRS"],
    "Left Atrial Enlargement": ["AtrialEnlargement.LAE_literature_proxy"],
    "Right Atrial Enlargement": ["AtrialEnlargement.RAE"],
    "Biatrial Enlargement": ["AtrialEnlargement"],
    "Brugada Syndrome Pattern": ["Brugada"],
    "Early Repolarization Pattern": ["PericarditisOrEarlyRepol.Early_Repolarization"],
    "Acute Anterior Myocardial Infarction": ["AcuteMISTEMI", "QWaveMI.Anterior_MI"],
    "Acute Inferior Myocardial Infarction": ["AcuteMISTEMI", "QWaveMI.Inferior_MI"],
    "Inferior Myocardial Infarction": ["QWaveMI.Inferior_MI"],
    "Anterolateral Infarct (closest)": ["QWaveMI.Lateral_MI"],
    "ST Elevation": ["STElevationInjury", "NonspecificSTElevation"],
    "ST Depression": ["STDepressionIschemia", "NonspecificSTDepression"],
    "T Wave Inversion": ["TWaveIschemia", "NonspecificTWave"],
    "Premature Atrial Complex": ["Ectopy"],
    "Premature Junctional Complexes": ["Ectopy"],
    "Premature Ventricular Complex": ["Ectopy"],
    "Bigeminy": ["Ectopy"],
    "Trigeminy": ["Ectopy"],
}

# Coarse GT families: combined GT prevalence vs combined engine fire (any label in group)
COARSE_GROUPS: list[tuple[str, list[str], list[str]]] = [
    ("AV block (9 labels)", [
        "1st Degree AV Block", "2:1 a-v conduction", "2nd Degree AV Block",
        "2nd Degree AV Block Mobitz I", "2nd Degree AV Block Mobitz II",
        "3:1 a-v conduction", "3rd Degree AV Block", "4:1 a-v conduction",
        "5:1 a-v conduction",
    ], ["AVBlock"]),
    ("Ectopy (5 labels)", [
        "Premature Atrial Complex", "Premature Junctional Complexes",
        "Premature Ventricular Complex", "Bigeminy", "Trigeminy",
    ], ["Ectopy"]),
    ("ST elevation", ["ST Elevation"], ["STElevationInjury", "NonspecificSTElevation"]),
    ("ST depression", ["ST Depression"], ["STDepressionIschemia", "NonspecificSTDepression"]),
    ("T wave inversion", ["T Wave Inversion"], ["TWaveIschemia", "NonspecificTWave"]),
    ("Acute MI (2 labels)", [
        "Acute Anterior Myocardial Infarction", "Acute Inferior Myocardial Infarction",
    ], ["AcuteMISTEMI"]),
]

# Parquet label → suppression_rules.yaml label (for --apply-suppression)
ABNORMALITY_TO_SUPPRESSION: dict[str, str] = {
    "Sinus Rhythm": "Sinus",
    "Atrial Fibrillation": "Afib",
    "Atrial Flutter": "Flutter",
    "Junctional Rhythm": "Junctional",
    "Ectopic Atrial Rhythm": "Ectropic Atrial",
    "Paced Rhythm": "Paced",
    "Wide QRS Rhythm": "Wide QRS",
    "Supraventricular Tachycardia (SVT)": "SVT",
    "Wolff-Parkinson-White Syndrome": "WPW",
    "Left Bundle Branch Block": "Left Bundle Branch Block",
    "Right Bundle Branch Block": "Right Bundle Branch Block",
    "Left Anterior Fascicular Block": "Left Anterior Fascicular Block",
    "Left Posterior Fascicular Block": "Left Posterior Fascicular Block",
    "Bifascicular block": "Bifascicular block",
    "Non-Specific Intraventricular Conduction Delay": "NICD",
    "Left Ventricular Hypertrophy": "Left Ventricular Hypertrophy",
    "Left Ventricular Strain Pattern": "Left Ventricular Hypertrophy",
    "Right Ventricular Hypertrophy": "Right Ventricular Hypertrophy",
    "Low Voltage QRS": "Low Voltage QRS",
    "Left Atrial Enlargement": "Left Atrial Enlargement",
    "Right Atrial Enlargement": "Right Atrial Enlargement",
    "Biatrial Enlargement": "Biatrial Enlargement",
    "Brugada Syndrome Pattern": "Brugada Syndrome Pattern",
    "Early Repolarization Pattern": "Early Repolarization Pattern",
    "ST Elevation": "ST_ST Elevation",
    "ST Depression": "ST_ST Depression",
    "T Wave Inversion": "T Wave Inversion",
    "1st Degree AV Block": "AV_BLOCK_1st Degree AV Block",
    "2nd Degree AV Block Mobitz I": "AV_BLOCK_2nd Degree AV Block Mobitz I",
    "3rd Degree AV Block": "AV_BLOCK_3rd Degree AV Block",
    "2:1 a-v conduction": "AV_BLOCK_2:1 a-v conduction",
    "3:1 a-v conduction": "AV_BLOCK_3:1 a-v conduction",
    "4:1 a-v conduction": "AV_BLOCK_4:1 a-v conduction",
    "5:1 a-v conduction": "AV_BLOCK_5:1 a-v conduction",
    "Acute Anterior Myocardial Infarction": "Acute MI Anterior",
    "Acute Inferior Myocardial Infarction": "Acute MI Inferior",
    "Inferior Myocardial Infarction": "Old MI",
    "Anterolateral Infarct (closest)": "Old MI",
}


def diseases_for_cohort(abnormality_map: dict[str, list[str]]) -> set[str]:
    """Minimal engine diseases to evaluate (44-label cohort only)."""
    needed: set[str] = set()
    for specs in abnormality_map.values():
        for spec in specs:
            needed.add(spec.split(".", 1)[0])
    needed.update(DISEASE_TO_FLAG.keys())
    needed.update(d for d, _ in VARIANT_TO_FLAG)
    return needed


def pred_col_name(spec: str) -> str:
    """Disease or Disease.Variant → column name in predictions CSV."""
    if "." in spec:
        disease, variant = spec.split(".", 1)
        return f"pred__{disease}__{variant}"
    return spec


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


def load_gt_table(gt_path: Path) -> pd.DataFrame:
    suffix = gt_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(gt_path)
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(gt_path, sep="\t" if suffix == ".tsv" else ",")
    raise ValueError(f"Unsupported GT format: {gt_path} (use .csv, .tsv, or .parquet)")


def resolve_gt_col(merged: pd.DataFrame, gt_col: str) -> str | None:
    """Column name for a GT label after pred/GT merge (GT side may be suffixed _gt)."""
    if f"{gt_col}_gt" in merged.columns:
        return f"{gt_col}_gt"
    if gt_col in merged.columns:
        return gt_col
    return None


def _prediction_positive(merged: pd.DataFrame, specs: list[str],
                        abnormality: str | None = None) -> pd.Series:
    """True if abnormality column or any mapped rule column fired."""
    if abnormality and abnormality in merged.columns:
        return merged[abnormality].apply(lambda v: pd.notna(v) and v >= 1)
    cols = [pred_col_name(s) for s in specs if pred_col_name(s) in merged.columns]
    if not cols:
        return pd.Series(False, index=merged.index)
    return merged[cols].apply(
        lambda row: any(v >= 1 for v in row if pd.notna(v)), axis=1
    )


def abnormality_preds_from_results(
    results: dict, abnormality_map: dict[str, list[str]]
) -> dict[str, int]:
    """One 0/1 per parquet abnormality from rule-engine results."""
    out: dict[str, int] = {}
    for abnormality, specs in abnormality_map.items():
        fired = False
        for spec in specs:
            if "." in spec:
                disease, variant = spec.split(".", 1)
                res = results.get(disease)
                if res:
                    for vr in res.variant_results:
                        if vr.variant_name == variant and vr.fired:
                            fired = True
                            break
            else:
                res = results.get(spec)
                if res and res.fired:
                    fired = True
            if fired:
                break
        out[abnormality] = int(fired)
    return out


def measurements_for_suppression(feats: dict[str, float]) -> dict[str, float | None]:
    return {
        "heart_rate": feats.get("ventricular_rate_bpm"),
        "pr_interval": feats.get("PR_ms"),
        "qrs_duration": feats.get("QRS_ms"),
        "axis": feats.get("QRS_axis_deg"),
        "qt_interval": feats.get("QT_ms"),
        "qtc": feats.get("QTc_Bazett_ms"),
    }


def _load_suppression_labels() -> list[str]:
    import importlib.util
    mod_path = ROOT / "scripts" / "apply_suppression.py"
    spec = importlib.util.spec_from_file_location("apply_suppression", mod_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return list(mod.ALL_LABELS)


def apply_suppression_to_abnormalities(
    abn_preds: dict[str, int], feats: dict[str, float]
) -> dict[str, int]:
    """Run suppression_rules.yaml; return 0/1 per parquet abnormality."""
    sys.path.insert(0, str(ROOT))
    from suppression.engine import apply_suppression

    sup_in = {label: 0 for label in _load_suppression_labels()}
    for abnormality, val in abn_preds.items():
        if not val:
            continue
        slabel = ABNORMALITY_TO_SUPPRESSION.get(abnormality, abnormality)
        if slabel in sup_in:
            sup_in[slabel] = 1
    result = apply_suppression(sup_in, measurements_for_suppression(feats))
    active = result.get("active_raw", {})
    out = dict(abn_preds)
    for abnormality in abn_preds:
        slabel = ABNORMALITY_TO_SUPPRESSION.get(abnormality, abnormality)
        out[abnormality] = int(active.get(slabel, abn_preds[abnormality]))
    return out


def print_abnormality_fire_summary(
    pred_df: pd.DataFrame,
    abnormality_map: dict[str, list[str]],
    *,
    gt_path: Path | None = None,
    pred_keys: list[str] | None = None,
    gt_keys: list[str] | None = None,
    use_supp_suffix: str = "",
) -> None:
    """Fire rates for the 44 parquet labels only (+ coarse groups)."""
    n = len(pred_df)
    col_suffix = use_supp_suffix
    print(f"\n=== Abnormality fire summary ({n} ECGs, parquet labels only) ===")
    gt_merged = None
    if gt_path and pred_keys and gt_keys:
        gt = load_gt_table(gt_path)
        col_lookup = {_norm(c): c for c in gt.columns}
        rename = {gk: pk for gk, pk in zip(gt_keys, pred_keys)}
        gt = gt.rename(columns=rename)
        for k in pred_keys:
            gt[k] = gt[k].astype(str)
        p = pred_df.copy()
        for k in pred_keys:
            p[k] = p[k].astype(str)
        gt_merged = p.merge(gt, on=pred_keys, how="inner", suffixes=("", "_gt"))

    for abnormality in abnormality_map:
        col = f"{abnormality}{col_suffix}" if col_suffix else abnormality
        if col not in pred_df.columns:
            continue
        fires = int((pred_df[col] >= 1).sum())
        gt_n = ""
        if gt_merged is not None:
            ncol = col_lookup.get(_norm(abnormality))
            gtc = resolve_gt_col(gt_merged, ncol) if ncol else None
            if gtc:
                gt_n = f"  GT+={int(gt_merged[gtc].apply(_truthy).sum())}"
        print(f"  {abnormality:42s}  pred {fires:>5d}/{n}{gt_n}")

    print("\n=== Coarse groups (combined GT+ vs combined engine fire) ===")
    for group_name, gt_labels, _engine_specs in COARSE_GROUPS:
        pred_cols = [
            (f"{a}{col_suffix}" if col_suffix else a)
            for a in gt_labels if (f"{a}{col_suffix}" if col_suffix else a) in pred_df.columns
        ]
        if pred_cols:
            eng_fires = int(pred_df[pred_cols].apply(
                lambda r: any(v >= 1 for v in r if pd.notna(v)), axis=1
            ).sum())
        else:
            eng_fires = 0
        if gt_merged is not None:
            gcols = []
            for label in gt_labels:
                ncol = col_lookup.get(_norm(label))
                if ncol:
                    gtc = resolve_gt_col(gt_merged, ncol)
                    if gtc:
                        gcols.append(gtc)
            if gcols:
                gt_fires = int(gt_merged[gcols].apply(
                    lambda r: any(_truthy(v) for v in r), axis=1
                ).sum())
                print(f"  {group_name:42s}  pred {eng_fires:>5d}/{n}  GT+={gt_fires:>5d}")
                continue
        print(f"  {group_name:42s}  pred {eng_fires:>5d}/{n}")


def compute_abnormality_metrics(
    pred_df: pd.DataFrame,
    gt_path: Path,
    pred_keys: list[str],
    gt_keys: list[str],
    abnormality_map: dict[str, list[str]],
    out_metrics: Path,
) -> None:
    """Score each GT abnormality column against mapped rule-engine outputs."""
    gt = load_gt_table(gt_path)
    col_lookup = {_norm(c): c for c in gt.columns}

    rename = {gk: pk for gk, pk in zip(gt_keys, pred_keys)}
    gt = gt.rename(columns=rename)
    for k in pred_keys:
        if k not in gt.columns:
            print(f"  GT ERROR: key '{k}' not in GT file after rename")
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

    # All label columns in file, or only those in the map
    label_cols_in_gt = [
        col_lookup[_norm(c)]
        for c in abnormality_map
        if _norm(c) in col_lookup
    ]
    unmapped_gt = [
        c for c in gt.columns
        if c not in GT_META_COLS and c not in pred_keys and _norm(c) not in {_norm(x) for x in abnormality_map}
    ]

    rows = []
    for abnormality, specs in abnormality_map.items():
        ncol = col_lookup.get(_norm(abnormality))
        if ncol is None:
            rows.append({
                "abnormality": abnormality, "rule_predictions": "; ".join(specs),
                "status": "NO GT COLUMN",
            })
            continue
        pred_col = f"{abnormality}_supp" if f"{abnormality}_supp" in merged.columns else abnormality
        if pred_col not in merged.columns and not any(
            pred_col_name(s) in merged.columns for s in specs
        ):
            rows.append({
                "abnormality": abnormality, "rule_predictions": "; ".join(specs),
                "status": "NO PRED COLUMN",
            })
            continue

        gtc = resolve_gt_col(merged, ncol)
        if not gtc:
            rows.append({
                "abnormality": abnormality, "rule_predictions": "; ".join(specs),
                "status": "NO GT COLUMN IN MERGE",
            })
            continue
        gt_pos = merged[gtc].apply(_truthy)
        pred_pos = _prediction_positive(merged, specs, abnormality=pred_col)
        tp = int((pred_pos & gt_pos).sum())
        fp = int((pred_pos & ~gt_pos).sum())
        fn = int((~pred_pos & gt_pos).sum())
        tn = int((~pred_pos & ~gt_pos).sum())
        n = len(merged)
        rows.append({
            "abnormality": abnormality,
            "rule_predictions": "; ".join(specs),
            "GT_pos": tp + fn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(safe_div(tp, tp + fn), 4),
            "specificity": round(safe_div(tn, tn + fp), 4),
            "PPV": round(safe_div(tp, tp + fp), 4),
            "F1": round(safe_div(2 * tp, 2 * tp + fp + fn), 4),
            "prevalence": round(safe_div(tp + fn, n), 4),
            "status": "ok",
        })

    mdf = pd.DataFrame(rows)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    mdf.to_csv(out_metrics, index=False)
    print(f"Abnormality metrics -> {out_metrics}")

    ok = mdf[mdf["status"] == "ok"].sort_values("F1", ascending=False)
    if len(ok):
        print("\n=== Abnormality metrics (your GT labels, sorted by F1) ===")
        print(ok[["abnormality", "rule_predictions", "GT_pos", "TP", "FP", "FN",
                  "sensitivity", "PPV", "F1"]].to_string(index=False))
    bad = mdf[mdf["status"] != "ok"]
    if len(bad):
        print("\nSkipped:", bad[["abnormality", "status"]].to_string(index=False))
    if unmapped_gt:
        print(f"\nGT columns not in map ({len(unmapped_gt)}): add to ABNORMALITY_TO_PRED if needed")


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


def pivot_lead_csv(
    df: pd.DataFrame, keys: list[str], amp_scale: float, *, min_leads: int = 1
) -> pd.DataFrame:
    """Long per-lead table → wide (one row per ECG) with engine column names."""
    df = df.copy()
    df["_lead"] = df["LEAD_NAME"].map(normalize_lead)
    df = df[df["_lead"].notna()]

    wide_rows: dict[tuple, dict] = {}
    lead_counts: dict[tuple, set[str]] = {}
    for _, r in df.iterrows():
        key = tuple(r.get(k) for k in keys)
        lead = r["_lead"]
        bucket = wide_rows.setdefault(key, {k: r.get(k) for k in keys})
        lead_counts.setdefault(key, set()).add(lead)

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

    rows = []
    for key, bucket in wide_rows.items():
        if len(lead_counts.get(key, set())) >= min_leads:
            rows.append(bucket)
    return pd.DataFrame(rows)


def features_complete(feats: dict[str, float]) -> tuple[bool, list[str]]:
    missing = sorted(REQUIRED_FEATURE_IDS - feats.keys())
    return (len(missing) == 0, missing)


def filter_complete_rows(merged: pd.DataFrame, amp_scale: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only rows with all 122 mapped measurement features."""
    keep_idx: list = []
    audit: list[dict] = []
    for idx, row in merged.iterrows():
        feats = row_to_features(row, amp_scale)
        ok, missing = features_complete(feats)
        audit.append({
            "row_index": idx,
            "n_features": len(feats),
            "complete": ok,
            "n_missing": len(missing),
            "missing_sample": "; ".join(missing[:8]) + ("..." if len(missing) > 8 else ""),
        })
        if ok:
            keep_idx.append(idx)
    audit_df = pd.DataFrame(audit)
    return merged.loc[keep_idx].copy(), audit_df


def merge_csvs(
    lead: pd.DataFrame,
    glob: pd.DataFrame,
    interval: pd.DataFrame | None,
    keys: list[str],
    amp_scale: float,
    *,
    complete_only: bool = False,
) -> pd.DataFrame:
    min_leads = 12 if complete_only else 1
    lead_wide = pivot_lead_csv(lead, keys, amp_scale, min_leads=min_leads)
    print(f"  per-lead pivoted (≥{min_leads} leads): {len(lead_wide)} ECGs, "
          f"{lead_wide.shape[1]} cols")

    def _prep(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy()
        for k in keys:
            d[k] = d[k].astype(str)
        return d

    join = "inner" if complete_only else "outer"
    merged = _prep(glob)
    if interval is not None:
        ivl = _prep(interval)
        extra = [c for c in ivl.columns if c not in merged.columns or c in keys]
        merged = merged.merge(ivl[extra], on=keys, how=join, suffixes=("", "_ivl"))
    merged = merged.merge(_prep(lead_wide), on=keys, how=join, suffixes=("", "_lead"))
    print(f"  merged ({join}): {len(merged)} ECGs, {merged.shape[1]} cols")
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
                    help="Comma-separated subset of diseases (default: cohort 44 only)")
    ap.add_argument("--all-diseases", action="store_true",
                    help="Evaluate all 56 YAML diseases (not recommended)")
    ap.add_argument("--apply-suppression", action="store_true",
                    help="Post-process with suppression_rules.yaml (helps ST/T/shared labels)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-preds", type=Path, default=ROOT / "reports" / "nfer_predictions.csv")
    ap.add_argument("--out-merged", type=Path, default=None,
                    help="Optional: write the merged wide table for inspection")
    ap.add_argument("--out-traces", type=Path, default=None)
    ap.add_argument("--gt-csv", type=Path, default=None,
                    help="4th file: wide one-hot GT (.csv, .tsv, or .parquet)")
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM",
                    help="GT key columns (aligned positionally to --keys)")
    ap.add_argument("--gt-map", type=Path, default=None,
                    help="Optional JSON overriding the disease->GT-column(s) map")
    ap.add_argument("--out-metrics", type=Path,
                    default=ROOT / "reports" / "nfer_abnormality_metrics.csv")
    ap.add_argument(
        "--complete-only",
        action="store_true",
        help="Only ECGs with all 122 measurement features and 12 leads "
             "(inner-join tables, then drop any row still missing a feature)",
    )
    ap.add_argument(
        "--out-complete-audit",
        type=Path,
        default=None,
        help="With --complete-only: CSV listing n_features / missing per row",
    )
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
    if args.complete_only:
        print("complete-only: ON (12 leads + all 122 measurement features)")

    lead_df = pd.read_csv(lead_path)
    glob_df = pd.read_csv(glob_path)
    ivl_df = pd.read_csv(ivl_path) if ivl_path else None

    merged = merge_csvs(
        lead_df, glob_df, ivl_df, keys, args.amp_scale,
        complete_only=args.complete_only,
    )
    if args.complete_only:
        n_before = len(merged)
        merged, audit_df = filter_complete_rows(merged, args.amp_scale)
        n_complete = len(merged)
        print(f"  complete cohort: {n_complete} / {n_before} ECGs "
              f"({100 * n_complete / n_before:.1f}%)" if n_before else
              f"  complete cohort: {n_complete} ECGs")
        if args.out_complete_audit:
            args.out_complete_audit.parent.mkdir(parents=True, exist_ok=True)
            audit_df.to_csv(args.out_complete_audit, index=False)
            print(f"  completeness audit -> {args.out_complete_audit}")
        if n_complete == 0:
            sys.exit("No ECGs with all 122 features. Relax filters or check data.")
    if args.limit:
        merged = merged.head(args.limit)
    if args.out_merged:
        merged.to_csv(args.out_merged, index=False)
        print(f"  wrote merged table -> {args.out_merged}")

    abnormality_map = ABNORMALITY_TO_PRED
    if args.diseases:
        disease_filter = set(args.diseases.split(","))
    elif args.all_diseases:
        disease_filter = None
    else:
        disease_filter = diseases_for_cohort(abnormality_map)
    diseases = load_diseases(disease_filter)
    n_need = len(disease_filter) if disease_filter else "all"
    print(f"Loaded {len(diseases)} rule files (target diseases: {n_need}), "
          f"{sum(len(d.variants) for d in diseases.values())} variants")

    disease_names = sorted(diseases.keys())
    abnormality_names = list(abnormality_map.keys())
    print(f"Output: {len(abnormality_names)} parquet abnormality columns"
          + (" + _supp after suppression" if args.apply_suppression else ""))
    pred_rows: list[dict] = []
    trace_rows: list[dict] = []
    skipped = 0
    mapped_counts: list[int] = []

    for idx, row in tqdm(merged.iterrows(), total=len(merged), unit="ecg"):
        feats = row_to_features(row, args.amp_scale)
        mapped_counts.append(len(feats))
        if args.complete_only:
            ok, _ = features_complete(feats)
            if not ok:
                skipped += 1
                continue
        elif "QRS_ms" not in feats:
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

        row_out = {**{k: row.get(k) for k in keys}}
        abn = abnormality_preds_from_results(results, abnormality_map)
        row_out.update(abn)
        if args.apply_suppression:
            abn_supp = apply_suppression_to_abnormalities(abn, feats)
            for ab, v in abn_supp.items():
                row_out[f"{ab}_supp"] = v
        pred_rows.append(row_out)
        if args.out_traces:
            trace_rows.append({
                **{k: row.get(k) for k in keys},
                "results": {dn: results[dn].to_dict() for dn in disease_names},
            })

    skip_msg = "incomplete features" if args.complete_only else "no QRS_ms"
    print(f"\nEvaluated {len(pred_rows)} ECGs ({skipped} skipped: {skip_msg})")
    if mapped_counts:
        import statistics
        print(f"Features mapped per ECG: median={int(statistics.median(mapped_counts))}, "
              f"max={max(mapped_counts)} (required: 122)")

    pred_df = pd.DataFrame(pred_rows)
    gt_keys_list = [k.strip() for k in args.gt_keys.split(",")] if args.gt_csv else None
    print_abnormality_fire_summary(
        pred_df, abnormality_map,
        gt_path=args.gt_csv, pred_keys=keys, gt_keys=gt_keys_list,
    )
    if args.apply_suppression:
        print_abnormality_fire_summary(
            pred_df, abnormality_map,
            gt_path=args.gt_csv, pred_keys=keys, gt_keys=gt_keys_list,
            use_supp_suffix="_supp",
        )

    out_cols = list(keys) + abnormality_names
    if args.apply_suppression:
        out_cols += [f"{a}_supp" for a in abnormality_names]
    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    pred_df[out_cols].to_csv(args.out_preds, index=False)
    print(f"\nPredictions -> {args.out_preds}  ({len(abnormality_names)} labels)")

    if args.out_traces and trace_rows:
        args.out_traces.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_traces, "w") as fh:
            json.dump(trace_rows, fh, indent=2, default=str)
        print(f"Traces      -> {args.out_traces}")

    # ── Ground-truth comparison (4th file) ─────────────────────────────────
    if args.gt_csv:
        if args.gt_map:
            with open(args.gt_map) as fh:
                abnormality_map = json.load(fh)
        gt_keys = [k.strip() for k in args.gt_keys.split(",")]
        metrics_df = pred_df
        if args.apply_suppression:
            supp_cols = {a: f"{a}_supp" for a in abnormality_names if f"{a}_supp" in pred_df.columns}
            tmp = pred_df.copy()
            for ab, sc in supp_cols.items():
                tmp[ab] = tmp[sc]
            metrics_df = tmp
            print("\nMetrics use suppression-adjusted labels (_supp columns).")
        compute_abnormality_metrics(
            metrics_df, args.gt_csv, keys, gt_keys, abnormality_map, args.out_metrics
        )


if __name__ == "__main__":
    main()
