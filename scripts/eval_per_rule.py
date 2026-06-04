#!/usr/bin/env python3
"""Per-rule evaluation against the nfer parquet GT, with optional suppression.

For each (abnormality in parquet, rule mapped to it) pair:

  1. Parse the rule's clauses, collect the EXACT features it touches.
  2. Filter the merged (3-CSV) cohort to ECGs where those features are present.
  3. Evaluate just that rule on the per-rule cohort.
  4. Score against the matching GT column from the abnormalities parquet.

When ``--apply-suppression`` is set the per-spec fires are also collapsed into
per-abnormality predictions, fed through ``suppression/engine.apply_suppression``
together with the ECG's measurements (HR, PR, QRS, axis), and scored a second
time so you can compare pre- vs post-suppression metrics side by side.

Outputs
-------
--out                    per-rule metrics (one row per (abnormality, rule))
--out-abn-metrics        per-abnormality metrics (pre and, if requested, post)
--out-predictions        per-ECG predictions (raw + post-suppression columns)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from ecg_rule_engine.dsl.expr import parse_expr
from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.dsl.schema import (
    AllOfClause, AnyOfClause, NotClause, PointScoreClause, ThresholdClause,
)
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

from run_nfer_csvs import (
    ABNORMALITY_TO_PRED,
    ABNORMALITY_TO_SUPPRESSION,
    DISEASE_TO_FLAG,
    VARIANT_TO_FLAG,
    _norm,
    _truthy,
    load_gt_table,
    merge_csvs,
    row_to_features,
    safe_div,
)

# Features the engine derives or auto-fills (NOT required from the CSV).
DERIVED_OR_OPTIONAL: set[str] = {
    *DISEASE_TO_FLAG.values(),
    *VARIANT_TO_FLAG.values(),
    "sex_M_flag", "sex_F_flag",
    "PP_ms",
}

RULES_DIR = ROOT / "rules"

# Measurement columns that the suppression engine consumes; these get baked
# into the per-ECG predictions CSV so it can be replayed via --predictions-in.
MEAS_COLS = ["heart_rate", "pr_interval", "qrs_duration", "axis",
             "qt_interval", "qtc"]


# ── clause walking ─────────────────────────────────────────────────────────

def features_in_clause(clause) -> set[str]:
    out: set[str] = set()
    if isinstance(clause, ThresholdClause):
        out |= parse_expr(clause.expr).features()
    elif isinstance(clause, (AllOfClause, AnyOfClause)):
        for c in clause.clauses:
            out |= features_in_clause(c)
    elif isinstance(clause, NotClause):
        out |= features_in_clause(clause.clause)
    elif isinstance(clause, PointScoreClause):
        for item in clause.items:
            out |= features_in_clause(item.when)
    return out


def required_features_for_spec(spec: str, diseases: dict) -> set[str]:
    if "." in spec:
        disease_name, variant_name = spec.split(".", 1)
    else:
        disease_name, variant_name = spec, None
    disease = diseases.get(disease_name)
    if disease is None:
        return set()
    feats: set[str] = set()
    for v in disease.variants:
        if variant_name is None or v.name == variant_name:
            feats |= features_in_clause(v.clause)
    return {f for f in feats if f not in DERIVED_OR_OPTIONAL}


# ── per-rule eval ──────────────────────────────────────────────────────────

def evaluate_one_rule(
    spec: str,
    diseases: dict,
    merged: pd.DataFrame,
    amp_scale: float,
    keys: list[str],
) -> tuple[pd.DataFrame, set[str], int]:
    required = required_features_for_spec(spec, diseases)
    if not required:
        return pd.DataFrame(), required, 0

    if "." in spec:
        disease_name, variant_name = spec.split(".", 1)
    else:
        disease_name, variant_name = spec, None
    disease = diseases.get(disease_name)

    rows: list[dict] = []
    skipped = 0
    for _, row in merged.iterrows():
        feats = row_to_features(row, amp_scale)
        if not required.issubset(feats.keys()):
            skipped += 1
            continue
        age = None
        av = feats.get("age_years")
        if av is None and "PERSON_AGE" in row.index and pd.notna(row.get("PERSON_AGE")):
            try:
                age = int(float(row.get("PERSON_AGE")))
            except (TypeError, ValueError):
                age = None
        ctx = EvalContext(features=dict(feats), sex=None, age_years=age)
        res = evaluate_disease(disease, ctx)
        fired = 0
        if variant_name is None:
            fired = int(res.fired)
        else:
            for vr in res.variant_results:
                if vr.variant_name == variant_name:
                    fired = int(vr.fired)
                    break
        rows.append({**{k: row.get(k) for k in keys}, "pred": fired})
    return pd.DataFrame(rows), required, skipped


def score_pair(
    abnormality: str, spec: str, preds: pd.DataFrame, gt: pd.DataFrame,
    pred_keys: list[str], col_lookup: dict[str, str],
    required: set[str], skipped: int,
) -> dict:
    gtc = col_lookup.get(_norm(abnormality))
    if gtc is None:
        return {"abnormality": abnormality, "rule": spec,
                "status": "NO GT COLUMN",
                "required_features": ", ".join(sorted(required)),
                "n_skipped_missing_features": skipped}
    if preds.empty:
        return {"abnormality": abnormality, "rule": spec,
                "status": "NO ECGs WITH ALL FEATURES",
                "required_features": ", ".join(sorted(required)),
                "n_skipped_missing_features": skipped}
    m = preds.merge(gt[pred_keys + [gtc]], on=pred_keys, how="inner")
    if m.empty:
        return {"abnormality": abnormality, "rule": spec,
                "status": "NO GT JOIN",
                "required_features": ", ".join(sorted(required)),
                "n_skipped_missing_features": skipped}
    gt_pos = m[gtc].apply(_truthy)
    pred_pos = m["pred"] >= 1
    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    tn = int((~pred_pos & ~gt_pos).sum())
    return {"abnormality": abnormality, "rule": spec,
            "required_features": ", ".join(sorted(required)),
            "n_required": len(required), "n_eval": len(m),
            "n_skipped_missing_features": skipped,
            "GT_pos": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(safe_div(tp, tp + fn), 4),
            "specificity": round(safe_div(tn, tn + fp), 4),
            "PPV": round(safe_div(tp, tp + fp), 4),
            "F1": round(safe_div(2 * tp, 2 * tp + fp + fn), 4),
            "prevalence": round(safe_div(tp + fn, len(m)), 4),
            "status": "ok"}


# ── abnormality-level metrics + suppression ──────────────────────────────

def build_measurements_table(
    merged: pd.DataFrame, amp_scale: float, keys: list[str]
) -> pd.DataFrame:
    """One row per ECG with the 6 inputs that suppression cares about."""
    rows = []
    for _, row in merged.iterrows():
        f = row_to_features(row, amp_scale)
        rows.append({
            **{k: row.get(k) for k in keys},
            "heart_rate": f.get("ventricular_rate_bpm"),
            "pr_interval": f.get("PR_ms"),
            "qrs_duration": f.get("QRS_ms"),
            "axis": f.get("QRS_axis_deg"),
            "qt_interval": f.get("QT_ms"),
            "qtc": f.get("QTc_Bazett_ms") or f.get("QTc_Framingham_ms")
                  or f.get("QTc_Fridericia_ms"),
        })
    df = pd.DataFrame(rows)
    for k in keys:
        df[k] = df[k].astype(str)
    return df


def score_abnormality(
    abnormality: str, mode: str,
    pred_col: str, df: pd.DataFrame, gt_col: str,
) -> dict:
    gt_pos = df[gt_col].apply(_truthy)
    pred_pos = df[pred_col].apply(_truthy)
    tp = int((pred_pos & gt_pos).sum())
    fp = int((pred_pos & ~gt_pos).sum())
    fn = int((~pred_pos & gt_pos).sum())
    tn = int((~pred_pos & ~gt_pos).sum())
    return {"abnormality": abnormality, "mode": mode,
            "n_eval": len(df), "GT_pos": tp + fn,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(safe_div(tp, tp + fn), 4),
            "specificity": round(safe_div(tn, tn + fp), 4),
            "PPV": round(safe_div(tp, tp + fp), 4),
            "F1": round(safe_div(2 * tp, 2 * tp + fp + fn), 4)}


def apply_suppression_per_row(
    pred_df: pd.DataFrame, meas_df: pd.DataFrame,
    abn_to_supp: dict[str, str], abn_labels: list[str], keys: list[str],
) -> pd.DataFrame:
    """Return a DataFrame of post-suppression abnormality predictions."""
    from suppression.engine import apply_suppression
    from scripts.apply_suppression import ALL_LABELS

    meas_cols_in_pred = all(c in pred_df.columns for c in MEAS_COLS)
    if meas_cols_in_pred:
        joined = pred_df
    else:
        joined = pred_df.merge(meas_df, on=keys, how="left")
    out_rows = []
    for _, row in tqdm(joined.iterrows(), total=len(joined),
                        desc="Suppression", unit="ecg"):
        # build suppression-vocabulary prediction dict
        raw = {l: 0 for l in ALL_LABELS}
        for abn in abn_labels:
            sup_label = abn_to_supp.get(abn)
            if sup_label and sup_label in raw and _truthy(row.get(abn)):
                raw[sup_label] = 1
        meas = {
            "heart_rate": row.get("heart_rate"),
            "pr_interval": row.get("pr_interval"),
            "qrs_duration": row.get("qrs_duration"),
            "axis": row.get("axis"),
            "qt_interval": row.get("qt_interval"),
            "qtc": row.get("qtc"),
        }
        meas = {k: (None if pd.isna(v) else float(v))
                for k, v in meas.items()}
        result = apply_suppression(raw, meas)
        active = result["active_raw"]
        out = {k: row[k] for k in keys}
        for abn in abn_labels:
            sup_label = abn_to_supp.get(abn)
            survived = bool(sup_label and active.get(sup_label, False))
            # If we never mapped this abnormality into the suppression
            # vocabulary, treat post-suppression == pre-suppression so we
            # don't artificially zero it out.
            if not sup_label or sup_label not in ALL_LABELS:
                survived = bool(_truthy(row.get(abn)))
            out[abn] = int(survived)
        out_rows.append(out)
    return pd.DataFrame(out_rows)


# ── main ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-csv", type=Path, default=None,
                    help="Required unless --predictions-in is set.")
    ap.add_argument("--global-csv", type=Path, default=None,
                    help="Required unless --predictions-in is set.")
    ap.add_argument("--interval-csv", type=Path, default=None)
    ap.add_argument("--keys", type=str, default="PERSON_ID,EVENT_DTM")
    ap.add_argument("--gt-csv", type=Path, required=True)
    ap.add_argument("--gt-keys", type=str, default="NFER_PID,NFER_DTM")
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--predictions-in", type=Path, default=None,
                    help="Skip rule evaluation. Load an existing per-ECG "
                         "predictions CSV (keys + 44 abnormality columns + "
                         "measurements) and only run suppression / scoring.")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "nfer_per_rule_metrics.csv")
    ap.add_argument("--out-abn-metrics", type=Path,
                    default=ROOT / "reports" / "nfer_per_abnormality_metrics.csv")
    ap.add_argument("--out-predictions", type=Path,
                    default=ROOT / "reports" / "nfer_per_rule_predictions.csv")
    ap.add_argument("--apply-suppression", action="store_true",
                    help="Run suppression/engine on the per-ECG predictions and "
                         "also write post-suppression metrics + columns.")
    args = ap.parse_args()

    if args.predictions_in is None and (args.lead_csv is None
                                        or args.global_csv is None):
        ap.error("--lead-csv and --global-csv are required unless "
                 "--predictions-in is given.")

    pred_keys = [k.strip() for k in args.keys.split(",")]
    gt_keys = [k.strip() for k in args.gt_keys.split(",")]
    abn_labels = list(ABNORMALITY_TO_PRED.keys())

    print("Loading GT parquet...")
    gt = load_gt_table(args.gt_csv)
    col_lookup = {_norm(c): c for c in gt.columns}
    gt = gt.rename(columns={gk: pk for gk, pk in zip(gt_keys, pred_keys)})
    for k in pred_keys:
        gt[k] = gt[k].astype(str)

    # ── Mode A: load existing predictions and skip rule eval ──────────
    if args.predictions_in is not None:
        print(f"Loading predictions from {args.predictions_in}...")
        pred_df = pd.read_csv(args.predictions_in)
        for k in pred_keys:
            if k not in pred_df.columns:
                raise SystemExit(f"--predictions-in is missing key column {k!r}")
            pred_df[k] = pred_df[k].astype(str)
        missing_meas = [c for c in MEAS_COLS if c not in pred_df.columns]
        if args.apply_suppression and missing_meas:
            if args.lead_csv is None or args.global_csv is None:
                raise SystemExit(
                    "--predictions-in is missing measurement columns "
                    f"{missing_meas} required for suppression. Pass "
                    "--lead-csv / --global-csv (and --interval-csv if you "
                    "have it) so they can be rebuilt from the source CSVs, "
                    "or regenerate the predictions file with this version "
                    "of eval_per_rule.py (it now embeds measurements)."
                )
            print("Predictions CSV lacks measurements; rebuilding from CSVs...")
            lead_df = pd.read_csv(args.lead_csv)
            glob_df = pd.read_csv(args.global_csv)
            ivl_df = pd.read_csv(args.interval_csv) if args.interval_csv else None
            mtmp = merge_csvs(lead_df, glob_df, ivl_df, pred_keys,
                              args.amp_scale, complete_only=False)
            for k in pred_keys:
                mtmp[k] = mtmp[k].astype(str)
            meas_df = build_measurements_table(mtmp, args.amp_scale, pred_keys)
            pred_df = pred_df.merge(meas_df, on=pred_keys, how="left")
        # If the CSV had *__supp columns from a previous run, drop them so we
        # don't clash with the new ones.
        drop_supp = [c for c in pred_df.columns if c.endswith("__supp")
                     or c.endswith("__supp_x") or c.endswith("__supp_y")]
        if drop_supp:
            pred_df = pred_df.drop(columns=drop_supp)
        print(f"Loaded predictions for {len(pred_df)} ECGs.")
        # We still need a metrics file slot for the per-rule CSV — write a
        # one-row note so downstream tools don't choke on a missing file.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"status": "skipped (--predictions-in)"}]).to_csv(
            args.out, index=False)
        merged = None  # not needed downstream
    else:
        # ── Mode B: full pipeline ─────────────────────────────────────
        print("Loading rules...")
        diseases: dict = {}
        for yf in sorted(RULES_DIR.glob("*.yaml")):
            try:
                d = load_disease_yaml(yf)
                diseases[d.disease] = d
            except Exception as e:
                print(f"  WARN: {yf.name}: {e}")

        print("Loading CSVs...")
        lead_df = pd.read_csv(args.lead_csv)
        glob_df = pd.read_csv(args.global_csv)
        ivl_df = pd.read_csv(args.interval_csv) if args.interval_csv else None
        merged = merge_csvs(lead_df, glob_df, ivl_df, pred_keys, args.amp_scale,
                            complete_only=False)
        print(f"Merged cohort: {len(merged)} ECGs")
        for k in pred_keys:
            merged[k] = merged[k].astype(str)

        # ── per-rule eval ─────────────────────────────────────────────
        results: list[dict] = []
        per_spec_preds: dict[str, pd.DataFrame] = {}
        pairs = [(ab, spec) for ab, specs in ABNORMALITY_TO_PRED.items()
                 for spec in specs]
        print(f"Evaluating {len(pairs)} (abnormality, rule) pairs...")
        for abnormality, spec in tqdm(pairs, unit="rule"):
            if spec not in per_spec_preds:
                preds, required, skipped = evaluate_one_rule(
                    spec, diseases, merged, args.amp_scale, pred_keys
                )
                per_spec_preds[spec] = preds
                preds.attrs["required"] = required
                preds.attrs["skipped"] = skipped
            preds = per_spec_preds[spec]
            results.append(score_pair(
                abnormality, spec, preds, gt, pred_keys, col_lookup,
                preds.attrs.get("required", set()),
                preds.attrs.get("skipped", 0),
            ))

        df = pd.DataFrame(results)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote per-rule metrics ({len(df)} rows) -> {args.out}")

        # ── aggregate to per-abnormality wide predictions ─────────────
        print("Aggregating per-spec → per-abnormality...")
        abn_pred: dict[tuple, dict[str, int]] = defaultdict(dict)
        all_keys_seen: set[tuple] = set()
        for abn, specs in ABNORMALITY_TO_PRED.items():
            for spec in specs:
                preds = per_spec_preds.get(spec)
                if preds is None or preds.empty:
                    continue
                for _, row in preds.iterrows():
                    k = tuple(str(row[c]) for c in pred_keys)
                    all_keys_seen.add(k)
                    prev = abn_pred[k].get(abn, 0)
                    abn_pred[k][abn] = max(prev, int(row["pred"]))

        pred_rows = []
        for k in sorted(all_keys_seen):
            row = dict(zip(pred_keys, k))
            for abn in abn_labels:
                row[abn] = int(abn_pred[k].get(abn, 0))
            pred_rows.append(row)
        pred_df = pd.DataFrame(pred_rows)

        # Bake measurements into the predictions table so a future
        # --predictions-in run can apply suppression without the CSVs.
        meas_df = build_measurements_table(merged, args.amp_scale, pred_keys)
        pred_df = pred_df.merge(meas_df, on=pred_keys, how="left")

    # ── per-abnormality metrics (pre-suppression) ─────────────────────
    abn_metrics: list[dict] = []
    pre_join = pred_df.merge(gt, on=pred_keys, how="inner", suffixes=("", "_gt"))
    for abn in abn_labels:
        gtc = col_lookup.get(_norm(abn))
        if gtc is None:
            continue
        # if merge collided, the GT column is now suffixed
        gt_col = gtc if gtc in pre_join.columns and gtc != abn else f"{gtc}_gt"
        if gt_col not in pre_join.columns:
            continue
        abn_metrics.append(score_abnormality(abn, "raw", abn, pre_join, gt_col))

    # ── optional suppression pass ─────────────────────────────────────
    post_df = None
    if args.apply_suppression:
        if all(c in pred_df.columns for c in MEAS_COLS):
            meas_df = pred_df[pred_keys + MEAS_COLS].copy()
        else:
            print("Building per-ECG measurements for suppression...")
            meas_df = build_measurements_table(merged, args.amp_scale, pred_keys)
        post_df = apply_suppression_per_row(
            pred_df, meas_df, ABNORMALITY_TO_SUPPRESSION, abn_labels, pred_keys
        )
        post_df = post_df.rename(columns={abn: f"{abn}__supp" for abn in abn_labels})
        post_join = post_df.merge(gt, on=pred_keys, how="inner", suffixes=("", "_gt"))
        for abn in abn_labels:
            gtc = col_lookup.get(_norm(abn))
            if gtc is None:
                continue
            gt_col = gtc if gtc in post_join.columns and gtc != f"{abn}__supp" else f"{gtc}_gt"
            if gt_col not in post_join.columns:
                continue
            abn_metrics.append(score_abnormality(
                abn, "post-suppression", f"{abn}__supp", post_join, gt_col,
            ))

    am = pd.DataFrame(abn_metrics)
    args.out_abn_metrics.parent.mkdir(parents=True, exist_ok=True)
    am.to_csv(args.out_abn_metrics, index=False)
    print(f"Wrote per-abnormality metrics ({len(am)} rows) -> "
          f"{args.out_abn_metrics}")

    # ── per-ECG predictions CSV ───────────────────────────────────────
    if post_df is not None:
        out_pred = pred_df.merge(post_df, on=pred_keys, how="left")
    else:
        out_pred = pred_df
    args.out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(args.out_predictions, index=False)
    print(f"Wrote per-ECG predictions ({len(out_pred)} rows) -> "
          f"{args.out_predictions}")

    # ── console summary ───────────────────────────────────────────────
    if not am.empty:
        pd.set_option("display.width", 200)
        pd.set_option("display.max_colwidth", 60)
        wide = am.pivot_table(
            index="abnormality", columns="mode",
            values=["F1", "sensitivity", "PPV", "TP", "FP", "FN", "GT_pos"],
            aggfunc="first",
        )
        print("\n=== Per-abnormality metrics (pre vs post suppression) ===")
        print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
