"""Top-level CLI.

Subcommands
-----------
    validate    Validate every YAML in rules/ against the DSL schema.
    parse-xml   Parse a Sapphire XML directory into a feature CSV.
    evaluate    End-to-end: parse XML + load rules + run engine + GT metrics +
                ensemble distillation + per-disease markdown reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from .dsl.loader import load_rules_dir
from .eval.distill import train_cart, train_k_of_n
from .eval.ge_crosscheck import (
    DEFAULT_DISEASE_KEYWORDS,
    agreement,
    ge_labels_from_statements,
)
from .eval.gt_loader import load_gt_csv, merge_features_and_gt, split_by_patient
from .eval.report import DiseaseReport, write_disease_report
from .eval.runner import run_engine
from .features.sapphire_xml import parse_dir

log = logging.getLogger("ecg_rule_engine")


def cmd_validate(args: argparse.Namespace) -> int:
    rules = load_rules_dir(args.rules)
    print(f"Loaded {len(rules)} disease(s):")
    for name, d in rules.items():
        print(f"  {name:<20} variants={len(d.variants):<2}  source={d.manual_source}")
    return 0


def cmd_parse_xml(args: argparse.Namespace) -> int:
    df = parse_dir(args.xml)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"Wrote {len(df)} row(s) to {out}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    rules = load_rules_dir(args.rules)
    log.info("Loaded %d disease(s).", len(rules))

    features = parse_dir(args.xml)
    if features.empty:
        log.error("No ECGs parsed from %s", args.xml)
        return 2
    log.info("Parsed %d ECG(s); %d feature columns.", len(features), features.shape[1])

    gt = load_gt_csv(args.gt)
    merged = merge_features_and_gt(features, gt, how="inner")
    if merged.empty:
        log.error("No overlap between feature ECGs and ground-truth ECGs.")
        return 3
    log.info("%d ECG(s) joined with GT.", len(merged))

    split = split_by_patient(
        merged, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac,
    )
    merged["split"] = split

    run_result = run_engine(rules, merged)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # GE label proxy
    stmts = merged.get("ge_statements_joined", pd.Series("", index=merged.index))
    ge_labels = ge_labels_from_statements(stmts, DEFAULT_DISEASE_KEYWORDS)

    for disease_name, engine_df in run_result.items():
        disease = rules[disease_name]
        y_col = f"gt__{disease_name}"
        if y_col not in merged.columns:
            log.warning("No GT column for %s; writing report without metrics.", disease_name)
            continue
        y_true = merged[y_col].astype(int)

        variant_cols = [c for c in engine_df.columns if c.startswith("fired_")]
        X = engine_df[variant_cols].astype(int)
        train_mask = (merged["split"] == "train")
        val_mask = (merged["split"] == "val")
        test_mask = (merged["split"] == "test")

        kofn = None
        cart = None
        if val_mask.sum() > 10 and train_mask.sum() > 10 and len(variant_cols) > 1:
            try:
                kofn = train_k_of_n(
                    X.loc[train_mask], y_true.loc[train_mask],
                    X.loc[val_mask],   y_true.loc[val_mask],
                    optimize_for="f1",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("k-of-N for %s failed: %s", disease_name, e)
            try:
                cart = train_cart(
                    X.loc[train_mask], y_true.loc[train_mask],
                    X.loc[val_mask],   y_true.loc[val_mask],
                    max_depth=args.cart_max_depth,
                    min_samples_leaf=max(5, int(0.01 * train_mask.sum())),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("CART for %s failed: %s", disease_name, e)

        ag = None
        if disease_name in ge_labels.columns:
            ours = engine_df["disease_fired"].astype(int).reindex(merged.index).fillna(0)
            ge_col = ge_labels[disease_name].reindex(merged.index).fillna(0)
            ag = agreement(ours.loc[test_mask | val_mask], ge_col.loc[test_mask | val_mask], disease_name)

        report = DiseaseReport(
            disease=disease,
            engine_output=engine_df,
            y_true=y_true,
            meta=merged[[c for c in ["patient_id", "sex", "age_years"] if c in merged.columns]],
            split=split,
            ge_labels=ge_labels.get(disease_name),
            agreement=ag,
            k_of_n=kofn,
            cart=cart,
        )
        path = write_disease_report(report, out / "reports")
        print(f"Wrote {path}")

    # Also save per-ecg engine output
    pooled = pd.concat(
        [df.add_prefix(f"{name}::") for name, df in run_result.items()], axis=1,
    )
    pooled.to_csv(out / "engine_output.csv")
    print(f"Wrote {out / 'engine_output.csv'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("ecg-rule-eval")
    p.add_argument("--log-level", default="INFO")
    sp = p.add_subparsers(dest="cmd", required=True)

    v = sp.add_parser("validate", help="Validate every YAML in rules/ dir")
    v.add_argument("--rules", default="rules")
    v.set_defaults(func=cmd_validate)

    pp = sp.add_parser("parse-xml", help="Parse GE Sapphire XMLs into a feature CSV")
    pp.add_argument("--xml", required=True)
    pp.add_argument("--out", required=True)
    pp.set_defaults(func=cmd_parse_xml)

    ev = sp.add_parser("evaluate", help="Full end-to-end eval + reports")
    ev.add_argument("--rules", default="rules")
    ev.add_argument("--xml", required=True)
    ev.add_argument("--gt", required=True)
    ev.add_argument("--out", default="reports")
    ev.add_argument("--train-frac", type=float, default=0.6)
    ev.add_argument("--val-frac", type=float, default=0.2)
    ev.add_argument("--test-frac", type=float, default=0.2)
    ev.add_argument("--cart-max-depth", type=int, default=3)
    ev.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
