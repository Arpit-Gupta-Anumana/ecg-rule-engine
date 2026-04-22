"""Spot-check LBBB / RBBB / IRBBB / ILBBB on a random sample + targeted cases.

Prints GE's own text and which of our variants fire, then prints confusion
counts using GE's interpretation text as the pseudo-GT.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
from ecg_rule_engine.features.sapphire_xml import parse_file

RULES = {
    "LBBB":                   "rules/lbbb.yaml",
    "RBBB":                   "rules/rbbb.yaml",
    "IncompleteBundleBlocks": "rules/incomplete_blocks.yaml",
}

GE_KW = {
    "LBBB":                   ["left bundle branch block"],
    "RBBB":                   ["right bundle branch block"],
    "IncompleteBundleBlocks": ["incomplete right", "incomplete left", "rsr'"],
}


def ge_has(text: str, kws: list[str]) -> bool:
    low = text.lower()
    return any(k in low for k in kws)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    xml_dir = Path("/Users/arpit.gupta/Downloads/XMLS")

    diseases = {name: load_disease_yaml(Path(p)) for name, p in RULES.items()}

    xmls = sorted(xml_dir.glob("*.Xml"))
    random.seed(42)
    sample = random.sample(xmls, min(n, len(xmls)))

    counts = {name: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for name in diseases}
    disagreements: list[tuple[str, str, str, str]] = []

    for xml in sample:
        try:
            rec = parse_file(xml)
        except Exception:
            continue
        merged = dict(rec.features)
        merged.update({k: float(v) for k, v in rec.ge_flags.items()})
        ctx = EvalContext(features=merged, sex=rec.sex, age_years=rec.age_years)
        ge_text = " | ".join(s.get("text", "") for s in rec.ge_statements if s.get("text"))

        for name, d in diseases.items():
            res = evaluate_disease(d, ctx)
            ge_pos = ge_has(ge_text, GE_KW[name])
            fired_variants = ",".join(v.variant_name for v in res.variant_results if v.fired)
            key = ("tp" if res.fired and ge_pos
                   else "fp" if res.fired and not ge_pos
                   else "fn" if not res.fired and ge_pos
                   else "tn")
            counts[name][key] += 1
            if key in ("fp", "fn"):
                disagreements.append((xml.name, name, ge_text[:180], fired_variants or "-"))

    print(f"\n=== {len(sample)} ECGs  -  confusion vs GE text ===")
    print(f"{'Disease':28s}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}   Sens   Spec")
    for name, c in counts.items():
        sens = c['tp'] / max(1, c['tp'] + c['fn'])
        spec = c['tn'] / max(1, c['tn'] + c['fp'])
        print(f"{name:28s}  {c['tp']:>4} {c['fp']:>4} {c['fn']:>4} {c['tn']:>4}   "
              f"{sens*100:5.1f}%  {spec*100:5.1f}%")

    print(f"\n=== {len(disagreements)} disagreements ===")
    for xname, dname, ge, ours in disagreements[:20]:
        print(f"[{dname:4}] {xname}")
        print(f"  GE:   {ge}")
        print(f"  Ours: fired={ours}\n")


if __name__ == "__main__":
    main()
