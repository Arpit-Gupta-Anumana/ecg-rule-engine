"""Spot-check the new MI/STEMI/ischemia rules on a sample of real GE XMLs.

For each sampled ECG we print:
    * GE's own interpretation text (ground-truth-ish baseline)
    * Which of our rules fired
    * Obvious disagreements (e.g. our rule fires when GE says normal, or
      vice-versa)

Usage:
    python scripts/spot_check_mi.py  [N=25]  [XML_DIR=/Users/.../Downloads/XMLS]
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
from ecg_rule_engine.features.sapphire_xml import parse_file

RULES_OF_INTEREST = [
    "rules/q_wave_mi.yaml",
    "rules/st_elevation_injury.yaml",
    "rules/st_depression_ischemia.yaml",
    "rules/t_wave_ischemia.yaml",
    "rules/pericarditis_early_repol.yaml",
]

GE_KEYWORDS = {
    "QWaveMI": [
        "infarct",
        "myocardial infarction",
        "cannot rule out",
        "anterior infarct",
        "inferior infarct",
        "lateral infarct",
        "septal infarct",
        "posterior infarct",
    ],
    "STElevationInjury": [
        "st elevation",
        "injury",
        "stemi",
        "acute mi",
        "injury pattern",
    ],
    "STDepressionIschemia": [
        "st depression",
        "subendocardial",
        "st abnormality",
    ],
    "TWaveIschemia": [
        "t wave",
        "ischemia",
        "t wave abnormality",
        "t wave inversion",
    ],
    "PericarditisOrEarlyRepol": [
        "early repolarization",
        "pericarditis",
    ],
}


def ge_says_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k in low for k in keywords)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    xml_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        "/Users/arpit.gupta/Downloads/XMLS"
    )

    diseases = [load_disease_yaml(Path(p)) for p in RULES_OF_INTEREST]
    print(
        "Loaded diseases:",
        ", ".join(f"{d.disease}({len(d.variants)})" for d in diseases),
    )

    xmls = sorted(xml_dir.glob("*.Xml"))
    random.seed(42)
    sample = random.sample(xmls, min(n, len(xmls)))
    print(f"Sampled {len(sample)} of {len(xmls)} XMLs from {xml_dir}\n")

    agreement = {d.disease: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for d in diseases}

    for xml in sample:
        try:
            rec = parse_file(xml)
        except Exception as exc:  # noqa: BLE001
            print(f"[SKIP] {xml.name}: parse error {exc}")
            continue

        merged_feats = dict(rec.features)
        merged_feats.update({k: float(v) for k, v in rec.ge_flags.items()})
        ctx = EvalContext(
            features=merged_feats,
            sex=rec.sex,
            age_years=rec.age_years,
        )
        ge_text = " | ".join(
            s.get("text", "") for s in rec.ge_statements if s.get("text")
        )

        fired_names: list[str] = []
        for d in diseases:
            res = evaluate_disease(d, ctx)
            ge_positive = ge_says_any(ge_text, GE_KEYWORDS.get(d.disease, []))
            key = (
                "tp" if res.fired and ge_positive
                else "fp" if res.fired and not ge_positive
                else "fn" if not res.fired and ge_positive
                else "tn"
            )
            agreement[d.disease][key] += 1
            if res.fired:
                fired_names.append(
                    f"{d.disease}({','.join(v.variant_name for v in res.variant_results if v.fired)})"
                )

        print(f"{xml.name}")
        print(f"  GE:    {ge_text[:200]}")
        print(f"  Ours:  {'  '.join(fired_names) if fired_names else '(none)'}")
        print()

    print("\n=== Agreement matrix (GE text keyword as pseudo-GT) ===")
    print(f"{'Disease':30s}  {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    for name, counts in agreement.items():
        print(
            f"{name:30s}  {counts['tp']:>4} {counts['fp']:>4} "
            f"{counts['fn']:>4} {counts['tn']:>4}"
        )


if __name__ == "__main__":
    main()
