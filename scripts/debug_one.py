"""Debug a single XML: print STJ, T, Q values for all 12 leads + any fired variants."""
from __future__ import annotations

import sys
from pathlib import Path

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
from ecg_rule_engine.engine.trace import format_trace
from ecg_rule_engine.features.sapphire_xml import parse_file

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def main() -> None:
    xml = Path(sys.argv[1])
    rec = parse_file(xml)
    merged = dict(rec.features)
    merged.update({k: float(v) for k, v in rec.ge_flags.items()})
    ctx = EvalContext(features=merged, sex=rec.sex, age_years=rec.age_years)

    print(f"=== {xml.name} ===")
    print(f"sex={rec.sex} age={rec.age_years}")
    print(f"GE text: {' | '.join(s.get('text','') for s in rec.ge_statements if s.get('text'))}\n")

    headers = ["lead", "R", "S", "Q", "Q_ms", "STJ", "STM", "T"]
    print(f"{'lead':>4} {'R':>7} {'S':>7} {'Q':>7} {'Qms':>6} {'STJ':>7} {'STM':>7} {'T':>7}")
    for lead in LEADS:
        def g(pfx: str, unit: str = "mV") -> str:
            v = merged.get(f"{pfx}_{lead}_{unit}")
            return f"{v:7.3f}" if v is not None else "    -  "

        print(f"{lead:>4} {g('R')} {g('S')} {g('Q')} {g('Q',unit='ms'):>6} "
              f"{g('STJ')} {g('STM')} {g('T')}")

    for yml in sys.argv[2:]:
        d = load_disease_yaml(Path(yml))
        res = evaluate_disease(d, ctx)
        if res.fired or any(v.fired for v in res.variant_results):
            print(f"\n--- {d.disease} ---  fired={res.fired}")
            for v in res.variant_results:
                print(f"  variant={v.variant_name}  fired={v.fired}")


if __name__ == "__main__":
    main()
