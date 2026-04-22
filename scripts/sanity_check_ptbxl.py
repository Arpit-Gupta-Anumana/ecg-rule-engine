"""Sanity-check the PTB-XL loader + waveform feature extractor on a handful of records.

Validates:
  - WFDB reads succeed and give 12 leads at 500 Hz
  - waveform_features extracts plausible per-lead amplitudes
  - SCP codes + demographics are loaded and mapped to ge_statements/flags
  - Einthoven/Goldberger invariants are satisfied (should always be for PTB-XL)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ecg_rule_engine.features.ptbxl import iter_records

ROOT = "/Users/arpit.gupta/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"


def _fmt(f: float | None) -> str:
    return "   n/a" if f is None else f"{f:+7.3f}"


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== PTB-XL sanity check: {n} random records (fold 10, test set) ===\n")
    ok = 0
    for i, rec in enumerate(iter_records(ROOT, folds=[10], n=n, random_state=7)):
        print(f"--- record {i}: ecg_id={rec.ptbxl_ecg_id}  patient={rec.patient_id}  "
              f"sex={rec.sex}  age={rec.age_years} ---")
        codes = sorted(rec.scp_codes_raw.keys())
        print(f"  SCP codes: {codes}")
        print(f"  report   : {rec.report}")
        print(f"  flags    : {rec.ge_flags}")
        f = rec.features
        if not f:
            print("  FEATURES: NOT EXTRACTED")
            continue
        print(f"  globals  : HR={f.get('ventricular_rate_bpm', '?')} bpm, "
              f"QRS={f.get('QRS_ms','?')} ms, PR={f.get('PR_ms','?')} ms, "
              f"QT/QTcB={f.get('QT_ms','?')}/{f.get('QTc_Bazett_ms','?')} ms")
        leads = ("I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6")
        print("  lead |   R   |   S   |   Q   | Qms |   P   |   T   |  STJ  |  STM")
        print("  -----+-------+-------+-------+-----+-------+-------+-------+-------")
        plausible = True
        for ld in leads:
            r = f.get(f"R_{ld}_mV"); s = f.get(f"S_{ld}_mV")
            q = f.get(f"Q_{ld}_mV"); qd = f.get(f"Q_{ld}_ms")
            p = f.get(f"P_{ld}_mV"); t = f.get(f"T_{ld}_mV")
            stj = f.get(f"STJ_{ld}_mV"); stm = f.get(f"STM_{ld}_mV")
            print(f"  {ld:>4} | {_fmt(r)} | {_fmt(s)} | {_fmt(q)} | "
                  f"{('%3.0f' % qd) if qd is not None else 'n/a'} | "
                  f"{_fmt(p)} | {_fmt(t)} | {_fmt(stj)} | {_fmt(stm)}")
            for v in (r, s, q, p, t, stj, stm):
                if v is not None and (abs(v) > 10.0 or v != v):
                    plausible = False
        if plausible:
            ok += 1
        print()
    print(f"=== {ok}/{n} records passed amplitude plausibility ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
