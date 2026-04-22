"""Promote a validated YAML rule from extracted/rules/ into the project rules/.

After promotion, re-validates the YAML and prints a summary.

Usage:
    python scripts/promote_rule.py extracted/rules/3_7_2_9_2__left_ventricular_hypertrophy.yaml
    python scripts/promote_rule.py extracted/rules/3_7_2_9_2__left_ventricular_hypertrophy.yaml --as lvh.yaml
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.dsl.schema import disease_features


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Path to the validated YAML in extracted/rules/")
    p.add_argument("--as", dest="as_name", default=None,
                   help="Rename the file when copying (default: keep source name)")
    p.add_argument("--rules-dir", default="rules",
                   help="Destination rules/ directory (default: rules)")
    args = p.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")

    d = load_disease_yaml(src)
    print(f"Loaded {d.disease} with {len(d.variants)} variant(s).")
    print(f"  features referenced: {sorted(disease_features(d))}")

    dst_dir = Path(args.rules_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_name = args.as_name or f"{d.disease.lower()}.yaml"
    dst = dst_dir / dst_name
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst} (delete it first, or use --as)")
    shutil.copy2(src, dst)
    print(f"Promoted to {dst}")

    load_disease_yaml(dst)
    print("Re-validated OK.")


if __name__ == "__main__":
    main()
