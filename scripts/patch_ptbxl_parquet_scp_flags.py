"""Add/update SCP-derived `ge_*_flag` columns on cached PTB-XL parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecg_rule_engine.features.ptbxl import _derive_flags_from_scp  # noqa: E402

PARQUET = ROOT / "data" / "ptbxl_features.parquet"

_PATCH_KEYS = (
    "ilbbb_flag",
    "irbbb_flag",
    "lafb_flag",
    "lpfb_flag",
    "rvh_flag",
    "ivcb_flag",
)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    for k in _PATCH_KEYS:
        col = f"ge_{k}"
        vals: list[int] = []
        for _, row in df.iterrows():
            raw = row.get("scp_codes_json")
            if pd.isna(raw):
                codes: dict[str, float] = {}
            else:
                codes = {c: 1.0 for c in json.loads(str(raw)).keys()}
            full = _derive_flags_from_scp(codes)
            vals.append(int(full[k]))
        df[col] = vals
    df.to_parquet(PARQUET, index=False)
    print(f"Updated {PARQUET} with ge_* columns for {_PATCH_KEYS}")


if __name__ == "__main__":
    main()
