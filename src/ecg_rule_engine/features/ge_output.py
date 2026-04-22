"""Extract GE's own interpretation output from a parsed Sapphire record.

Kept separate from `sapphire_xml.py` so the cross-check logic (Phase 5) can
re-use it without importing measurement-parsing code. Provides two views:

1. `ge_statement_matches(record, substrings)` -> bool
   Whether ANY GE statement matches any of the given substrings (case-insensitive).
   Used to ask "did GE call this LVH?" when building the agreement matrix.

2. `build_ge_label_matrix(records, disease_keyword_map)` -> DataFrame
   One column per disease, 1 if GE flagged the disease, else 0.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .sapphire_xml import SapphireRecord


def ge_statement_matches(record: SapphireRecord, substrings: Iterable[str]) -> bool:
    subs = [s.lower() for s in substrings]
    joined = " | ".join((s.get("text") or "").lower() for s in record.ge_statements)
    return any(sub in joined for sub in subs)


def build_ge_label_matrix(
    records: list[SapphireRecord],
    disease_keyword_map: dict[str, list[str]],
) -> pd.DataFrame:
    """Return a DataFrame indexed by ecg_id with one 0/1 column per disease."""
    rows: list[dict[str, int | str]] = []
    for rec in records:
        row: dict[str, int | str] = {"ecg_id": rec.ecg_id}
        for disease, keywords in disease_keyword_map.items():
            row[disease] = int(ge_statement_matches(rec, keywords))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ecg_id")


__all__ = ["ge_statement_matches", "build_ge_label_matrix"]
