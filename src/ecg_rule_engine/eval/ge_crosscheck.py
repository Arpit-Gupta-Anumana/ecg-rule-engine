"""Agreement matrix between our rule engine and GE's native interpretation.

GE's Sapphire XML emits its own disease statements (e.g. "Left ventricular
hypertrophy (LVH2)"). We can map those to canonical disease names via keyword
lists and then compare:

- 2x2 agreement matrix (ours vs GE) per disease.
- List of ECGs where GE disagrees with us in either direction — these are
  prime candidates for extraction-bug review, because the GE device IS the
  reference implementation of the manual.

This module is deliberately thin — it's ~100 lines of pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# Default disease -> GE statement keywords. Keys must match the disease names
# in the loaded YAML ruleset. The keyword list is case-insensitive substring
# matched against the joined GE statement text.
DEFAULT_DISEASE_KEYWORDS: dict[str, list[str]] = {
    "LVH": [
        "left ventricular hypertrophy",
        "voltage criteria for left ventricular hypertrophy",
    ],
    "RVH": ["right ventricular hypertrophy"],
    "BVH": ["biventricular hypertrophy"],
    "RBBB": ["right bundle branch block", "rbbb"],
    "LBBB": ["left bundle branch block", "lbbb"],
    "AFib": ["atrial fibrillation"],
    "AFlutter": ["atrial flutter"],
    "WPW": ["wpw", "wolff-parkinson-white", "pre-excitation", "preexcitation"],
    "LAE": ["left atrial enlargement"],
    "RAE": ["right atrial enlargement"],
    "LAFB": ["left anterior fascicular block", "lafb"],
    "LPFB": ["left posterior fascicular block", "lpfb"],
}


@dataclass
class Agreement:
    disease: str
    both_positive: int
    ours_only: int   # our engine fires, GE does not
    ge_only: int     # GE fires, our engine does not
    both_negative: int
    n: int

    def agreement_rate(self) -> float:
        return (self.both_positive + self.both_negative) / self.n if self.n else float("nan")

    def cohens_kappa(self) -> float:
        if self.n == 0:
            return float("nan")
        po = (self.both_positive + self.both_negative) / self.n
        p_ours = (self.both_positive + self.ours_only) / self.n
        p_ge = (self.both_positive + self.ge_only) / self.n
        pe = p_ours * p_ge + (1 - p_ours) * (1 - p_ge)
        if 1 - pe == 0:
            return float("nan")
        return (po - pe) / (1 - pe)

    def to_dict(self) -> dict:
        return {
            "disease": self.disease,
            "n": self.n,
            "both_pos": self.both_positive,
            "ours_only": self.ours_only,
            "ge_only": self.ge_only,
            "both_neg": self.both_negative,
            "agreement_rate": self.agreement_rate(),
            "cohens_kappa": self.cohens_kappa(),
        }


def ge_labels_from_statements(
    statements_series: pd.Series,
    disease_keywords: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Turn a Series of joined GE statement text into a 0/1 DataFrame of disease labels."""
    kw_map = disease_keywords or DEFAULT_DISEASE_KEYWORDS
    s = statements_series.fillna("").astype(str).str.lower()
    out: dict[str, pd.Series] = {}
    for disease, keywords in kw_map.items():
        mask = None
        for kw in keywords:
            m = s.str.contains(kw.lower(), regex=False)
            mask = m if mask is None else (mask | m)
        out[disease] = (mask if mask is not None else pd.Series(False, index=s.index)).astype(int)
    return pd.DataFrame(out, index=statements_series.index)


def agreement(ours: pd.Series, ge: pd.Series, disease: str) -> Agreement:
    df = pd.concat([ours.rename("ours"), ge.rename("ge")], axis=1).dropna().astype(int)
    both_pos = int(((df["ours"] == 1) & (df["ge"] == 1)).sum())
    ours_only = int(((df["ours"] == 1) & (df["ge"] == 0)).sum())
    ge_only = int(((df["ours"] == 0) & (df["ge"] == 1)).sum())
    both_neg = int(((df["ours"] == 0) & (df["ge"] == 0)).sum())
    return Agreement(
        disease=disease,
        both_positive=both_pos, ours_only=ours_only, ge_only=ge_only, both_negative=both_neg,
        n=len(df),
    )


def build_agreement_table(
    engine_outputs: dict[str, pd.DataFrame],
    ge_label_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    for disease, engine_df in engine_outputs.items():
        if disease not in ge_label_df.columns:
            continue
        ours = engine_df["disease_fired"].astype(int)
        ge = ge_label_df[disease].astype(int)
        ag = agreement(ours, ge, disease)
        rows.append(ag.to_dict())
    return pd.DataFrame(rows).set_index("disease") if rows else pd.DataFrame()


def disagreement_examples(
    engine_df: pd.DataFrame,
    ge_labels: pd.Series,
    disease: str,
    *,
    direction: str = "both",  # "ours_only" | "ge_only" | "both"
    limit: int = 20,
) -> pd.DataFrame:
    df = pd.concat([engine_df["disease_fired"].rename("ours"), ge_labels.rename("ge")], axis=1).dropna()
    df["disease"] = disease
    if direction == "ours_only":
        df = df[(df["ours"] == 1) & (df["ge"] == 0)]
    elif direction == "ge_only":
        df = df[(df["ours"] == 0) & (df["ge"] == 1)]
    else:
        df = df[df["ours"] != df["ge"]]
    return df.head(limit)


__all__ = [
    "Agreement",
    "DEFAULT_DISEASE_KEYWORDS",
    "ge_labels_from_statements",
    "agreement",
    "build_agreement_table",
    "disagreement_examples",
]
