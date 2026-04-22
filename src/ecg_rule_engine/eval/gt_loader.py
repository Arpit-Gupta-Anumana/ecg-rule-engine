"""Ground truth loader + patient-level train/val/test split.

Input: CSV with columns:
    ecg_id, patient_id, sex, age_years, disease, label  [,clinician_id]

The loader:
1. Reads the CSV.
2. Validates types.
3. Joins with the feature DataFrame produced by `features.sapphire_xml.parse_dir`,
   warning loudly about mismatched `ecg_id`s.
4. Pivots into a wide label matrix (one column per disease).
5. Splits by `patient_id` (never by `ecg_id`) using a deterministic hash so
   splits are reproducible without re-shuffling when new patients are added.

Split sizes default to 60 / 20 / 20.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

log = logging.getLogger(__name__)

Split = Literal["train", "val", "test"]
_REQUIRED_COLS = {"ecg_id", "patient_id", "sex", "disease", "label"}


@dataclass
class GroundTruth:
    long_df: pd.DataFrame          # one row per (ecg_id, disease)
    wide_labels: pd.DataFrame      # index=ecg_id, columns=diseases, values in {0,1}
    meta: pd.DataFrame             # index=ecg_id, columns=[patient_id, sex, age_years]

    @property
    def diseases(self) -> list[str]:
        return list(self.wide_labels.columns)


def load_gt_csv(path: str | Path) -> GroundTruth:
    p = Path(path)
    df = pd.read_csv(p)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing required columns: {sorted(missing)}")

    df["ecg_id"] = df["ecg_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df["disease"] = df["disease"].astype(str)
    df["sex"] = df["sex"].astype(str).str.upper().str[:1]
    df["label"] = df["label"].astype(int).clip(0, 1)
    if "age_years" in df.columns:
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")

    meta = (
        df.drop_duplicates("ecg_id")
          .set_index("ecg_id")[[c for c in ["patient_id", "sex", "age_years"] if c in df.columns]]
    )

    wide = (
        df.pivot_table(index="ecg_id", columns="disease", values="label", aggfunc="max", fill_value=0)
          .astype(int)
    )

    return GroundTruth(long_df=df, wide_labels=wide, meta=meta)


def merge_features_and_gt(
    features_df: pd.DataFrame,
    gt: GroundTruth,
    *,
    how: Literal["inner", "left"] = "inner",
) -> pd.DataFrame:
    """Return a DataFrame with one row per ECG and every feature, GE flag, and GT label column."""
    if features_df.index.name != "ecg_id":
        raise ValueError("features_df must be indexed by ecg_id")
    missing_in_features = set(gt.wide_labels.index) - set(features_df.index)
    missing_in_gt = set(features_df.index) - set(gt.wide_labels.index)
    if missing_in_features:
        log.warning("%d ecg_ids in GT but not in features.", len(missing_in_features))
    if missing_in_gt:
        log.warning("%d ecg_ids in features but not in GT.", len(missing_in_gt))

    labels = gt.wide_labels.add_prefix("gt__")
    # Prefer demographic columns already present on the feature DF (parsed from
    # XML) over the GT-provided copies; only bring in meta columns that are
    # missing.  This keeps the merged frame free of duplicate/overlapping
    # columns while still backfilling from GT when XML lacked them.
    meta_extra = gt.meta[[c for c in gt.meta.columns if c not in features_df.columns]]
    merged = features_df.join(meta_extra, how=how).join(labels, how=how)
    for col in ("patient_id", "sex", "age_years"):
        if col in features_df.columns and col in gt.meta.columns:
            merged[col] = merged[col].fillna(gt.meta[col])
    return merged


def _hash01(s: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}::{s}".encode()).hexdigest()
    return int(h[:12], 16) / (16 ** 12)


def split_by_patient(
    merged: pd.DataFrame,
    *,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    salt: str = "ecg_rule_engine_v1",
) -> pd.Series:
    """Return a Series (indexed like `merged`) with values in {'train','val','test'}.

    Assignment is deterministic on (salt, patient_id): every ECG of the same
    patient gets the same split. Adding new patients later does not reshuffle
    existing ones (only adds to splits).
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")
    if "patient_id" not in merged.columns:
        raise ValueError("merged DataFrame must have a 'patient_id' column")

    pid = merged["patient_id"].astype(str)
    u = pid.map(lambda s: _hash01(s, salt))
    split = pd.Series(index=merged.index, dtype="object", name="split")
    split[u < train_frac] = "train"
    split[(u >= train_frac) & (u < train_frac + val_frac)] = "val"
    split[u >= train_frac + val_frac] = "test"
    return split


__all__ = [
    "GroundTruth",
    "load_gt_csv",
    "merge_features_and_gt",
    "split_by_patient",
]
