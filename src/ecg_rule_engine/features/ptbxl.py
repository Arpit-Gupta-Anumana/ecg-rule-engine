"""Load PTB-XL WFDB records into SapphireRecord-compatible objects.

PTB-XL is a 21,799-ECG public dataset from PhysioNet. Each record is stored as
a WFDB pair (.dat + .hea) with 12 standard leads at 500 Hz (or 100 Hz in
records100/). Metadata (age, sex, SCP codes, splits) is in ptbxl_database.csv
and the statement dictionary in scp_statements.csv.

This loader:
1. Reads ptbxl_database.csv, optionally filtered by strat_fold.
2. For each row, loads the WFDB record (500 Hz by default).
3. Runs `extract_waveform_features` to derive per-lead amplitudes/ST offsets/
   Q-duration plus global QRS/PR/QT/heart-rate.
4. Parses the `scp_codes` dict (e.g. `{"CLBBB": 100.0, "SR": 0.0}`), joins the
   statement descriptions from scp_statements.csv, and puts them into the
   `ge_statements` / `ge_flags` fields - so downstream cross-check code that
   was written for GE Sapphire records works unchanged.

Units out: mV / ms / bpm / deg (matches features/registry.py).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import wfdb

from .sapphire_xml import SapphireRecord
from .waveform_features import LEADS_12, extract_waveform_features

# ---------------------------------------------------------------------------
# SCP-code -> rule-engine label mapping.
#
# The rule engine's downstream cross-check works by matching substrings against
# statement text. We build the statement text from the SCP description so that
# existing keyword lists in `DEFAULT_STATEMENT_FLAG_MAP` (e.g. "left bundle
# branch block") match naturally. For codes whose description is too terse or
# uses different phrasing, we add synthetic keyword expansions in
# `SCP_KEYWORD_EXPANSIONS` so the downstream match succeeds.
# ---------------------------------------------------------------------------

# Synthetic keywords appended to the statement text to make substring matches
# line up with the rule-engine's disease labels. Example: the SCP code "IMI"
# ("inferior myocardial infarction") gets the extra keywords
#   ["inferior infarct", "inferior mi"]
# so a rule disease labeled "Inferior_MI" (which matches on "inferior infarct"
# or "inferior mi") will pick it up.
SCP_KEYWORD_EXPANSIONS: dict[str, list[str]] = {
    # Conduction
    "CLBBB":  ["left bundle branch block", "lbbb"],
    "CRBBB":  ["right bundle branch block", "rbbb"],
    "ILBBB":  ["incomplete left bundle branch block", "ilbbb"],
    "IRBBB":  ["incomplete right bundle branch block", "irbbb"],
    "LAFB":   ["left anterior fascicular block", "lafb"],
    "LPFB":   ["left posterior fascicular block", "lpfb"],
    "IVCD":   ["intraventricular conduction", "ivcd"],
    "WPW":    ["wpw", "wolff-parkinson-white", "pre-excitation"],
    "1AVB":   ["first degree av block", "1st degree av block"],
    "2AVB":   ["second degree av block"],
    "3AVB":   ["third degree av block"],
    "LPR":    ["prolonged pr"],
    # Hypertrophy
    "LVH":    ["lvh", "left ventricular hypertrophy"],
    "RVH":    ["rvh", "right ventricular hypertrophy"],
    "VCLVH":  ["voltage criteria for lvh", "lvh"],
    "LAO/LAE": ["left atrial enlargement", "lae"],
    "RAO/RAE": ["right atrial enlargement", "rae"],
    "SEHYP":  ["septal hypertrophy"],
    # MI (diagnostic)
    "IMI":    ["inferior mi", "inferior infarct"],
    "AMI":    ["anterior mi", "anterior infarct"],
    "ASMI":   ["anteroseptal mi", "anteroseptal infarct", "septal mi", "septal infarct"],
    "ALMI":   ["anterolateral mi", "anterolateral infarct"],
    "ILMI":   ["inferolateral mi", "inferolateral infarct"],
    "IPMI":   ["inferoposterior mi"],
    "IPLMI":  ["inferoposterolateral mi"],
    "LMI":    ["lateral mi", "lateral infarct"],
    "PMI":    ["posterior mi", "posterior infarct"],
    "QWAVE":  ["q waves", "q wave mi"],
    # ST-elevation "injury" codes (closest PTB-XL has to STEMI)
    "INJAS":  ["anteroseptal injury", "anteroseptal stemi"],
    "INJAL":  ["anterolateral injury", "anterolateral stemi"],
    "INJIL":  ["inferolateral injury", "inferolateral stemi"],
    "INJIN":  ["inferior injury", "inferior stemi"],
    "INJLA":  ["lateral injury", "lateral stemi"],
    "STE_":   ["st elevation", "stemi"],
    # ST-depression / ischemia
    "STD_":   ["st depression"],
    "ISC_":   ["ischemia", "ischemic"],
    "ISCAS":  ["anteroseptal ischemia"],
    "ISCAL":  ["anterolateral ischemia"],
    "ISCLA":  ["lateral ischemia"],
    "ISCIL":  ["inferolateral ischemia"],
    "ISCIN":  ["inferior ischemia"],
    "ISCAN":  ["anterior ischemia"],
    # T-wave
    "INVT":   ["t wave inversion", "inverted t waves"],
    "NDT":    ["non-diagnostic t", "t abnormality"],
    "NT_":    ["non-specific t"],
    "TAB_":   ["t wave abnormality"],
    "LOWT":   ["low t"],
    # Rhythm
    "AFIB":   ["atrial fibrillation", "afib"],
    "AFLT":   ["atrial flutter"],
    "SR":     ["sinus rhythm"],
    "STACH":  ["sinus tachycardia"],
    "SBRAD":  ["sinus bradycardia"],
    "SARRH":  ["sinus arrhythmia"],
    "PACE":   ["pacemaker", "paced"],
    "PVC":    ["premature ventricular"],
    "PAC":    ["premature atrial"],
    "BIGU":   ["bigeminy"],
    "TRIGU":  ["trigeminy"],
    # Other
    "LNGQT":  ["long qt", "prolonged qt"],
    "LVOLT":  ["low voltage"],
    "HVOLT":  ["high voltage"],
    # Pericarditis / repolarization - PTB-XL has no dedicated code; closest are
    # NORM or STE_. Rules for those will fire mostly based on waveform, not label.
    "NORM":   ["normal ecg"],
}


@dataclass
class PTBXLRecord(SapphireRecord):
    ptbxl_ecg_id: int | None = None
    scp_codes_raw: dict[str, float] = field(default_factory=dict)
    report: str | None = None
    strat_fold: int | None = None
    heart_axis_str: str | None = None
    device: str | None = None


def _parse_scp_codes_col(s: str) -> dict[str, float]:
    """ptbxl_database.csv stores scp_codes as a Python-literal dict string."""
    if not isinstance(s, str) or not s.strip():
        return {}
    try:
        d = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}
    if not isinstance(d, dict):
        return {}
    return {str(k): float(v) for k, v in d.items()}


def _scp_to_statements(
    scp_codes: dict[str, float],
    scp_descriptions: dict[str, str],
    min_likelihood: float = 0.0,
) -> list[dict[str, Any]]:
    """Turn the scp_codes dict into a list of statement dicts, one per code.

    We keep codes even when likelihood=0 because PTB-XL uses 0 to mean "no
    likelihood information available"- not "ruled out". The presence of the
    code in the dict is the label.
    """
    stmts = []
    for code, likelihood in scp_codes.items():
        if likelihood < min_likelihood:
            continue
        desc = scp_descriptions.get(code, code)
        kw_exp = SCP_KEYWORD_EXPANSIONS.get(code, [])
        combined = desc + (" | " + " | ".join(kw_exp) if kw_exp else "")
        stmts.append({
            "text": combined,
            "code": code,
            "reason": f"likelihood={likelihood}",
        })
    return stmts


def _derive_flags_from_scp(scp_codes: dict[str, float]) -> dict[str, int]:
    """Produce the same flag set as `DEFAULT_STATEMENT_FLAG_MAP` uses - but
    sourced directly from SCP codes, not text matching.
    """
    codes = set(scp_codes.keys())
    return {
        "rhythm_afib_flag":     int("AFIB" in codes),
        "rhythm_aflutter_flag": int("AFLT" in codes),
        "rhythm_sinus_flag":    int("SR" in codes or "SARRH" in codes
                                    or "STACH" in codes or "SBRAD" in codes),
        "paced_flag":           int("PACE" in codes),
        "wpw_flag":             int("WPW" in codes),
        "lbbb_flag":            int("CLBBB" in codes),
        "rbbb_flag":            int("CRBBB" in codes),
        # Stated conduction (SCP) — used for exclusions that mirror the manual's
        # "if X is stated, skip …" logic when evaluating on PTB-XL.
        "ilbbb_flag":           int("ILBBB" in codes),
        "irbbb_flag":           int("IRBBB" in codes),
        "lafb_flag":            int("LAFB" in codes),
        "lpfb_flag":            int("LPFB" in codes),
        "rvh_flag":             int("RVH" in codes),
        # IVCB per §3.7.2.14: prolonged-QT tests are skipped if IVCB is stated.
        # Map to SCP IVCD plus incomplete bundle codes (see manual_departure_notes
        # on prolonged_qt.yaml).
        "ivcb_flag": int(
            "IVCD" in codes or "ILBBB" in codes or "IRBBB" in codes
        ),
    }


def _sex_str(v: Any) -> str | None:
    # PTB-XL: sex column is integer (0=male, 1=female per data dictionary)
    try:
        iv = int(v)
    except (TypeError, ValueError):
        return None
    return "M" if iv == 0 else ("F" if iv == 1 else None)


def _age_int(v: Any) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------


def load_scp_descriptions(scp_statements_csv: str | Path) -> dict[str, str]:
    df = pd.read_csv(scp_statements_csv, index_col=0)
    return {str(k): str(v) for k, v in df["description"].items()}


def load_database(ptbxl_database_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(ptbxl_database_csv, index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(_parse_scp_codes_col)
    return df


def parse_record(
    row: pd.Series,
    *,
    dataset_root: Path,
    scp_descriptions: dict[str, str],
    sampling: str = "hr",  # "hr"=500Hz or "lr"=100Hz
    compute_features: bool = True,
) -> PTBXLRecord:
    filename_col = "filename_hr" if sampling == "hr" else "filename_lr"
    rel_path = row[filename_col]
    wfdb_path = str(dataset_root / rel_path)

    sig, meta = wfdb.rdsamp(wfdb_path)
    fs = int(meta["fs"])
    # WFDB returns shape (n_samples, 12), lead order per meta["sig_name"].
    sig_names = [n.upper() for n in meta["sig_name"]]
    # Normalize names for the extractor: "AVR" -> "aVR" etc.
    name_map = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}
    leads_dict: dict[str, np.ndarray] = {}
    for i, nm in enumerate(sig_names):
        canonical = name_map.get(nm, nm)
        leads_dict[canonical] = sig[:, i].astype(np.float64)

    features: dict[str, float] = {}
    if compute_features:
        try:
            features.update(extract_waveform_features(leads_dict, fs=fs))
        except Exception:
            pass

    # Heart axis from the CSV as a coarse fallback when we fail to delineate.
    # It's a categorical string (e.g. "NORMAL", "LAD", "RAD") - we don't
    # convert to degrees to avoid spurious precision.

    scp = row["scp_codes"] if isinstance(row["scp_codes"], dict) else {}
    stmts = _scp_to_statements(scp, scp_descriptions)
    flags = _derive_flags_from_scp(scp)

    ecg_id = int(row.name) if row.name is not None else None
    patient_id = str(row["patient_id"]) if pd.notna(row.get("patient_id")) else None

    return PTBXLRecord(
        ecg_id=str(ecg_id),
        patient_id=patient_id,
        sex=_sex_str(row.get("sex")),
        age_years=_age_int(row.get("age")),
        acquisition_datetime=str(row.get("recording_date")) if pd.notna(row.get("recording_date")) else None,
        features=features,
        ge_statements=stmts,
        ge_flags=flags,
        ptbxl_ecg_id=ecg_id,
        scp_codes_raw=scp,
        report=str(row.get("report")) if pd.notna(row.get("report")) else None,
        strat_fold=int(row["strat_fold"]) if pd.notna(row.get("strat_fold")) else None,
        heart_axis_str=str(row["heart_axis"]) if pd.notna(row.get("heart_axis")) else None,
        device=str(row["device"]) if pd.notna(row.get("device")) else None,
    )


def iter_records(
    dataset_root: str | Path,
    *,
    sampling: str = "hr",
    folds: list[int] | None = None,
    n: int | None = None,
    random_state: int | None = 42,
    compute_features: bool = True,
) -> Iterator[PTBXLRecord]:
    """Stream PTB-XL records, optionally restricted to specified stratification
    folds (e.g. folds=[10] for the final test set) and capped at `n` records.

    If n is set and random_state is set, records are randomly sub-sampled for
    determinism instead of taking the first-n rows.
    """
    root = Path(dataset_root)
    db = load_database(root / "ptbxl_database.csv")
    scp_desc = load_scp_descriptions(root / "scp_statements.csv")

    if folds is not None:
        db = db[db["strat_fold"].isin(folds)]
    if n is not None and random_state is not None and len(db) > n:
        db = db.sample(n=n, random_state=random_state)
    elif n is not None:
        db = db.head(n)

    for _, row in db.iterrows():
        try:
            yield parse_record(
                row,
                dataset_root=root,
                scp_descriptions=scp_desc,
                sampling=sampling,
                compute_features=compute_features,
            )
        except Exception as e:  # noqa: BLE001
            # Keep going on single-record failures; caller can inspect None-features
            yield PTBXLRecord(
                ecg_id=str(row.name),
                patient_id=None,
                sex=None,
                age_years=None,
                acquisition_datetime=None,
                features={},
                ge_statements=[{"text": f"[load_error] {e}", "code": None, "reason": None}],
                ge_flags={},
                ptbxl_ecg_id=int(row.name) if row.name is not None else None,
                scp_codes_raw={},
            )


__all__ = [
    "PTBXLRecord",
    "SCP_KEYWORD_EXPANSIONS",
    "iter_records",
    "parse_record",
    "load_database",
    "load_scp_descriptions",
]
