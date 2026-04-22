"""Parse GE Sapphire 3 ECG XML exports into a flat feature DataFrame.

Maps XML elements to the canonical names defined in `features.registry`.
Units are normalized:
- Per-lead amplitudes: uV in XML -> mV in output (÷1000)
- Durations, intervals: ms
- Rates: bpm
- Axes: degrees

Lead names are normalized from GE's "AVR/AVL/AVF" to the canonical "aVR/aVL/aVF".

Also extracts GE's own interpretation output (statements + injury-class flags)
for Phase 5 cross-check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from lxml import etree

NS = {"s": "urn:ge:sapphire:sapphire_3"}

# XML global num tag -> canonical feature name
_GLOBAL_MAP: dict[str, str] = {
    "QRS_Duration": "QRS_ms",
    "QT_Interval": "QT_ms",
    "PR_Interval": "PR_ms",
    "qtcBazettCorrection": "QTc_Bazett_ms",
    "qtcFredericiaCorrection": "QTc_Fridericia_ms",
    "qtcFraminghamCorrection": "QTc_Framingham_ms",
    "P_Axis": "P_axis_deg",
    "R_Axis": "QRS_axis_deg",
    "T_Axis": "T_axis_deg",
    "ventricularRate": "ventricular_rate_bpm",
    "atrialRate": "atrial_rate_bpm",
}

# XML per-lead tag -> (canonical prefix, canonical unit, scale factor applied to the raw V)
_PERLEAD_MAP: dict[str, tuple[str, str, float]] = {
    "R_Amp": ("R", "mV", 0.001),
    "S_Amp": ("S", "mV", 0.001),
    "Q_Amp": ("Q", "mV", 0.001),
    "P_Amp": ("P", "mV", 0.001),
    "T_Amp": ("T", "mV", 0.001),
    "STJ_Amp": ("STJ", "mV", 0.001),
    "STM_Amp": ("STM", "mV", 0.001),
    "Q_Dur": ("Q", "ms", 1.0),
}

# GE sometimes writes AVR/AVL/AVF — normalize to aVR/aVL/aVF
_LEAD_NORMALIZE = {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}

# GE's <statement V="..."> categories we want to surface as explicit flags
# (substring match, case-insensitive). This is our first-cut mapping; users can
# override via `statement_flag_map` when calling `parse_dir`.
DEFAULT_STATEMENT_FLAG_MAP: dict[str, list[str]] = {
    "rhythm_afib_flag": ["atrial fibrillation"],
    "rhythm_aflutter_flag": ["atrial flutter"],
    "rhythm_sinus_flag": ["sinus rhythm"],
    "paced_flag": ["paced", "pacemaker"],
    "wpw_flag": ["wpw", "wolff-parkinson-white", "pre-excitation", "preexcitation"],
    "lbbb_flag": ["left bundle branch block", "lbbb"],
    "rbbb_flag": ["right bundle branch block", "rbbb"],
}


@dataclass
class SapphireRecord:
    """One parsed ECG."""
    ecg_id: str
    patient_id: str | None
    sex: str | None
    age_years: int | None
    acquisition_datetime: str | None
    features: dict[str, float] = field(default_factory=dict)
    ge_statements: list[dict[str, Any]] = field(default_factory=list)
    ge_flags: dict[str, int] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "ecg_id": self.ecg_id,
            "patient_id": self.patient_id,
            "sex": self.sex,
            "age_years": self.age_years,
            "acquisition_datetime": self.acquisition_datetime,
        }
        row.update(self.features)
        row.update(self.ge_flags)
        row["ge_statements_joined"] = " | ".join(
            s.get("text", "") for s in self.ge_statements if s.get("text")
        )
        return row


def _as_float(elem: etree._Element | None) -> float | None:
    if elem is None:
        return None
    v = elem.get("V")
    if v is None or v == "" or v == "-32768":  # INV sentinel in GE XML
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _first(root: etree._Element, xpath: str) -> etree._Element | None:
    res = root.xpath(xpath, namespaces=NS)
    return res[0] if res else None


def _parse_globals(root: etree._Element) -> dict[str, float]:
    out: dict[str, float] = {}
    global_elem = _first(root, ".//s:medianTemplate/s:num/s:global")
    if global_elem is None:
        return out
    for child in global_elem:
        if not isinstance(child.tag, str):
            continue
        tag = etree.QName(child.tag).localname
        if tag in _GLOBAL_MAP:
            v = _as_float(child)
            if v is not None:
                out[_GLOBAL_MAP[tag]] = v

    # Some fields (e.g. atrialRate) may live as siblings of <global> rather than inside it
    num_elem = _first(root, ".//s:medianTemplate/s:num")
    if num_elem is not None:
        for child in num_elem:
            if not isinstance(child.tag, str):
                continue
            tag = etree.QName(child.tag).localname
            if tag in _GLOBAL_MAP and _GLOBAL_MAP[tag] not in out:
                v = _as_float(child)
                if v is not None:
                    out[_GLOBAL_MAP[tag]] = v

    # P-wave duration derived from P_Onset / P_Offset when present
    p_on = _as_float(global_elem.find("s:P_Onset", NS))
    p_off = _as_float(global_elem.find("s:P_Offset", NS))
    if p_on is not None and p_off is not None and p_off > p_on:
        out["P_ms"] = p_off - p_on

    return out


def _parse_per_lead(root: etree._Element) -> dict[str, float]:
    out: dict[str, float] = {}
    per_leads = root.xpath(".//s:medianTemplate/s:num/s:PerLead", namespaces=NS)
    for pl in per_leads:
        lead_raw = pl.get("lead") or pl.get("displayName")
        if not lead_raw:
            continue
        lead = _LEAD_NORMALIZE.get(lead_raw, lead_raw)
        for child in pl:
            if not isinstance(child.tag, str):
                continue
            tag = etree.QName(child.tag).localname
            if tag not in _PERLEAD_MAP:
                continue
            prefix, unit, scale = _PERLEAD_MAP[tag]
            v = _as_float(child)
            if v is None:
                continue
            out[f"{prefix}_{lead}_{unit}"] = v * scale
    return out


def _parse_statements(root: etree._Element) -> list[dict[str, Any]]:
    stmts: list[dict[str, Any]] = []
    for st in root.xpath(".//s:interpretation//s:statement", namespaces=NS):
        v = st.get("V")
        if not v:
            continue
        code_elem = st.find("s:code", NS)
        reason_elem = st.find("s:reason", NS)
        stmts.append({
            "text": v,
            "code": code_elem.get("V") if code_elem is not None else None,
            "reason": reason_elem.get("V") if reason_elem is not None else None,
        })
    return stmts


def _derive_flags(
    statements: list[dict[str, Any]],
    flag_map: dict[str, list[str]],
) -> dict[str, int]:
    texts_lc = " | ".join((s.get("text") or "").lower() for s in statements)
    flags: dict[str, int] = {}
    for flag_name, keywords in flag_map.items():
        flags[flag_name] = int(any(kw in texts_lc for kw in keywords))
    return flags


def _parse_demographics(root: etree._Element) -> tuple[str | None, str | None, int | None]:
    pid_elem = _first(
        root,
        ".//s:demographics/s:patientInfo/s:identifier[@*[local-name()='type']='ge.idType.MUSEPID']/s:id",
    )
    if pid_elem is None:
        pid_elem = _first(root, ".//s:demographics/s:patientInfo/s:identifier/s:id")
    patient_id = pid_elem.get("V") if pid_elem is not None else None

    sex_elem = _first(root, ".//s:demographics/s:patientInfo/s:gender")
    sex = sex_elem.get("V") if sex_elem is not None else None
    if sex:
        sex = sex.upper()[:1] if sex.upper()[:1] in ("M", "F") else None

    age_elem = _first(root, ".//s:demographics//s:patientAge")
    age: int | None = None
    if age_elem is not None and age_elem.get("V"):
        try:
            age = int(float(age_elem.get("V")))
        except ValueError:
            age = None
    return patient_id, sex, age


def parse_file(
    path: str | Path,
    *,
    statement_flag_map: dict[str, list[str]] | None = None,
) -> SapphireRecord:
    """Parse a single Sapphire XML into a SapphireRecord."""
    p = Path(path)
    tree = etree.parse(str(p))
    root = tree.getroot()

    patient_id, sex, age = _parse_demographics(root)

    acq_elem = _first(root, ".//s:testInfo/s:acquisitionDateTime")
    acq_dt = acq_elem.get("V") if acq_elem is not None else None

    features: dict[str, float] = {}
    features.update(_parse_globals(root))
    features.update(_parse_per_lead(root))

    stmts = _parse_statements(root)
    flag_map = statement_flag_map or DEFAULT_STATEMENT_FLAG_MAP
    flags = _derive_flags(stmts, flag_map)

    return SapphireRecord(
        ecg_id=p.stem,
        patient_id=patient_id,
        sex=sex,
        age_years=age,
        acquisition_datetime=acq_dt,
        features=features,
        ge_statements=stmts,
        ge_flags=flags,
    )


def parse_dir(
    xml_dir: str | Path,
    *,
    glob: str = "*.[xX][mM][lL]",
    statement_flag_map: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Parse every XML file in `xml_dir` into a single DataFrame (one row per ECG)."""
    d = Path(xml_dir)
    rows = []
    for p in sorted(d.glob(glob)):
        try:
            rec = parse_file(p, statement_flag_map=statement_flag_map)
            rows.append(rec.to_row())
        except Exception as e:  # noqa: BLE001 - we want to keep going
            rows.append({"ecg_id": p.stem, "parse_error": str(e)})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("ecg_id")


__all__ = [
    "SapphireRecord",
    "parse_file",
    "parse_dir",
    "DEFAULT_STATEMENT_FLAG_MAP",
]
