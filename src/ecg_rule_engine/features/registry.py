"""Canonical ECG feature registry.

Single source of truth for feature names, units, and per-lead expansions,
shared by both the DSL validator (rules must reference known features) and
the Sapphire XML parser (which populates the same column names).

Units convention:
- Amplitudes:       mV       (e.g. R_aVL_mV)
- Durations/intervals: ms    (e.g. QRS_ms, QT_ms, PR_ms)
- ST offsets:       mV       (e.g. STJ_V2_mV)
- Rates:            bpm      (ventricular_rate_bpm, atrial_rate_bpm)
- Axes:             deg      (QRS_axis_deg, P_axis_deg, T_axis_deg)
- Booleans (flags): 0/1      (rhythm_afib, paced, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LEADS_12: tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

LIMB_LEADS: tuple[str, ...] = ("I", "II", "III", "aVR", "aVL", "aVF")
PRECORDIAL_LEADS: tuple[str, ...] = ("V1", "V2", "V3", "V4", "V5", "V6")

FeatureKind = Literal["amplitude", "duration", "st_offset", "rate", "axis", "flag", "other"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    kind: FeatureKind
    description: str


def _per_lead(prefix: str, unit: str, kind: FeatureKind, desc_tpl: str) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name=f"{prefix}_{lead}_{unit_suffix(unit)}",
            unit=unit,
            kind=kind,
            description=desc_tpl.format(lead=lead),
        )
        for lead in LEADS_12
    ]


def unit_suffix(unit: str) -> str:
    return {"mV": "mV", "ms": "ms", "bpm": "bpm", "deg": "deg", "flag": "flag"}[unit]


_GLOBAL_FEATURES: list[FeatureSpec] = [
    # Global intervals
    FeatureSpec("QRS_ms", "ms", "duration", "Global QRS duration"),
    FeatureSpec("QT_ms", "ms", "duration", "Global QT interval"),
    FeatureSpec("QTc_Bazett_ms", "ms", "duration", "QT corrected, Bazett"),
    FeatureSpec("QTc_Fridericia_ms", "ms", "duration", "QT corrected, Fridericia"),
    FeatureSpec("QTc_Framingham_ms", "ms", "duration", "QT corrected, Framingham"),
    FeatureSpec("PR_ms", "ms", "duration", "PR interval"),
    FeatureSpec("P_ms", "ms", "duration", "P-wave duration"),
    FeatureSpec("RR_ms", "ms", "duration", "Mean RR interval"),
    FeatureSpec("PP_ms", "ms", "duration", "Mean PP interval"),
    # Rates
    FeatureSpec("ventricular_rate_bpm", "bpm", "rate", "Ventricular rate"),
    FeatureSpec("atrial_rate_bpm", "bpm", "rate", "Atrial rate"),
    # Axes
    FeatureSpec("P_axis_deg", "deg", "axis", "Frontal-plane P-wave axis"),
    FeatureSpec("QRS_axis_deg", "deg", "axis", "Frontal-plane QRS axis"),
    FeatureSpec("T_axis_deg", "deg", "axis", "Frontal-plane T-wave axis"),
    # Rhythm / conduction flags (from GE interpretation or upstream classifier)
    FeatureSpec("rhythm_afib_flag", "flag", "flag", "Atrial fibrillation rhythm flag"),
    FeatureSpec("rhythm_aflutter_flag", "flag", "flag", "Atrial flutter rhythm flag"),
    FeatureSpec("rhythm_sinus_flag", "flag", "flag", "Sinus rhythm flag"),
    FeatureSpec("paced_flag", "flag", "flag", "Ventricular pacing flag"),
    FeatureSpec("wpw_flag", "flag", "flag", "Wolff-Parkinson-White pre-excitation flag"),
    FeatureSpec("lbbb_flag", "flag", "flag", "Complete LBBB flag"),
    FeatureSpec("rbbb_flag", "flag", "flag", "Complete RBBB flag"),
    FeatureSpec("ilbbb_flag", "flag", "flag", "Incomplete LBBB stated flag"),
    FeatureSpec("irbbb_flag", "flag", "flag", "Incomplete RBBB stated flag"),
    FeatureSpec("lafb_flag", "flag", "flag", "Left anterior fascicular block stated flag"),
    FeatureSpec("lpfb_flag", "flag", "flag", "Left posterior fascicular block stated flag"),
    FeatureSpec("ivcb_flag", "flag", "flag", "Intraventricular conduction block stated flag (IVCD/ILBBB/IRBBB proxy)"),
    FeatureSpec("rvh_flag", "flag", "flag", "Right ventricular hypertrophy stated flag"),
    # Sex indicators auto-populated by the evaluator from EvalContext.sex; lets
    # sex-conditional expressions be written directly in the DSL.
    FeatureSpec("sex_M_flag", "flag", "flag", "1 if sex==M else 0 (auto-populated)"),
    FeatureSpec("sex_F_flag", "flag", "flag", "1 if sex==F else 0 (auto-populated)"),
    # Age in years, auto-populated from EvalContext.age_years when available.
    FeatureSpec("age_years", "other", "other", "Patient age in years (auto-populated)"),
]

_PER_LEAD_FEATURES: list[FeatureSpec] = (
    _per_lead("R", "mV", "amplitude", "R-wave amplitude in lead {lead}")
    + _per_lead("S", "mV", "amplitude", "S-wave amplitude (absolute) in lead {lead}")
    + _per_lead("Q", "mV", "amplitude", "Q-wave amplitude (absolute) in lead {lead}")
    + _per_lead("P", "mV", "amplitude", "P-wave amplitude in lead {lead}")
    + _per_lead("T", "mV", "amplitude", "T-wave amplitude in lead {lead}")
    + _per_lead("STJ", "mV", "st_offset", "ST offset at J-point in lead {lead}")
    + _per_lead("STM", "mV", "st_offset", "ST offset at mid-ST in lead {lead}")
    + _per_lead("QRS", "ms", "duration", "QRS duration in lead {lead}")
    + _per_lead("Q", "ms", "duration", "Q-wave duration in lead {lead}")
)


def _build_registry() -> dict[str, FeatureSpec]:
    reg: dict[str, FeatureSpec] = {}
    for f in _GLOBAL_FEATURES + _PER_LEAD_FEATURES:
        if f.name in reg:
            raise RuntimeError(f"Duplicate feature in registry: {f.name}")
        reg[f.name] = f
    return reg


REGISTRY: dict[str, FeatureSpec] = _build_registry()


def is_known_feature(name: str) -> bool:
    return name in REGISTRY


def feature_unit(name: str) -> str:
    return REGISTRY[name].unit


def all_feature_names() -> list[str]:
    return sorted(REGISTRY.keys())


__all__ = [
    "LEADS_12",
    "LIMB_LEADS",
    "PRECORDIAL_LEADS",
    "FeatureSpec",
    "REGISTRY",
    "is_known_feature",
    "feature_unit",
    "all_feature_names",
]
