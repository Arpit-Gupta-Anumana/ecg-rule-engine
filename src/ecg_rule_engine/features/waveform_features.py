"""Extract per-lead amplitudes, ST offsets, and Q durations from raw 12-lead ECG.

This replaces the pre-computed per-lead features GE's Sapphire XML ships directly,
for data sources that only give us raw samples (e.g. the JSON parts export).

Approach (v2, delineation-based):
1. Clean lead II and detect R-peaks (neurokit2 `ecg_peaks`).
2. Delineate lead II using neurokit2's DWT delineator -> per-beat P/Q/R/S/T
   onsets/peaks/offsets. Average the per-beat offsets (relative to each R-peak)
   to get ONE robust set of offsets we apply to every lead.
3. For each lead, for each beat, measure amplitudes and ST offsets relative to
   a PR-segment baseline. Take the MEDIAN across beats to reject artifacts
   (PVCs, noise) without being biased by a bad beat.

Units:
- Input  : samples already in millivolts (JSON dataset stores mV).
- Output : mV (amplitudes) and ms (Q_duration).

Limitations:
- The J-point is taken from lead II's QRS offset. If a particular lead has a
  substantially later QRS offset (e.g. V1 in RBBB), STJ on that lead will be
  measured slightly too early. The rules in this repo compensate by using
  thresholds calibrated against real ECGs, and STJ is still dominated by the
  baseline shift of the ST segment, which is what the rules test.
- No attempt is made to detect QRS notching, ST slope, or fragmented QRS.
  Those features are not present in the feature registry.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import neurokit2 as nk


LEADS_12 = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def _as_array(x: Sequence[float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _clean_signal(sig: np.ndarray, fs: int) -> np.ndarray:
    try:
        return nk.ecg_clean(sig, sampling_rate=fs, method="neurokit")
    except Exception:
        return sig


def _detect_r_peaks(lead_ii: np.ndarray, fs: int) -> np.ndarray:
    try:
        _, info = nk.ecg_peaks(lead_ii, sampling_rate=fs, method="neurokit", correct_artifacts=True)
        return np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
    except Exception:
        return np.asarray([], dtype=int)


def _delineate(lead_ii: np.ndarray, r_peaks: np.ndarray, fs: int) -> dict[str, list[int]]:
    """Return DWT-based fiducials as sample indices (lists, one per R-peak).

    Keys we care about: ECG_P_Onsets, ECG_R_Onsets, ECG_Q_Peaks, ECG_S_Peaks,
    ECG_R_Offsets, ECG_T_Peaks, ECG_T_Offsets.
    """
    try:
        _, waves = nk.ecg_delineate(
            lead_ii,
            rpeaks=r_peaks,
            sampling_rate=fs,
            method="dwt",
        )
        return waves or {}
    except Exception:
        return {}


def _robust_offset(values: list[int], r_peaks: np.ndarray) -> int | None:
    """Average (as int) delta between each fiducial index and its R-peak.

    Delineation often returns NaN for beats it can't handle - skip those.
    Returns None if we have fewer than 2 usable beats.
    """
    deltas = []
    for v, r in zip(values, r_peaks):
        if v is None:
            continue
        try:
            vi = int(v)
        except (TypeError, ValueError):
            continue
        if vi <= 0:
            continue
        deltas.append(vi - int(r))
    if len(deltas) < 2:
        return None
    # median is more robust to outliers than mean
    return int(np.median(deltas))


def _baseline_per_beat(sig: np.ndarray, r: int, qrs_onset_off: int, fs: int) -> float:
    """Mean of 40 ms PR-segment ending 10 ms before QRS onset, clipped to signal."""
    onset = r + qrs_onset_off   # qrs_onset_off is negative
    end = max(0, onset - int(0.010 * fs))
    start = max(0, end - int(0.040 * fs))
    if end <= start + 2:
        return 0.0
    return float(np.mean(sig[start:end]))


def _measure_one_beat(
    sig: np.ndarray,
    r: int,
    offsets: dict[str, int],
    fs: int,
) -> dict[str, float] | None:
    """Measure amplitudes on one beat using lead-II-derived offsets."""
    n = len(sig)
    qrs_on  = r + offsets["qrs_on"]   # negative
    qrs_off = r + offsets["qrs_off"]  # positive
    p_on    = r + offsets.get("p_on", offsets["qrs_on"] - int(0.08 * fs))
    t_peak  = r + offsets.get("t_peak", offsets["qrs_off"] + int(0.15 * fs))

    if qrs_on < 0 or qrs_off >= n:
        return None

    bl = _baseline_per_beat(sig, r, offsets["qrs_on"], fs)

    qrs = sig[qrs_on : qrs_off + 1] - bl
    if len(qrs) < 3:
        return None

    r_local = int(np.argmax(qrs))
    r_amp = float(qrs[r_local])
    # S = most negative AFTER the lead's R peak
    after = qrs[r_local:]
    s_rel = float(np.min(after)) if len(after) else 0.0
    # Q = most negative BEFORE the lead's R peak
    before = qrs[: r_local + 1]
    q_rel = float(np.min(before)) if len(before) else 0.0

    # Q duration (contiguous negative run before R, in ms). Only meaningful if
    # there's a clearly negative Q.
    q_dur_ms = 0.0
    if q_rel < -0.01 and r_local > 1:
        i = r_local
        # Walk left through the negative run
        while i > 0 and before[i - 1] < 0:
            i -= 1
        q_start = i
        # Extend further: walk further left through any remaining negative samples
        while q_start > 0 and before[q_start - 1] < 0:
            q_start -= 1
        q_dur_ms = 1000.0 * (r_local - q_start) / fs
        q_dur_ms = min(q_dur_ms, 120.0)  # anatomic cap

    # ST offsets
    j_abs = qrs_off
    j80_abs = min(n - 1, j_abs + int(0.080 * fs))
    stj = float(sig[j_abs] - bl)
    stm = float(sig[j80_abs] - bl)

    # T amplitude: signed peak over 80..400 ms after J (we fall back to t_peak
    # if delineator gave us one)
    t_a = min(n - 1, j_abs + int(0.080 * fs))
    t_b = min(n, j_abs + int(0.400 * fs))
    if t_peak > t_a and t_peak < t_b:
        window = sig[t_a:t_b] - bl
        # Use the max |signed deflection| in this window
        idx = int(np.argmax(np.abs(window)))
        t_amp = float(window[idx])
    elif t_b > t_a + 2:
        window = sig[t_a:t_b] - bl
        idx = int(np.argmax(np.abs(window)))
        t_amp = float(window[idx])
    else:
        t_amp = 0.0

    # P amplitude: signed peak over [p_on, qrs_on)
    p_a = max(0, min(p_on, qrs_on - 2))
    p_b = max(0, qrs_on - int(0.020 * fs))
    if p_b > p_a + 2:
        window = sig[p_a:p_b] - bl
        idx = int(np.argmax(np.abs(window)))
        p_amp = float(window[idx])
    else:
        p_amp = 0.0

    return {
        "R":  max(r_amp, 0.0),
        "S":  -s_rel if s_rel < 0 else 0.0,
        "Q":  -q_rel if q_rel < 0 else 0.0,
        "Q_ms": q_dur_ms,
        "STJ": stj,
        "STM": stm,
        "T":  t_amp,
        "P":  p_amp,
    }


def _median_measurement(per_beat: list[dict[str, float]]) -> dict[str, float]:
    """Median across beats for every field. Empty -> empty."""
    if not per_beat:
        return {}
    keys = per_beat[0].keys()
    out: dict[str, float] = {}
    for k in keys:
        vals = [b[k] for b in per_beat if k in b and b[k] == b[k]]  # drop NaN
        if vals:
            out[k] = float(np.median(vals))
        else:
            out[k] = 0.0
    return out


def extract_waveform_features(
    leads: dict[str, Sequence[float]],
    fs: int = 1000,
) -> dict[str, float]:
    """Top-level entry. Returns a flat dict matching the feature registry:
        R_V1_mV, S_V1_mV, Q_V1_mV, Q_V1_ms, P_V1_mV, T_V1_mV,
        STJ_V1_mV, STM_V1_mV,
        ... for every lead.
    Plus `RR_ms` (mean).
    Returns {} if delineation fails.
    """
    if "II" not in leads:
        return {}

    ii_raw = _as_array(leads["II"])
    ii = _clean_signal(ii_raw, fs)
    r_peaks = _detect_r_peaks(ii, fs)
    if len(r_peaks) < 2:
        return {}

    waves = _delineate(ii, r_peaks, fs)
    qrs_on_off   = _robust_offset(waves.get("ECG_R_Onsets", []),  r_peaks)
    qrs_off_off  = _robust_offset(waves.get("ECG_R_Offsets", []), r_peaks)
    p_on_off     = _robust_offset(waves.get("ECG_P_Onsets", []),  r_peaks)
    t_peak_off   = _robust_offset(waves.get("ECG_T_Peaks", []),   r_peaks)

    # Sanity-clip implausible DWT outputs. A QRS onset more than 100 ms before
    # R is wrong (normal QRS duration is 80-120 ms, with R roughly centered).
    # Likewise a QRS offset more than 140 ms after R is implausible even for
    # wide bundle-branch-blocks.
    if qrs_on_off is not None and not (-int(0.100 * fs) <= qrs_on_off <= -int(0.015 * fs)):
        qrs_on_off = None
    if qrs_off_off is not None and not (int(0.015 * fs) <= qrs_off_off <= int(0.140 * fs)):
        qrs_off_off = None
    if p_on_off is not None and qrs_on_off is not None and p_on_off >= qrs_on_off:
        p_on_off = None

    # Fallbacks if delineation is partially unavailable or was rejected above
    if qrs_on_off is None:
        qrs_on_off = -int(0.045 * fs)   # 45 ms before R
    if qrs_off_off is None:
        qrs_off_off = int(0.055 * fs)   # 55 ms after R -> ~100 ms QRS
    if p_on_off is None:
        p_on_off = qrs_on_off - int(0.120 * fs)
    if t_peak_off is None:
        t_peak_off = qrs_off_off + int(0.160 * fs)

    offsets = {
        "qrs_on":  qrs_on_off,
        "qrs_off": qrs_off_off,
        "p_on":    p_on_off,
        "t_peak":  t_peak_off,
    }

    out: dict[str, float] = {}
    for lead in LEADS_12:
        if lead not in leads:
            continue
        sig = _clean_signal(_as_array(leads[lead]), fs)
        per_beat = []
        for r in r_peaks:
            m = _measure_one_beat(sig, int(r), offsets, fs)
            if m is not None:
                per_beat.append(m)
        if not per_beat:
            continue
        med = _median_measurement(per_beat)
        out[f"R_{lead}_mV"]  = med.get("R", 0.0)
        out[f"S_{lead}_mV"]  = med.get("S", 0.0)
        out[f"Q_{lead}_mV"]  = med.get("Q", 0.0)
        out[f"Q_{lead}_ms"]  = med.get("Q_ms", 0.0)
        out[f"P_{lead}_mV"]  = med.get("P", 0.0)
        out[f"T_{lead}_mV"]  = med.get("T", 0.0)
        out[f"STJ_{lead}_mV"] = med.get("STJ", 0.0)
        out[f"STM_{lead}_mV"] = med.get("STM", 0.0)

    # Global interval features derived from the lead-II fiducials
    if qrs_off_off is not None and qrs_on_off is not None:
        out["QRS_ms"] = 1000.0 * (qrs_off_off - qrs_on_off) / fs
    if p_on_off is not None and qrs_on_off is not None and qrs_on_off > p_on_off:
        out["PR_ms"] = 1000.0 * (qrs_on_off - p_on_off) / fs
    t_off_off = _robust_offset(waves.get("ECG_T_Offsets", []), r_peaks)
    if t_off_off is not None and qrs_on_off is not None and t_off_off > qrs_on_off:
        out["QT_ms"] = 1000.0 * (t_off_off - qrs_on_off) / fs

    if len(r_peaks) >= 2:
        rr_ms = 1000.0 * float(np.mean(np.diff(r_peaks))) / fs
        out["RR_ms"] = rr_ms
        out["ventricular_rate_bpm"] = 60000.0 / rr_ms if rr_ms > 0 else 0.0
        if "QT_ms" in out and rr_ms > 0:
            out["QTc_Bazett_ms"] = out["QT_ms"] / ((rr_ms / 1000.0) ** 0.5)

    # Frontal-plane axes from lead-I / lead-aVF amplitudes.
    # axis = atan2(aVF, I) * 180/pi. QRS uses net deflection R - Q - S.
    def _axis(amp_I: float | None, amp_aVF: float | None) -> float | None:
        if amp_I is None or amp_aVF is None:
            return None
        return float(np.degrees(np.arctan2(amp_aVF, amp_I)))

    qrs_I = (out.get("R_I_mV", 0.0) or 0.0) - (out.get("Q_I_mV", 0.0) or 0.0) \
        - (out.get("S_I_mV", 0.0) or 0.0)
    qrs_aVF = (out.get("R_aVF_mV", 0.0) or 0.0) - (out.get("Q_aVF_mV", 0.0) or 0.0) \
        - (out.get("S_aVF_mV", 0.0) or 0.0)
    qrs_axis = _axis(qrs_I, qrs_aVF)
    if qrs_axis is not None:
        out["QRS_axis_deg"] = qrs_axis
    p_axis = _axis(out.get("P_I_mV"), out.get("P_aVF_mV"))
    if p_axis is not None:
        out["P_axis_deg"] = p_axis
    t_axis = _axis(out.get("T_I_mV"), out.get("T_aVF_mV"))
    if t_axis is not None:
        out["T_axis_deg"] = t_axis

    return out


__all__ = ["extract_waveform_features", "LEADS_12"]
