"""Derive frontal-plane axes (QRS_axis_deg, P_axis_deg, T_axis_deg) from the
already-cached per-lead amplitudes and add them as new columns to
data/ptbxl_features.parquet.

The standard method is atan2(aVF_amplitude, I_amplitude) in degrees. For QRS
we use the net deflection (R - S - Q); for P and T we use the lead amplitude
directly (positive = upright).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/arpit.gupta/Desktop/ecg_rule_engine")
PARQUET = ROOT / "data" / "ptbxl_features.parquet"


def frontal_axis(amp_I: pd.Series, amp_aVF: pd.Series) -> pd.Series:
    """atan2(aVF, I) in degrees, NaN-safe."""
    rad = np.arctan2(amp_aVF.astype(float), amp_I.astype(float))
    deg = np.degrees(rad)
    # Treat NaN inputs as NaN output
    return deg.where(amp_I.notna() & amp_aVF.notna(), np.nan)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df)} rows, {df.shape[1]} cols")

    # QRS: net deflection = R - Q - S (all as positive magnitudes in our schema;
    # Q and S are negative deflections so we stored them as positive mV.)
    qrs_I   = df["R_I_mV"]   - df["Q_I_mV"]   - df["S_I_mV"]
    qrs_aVF = df["R_aVF_mV"] - df["Q_aVF_mV"] - df["S_aVF_mV"]
    df["QRS_axis_deg"] = frontal_axis(qrs_I, qrs_aVF)

    # P and T wave axes
    df["P_axis_deg"] = frontal_axis(df["P_I_mV"], df["P_aVF_mV"])
    df["T_axis_deg"] = frontal_axis(df["T_I_mV"], df["T_aVF_mV"])

    # Quick sanity summary
    for col in ("QRS_axis_deg", "P_axis_deg", "T_axis_deg"):
        s = df[col].dropna()
        print(f"{col}: n={len(s)}, mean={s.mean():.1f}, "
              f"median={s.median():.1f}, 5%={s.quantile(0.05):.1f}, "
              f"95%={s.quantile(0.95):.1f}")

    df.to_parquet(PARQUET, index=False)
    print(f"Rewrote {PARQUET} with {df.shape[1]} cols.")


if __name__ == "__main__":
    main()
