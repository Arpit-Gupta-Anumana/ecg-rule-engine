"""One-time: precompute waveform + SCP-code features for ALL PTB-XL records and
cache to a parquet file. Subsequent rule-engine evaluations then just read the
parquet, which is ~100x faster than re-reading WFDB + re-delineating every time.

Usage:
    python scripts/cache_features_ptbxl.py                      # all 21799 records
    python scripts/cache_features_ptbxl.py --limit 500 --fold 10 # smoke test
    python scripts/cache_features_ptbxl.py --workers 4           # parallel

Output: data/ptbxl_features.parquet  (+ .tmp_chunk_*.parquet while running)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ecg_rule_engine.features.ptbxl import (
    load_database,
    load_scp_descriptions,
    parse_record,
)

ROOT = Path("/Users/arpit.gupta/Downloads/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
OUT_PATH = Path("/Users/arpit.gupta/Desktop/ecg_rule_engine/data/ptbxl_features.parquet")


def _rec_to_row(rec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ecg_id":      rec.ptbxl_ecg_id,
        "patient_id":  rec.patient_id,
        "sex":         rec.sex,
        "age_years":   rec.age_years,
        "strat_fold":  rec.strat_fold,
        "heart_axis":  rec.heart_axis_str,
        "device":      rec.device,
        "report":      rec.report,
        # SCP codes as a JSON string (parquet-friendly) plus a space-joined list
        "scp_codes_json":  json.dumps(rec.scp_codes_raw),
        "scp_codes_str":   " ".join(sorted(rec.scp_codes_raw.keys())),
        # Rhythm/conduction flags sourced from SCP codes (used by rule engine)
        **{f"ge_{k}": int(v) for k, v in rec.ge_flags.items()},
    }
    row.update(rec.features)
    return row


_db = None
_scp = None


def _worker_init(root_str: str) -> None:
    # Load the metadata tables ONCE per worker (not once per record).
    global _db, _scp
    root = Path(root_str)
    _db = load_database(root / "ptbxl_database.csv")
    _scp = load_scp_descriptions(root / "scp_statements.csv")


def _process_ecg_id(args: tuple[int, str]) -> dict[str, Any]:
    ecg_id, root_str = args
    root = Path(root_str)
    try:
        row = _db.loc[ecg_id]
        rec = parse_record(row, dataset_root=root, scp_descriptions=_scp, sampling="hr")
        return _rec_to_row(rec)
    except Exception as e:  # noqa: BLE001
        return {"ecg_id": ecg_id, "_error": str(e)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only first N records")
    ap.add_argument("--fold", type=int, default=None, help="restrict to one strat_fold")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    db = load_database(ROOT / "ptbxl_database.csv")
    if args.fold is not None:
        db = db[db["strat_fold"] == args.fold]
    if args.limit is not None:
        db = db.head(args.limit)
    ecg_ids = list(db.index)

    print(f"Processing {len(ecg_ids)} records with {args.workers} worker(s). "
          f"Out: {args.out}")

    t0 = time.time()
    work = [(ecg_id, str(ROOT)) for ecg_id in ecg_ids]
    rows: list[dict[str, Any]] = []

    if args.workers <= 1:
        _worker_init(str(ROOT))
        for a in tqdm(work, unit="ecg"):
            rows.append(_process_ecg_id(a))
    else:
        with mp.Pool(args.workers, initializer=_worker_init, initargs=(str(ROOT),)) as pool:
            for row in tqdm(pool.imap_unordered(_process_ecg_id, work, chunksize=8),
                            total=len(work), unit="ecg"):
                rows.append(row)
                if len(rows) % args.chunk_size == 0:
                    pd.DataFrame(rows).to_parquet(
                        args.out.with_suffix(".partial.parquet"), index=False
                    )

    dt = time.time() - t0
    df = pd.DataFrame(rows)
    n_err = int(df.get("_error", pd.Series(dtype=object)).notna().sum())
    print(f"\nDone in {dt/60:.1f} min. "
          f"Rows: {len(df)}, errors: {n_err}, columns: {df.shape[1]}")
    print(f"Writing -> {args.out}")
    df.to_parquet(args.out, index=False)
    # Clean up partial file
    partial = args.out.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
