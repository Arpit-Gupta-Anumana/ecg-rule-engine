"""CLI wrapper: segment the 12SL PDF into per-disease text files.

Usage:
    python scripts/segment_pdf.py \
        --pdf "/Users/arpit.gupta/Downloads/12SL Physicians Guide v24_UM_2056246-007_2.pdf" \
        --out extracted/sections/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ecg_rule_engine.extraction.pdf_to_sections import segment_pdf, write_sections


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", required=True, help="Path to the 12SL Physician's Guide PDF")
    p.add_argument("--out", required=True, help="Output directory for .txt sections")
    p.add_argument("--prefix", default="3.7", help="Section prefix to extract (default: 3.7)")
    p.add_argument("--min-depth", type=int, default=3, help="Minimum section depth")
    args = p.parse_args()

    sections = segment_pdf(args.pdf, prefix=args.prefix, min_depth=args.min_depth)
    print(f"Found {len(sections)} sections under {args.prefix} (depth >= {args.min_depth}).")
    for s in sections:
        print(f"  {s.section_id:<12} p.{s.start_page:>3}-{s.end_page:<3}  {s.title}")
    out_paths = write_sections(sections, Path(args.out))
    print(f"\nWrote {len(out_paths)} files to {args.out}")


if __name__ == "__main__":
    main()
