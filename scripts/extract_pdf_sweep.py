"""Full-PDF sweep: re-extract everything, then slice by §3.7.x headings.

This augments `extracted/sections/` (which we trust) with a flat per-page dump
`extracted/pages/<NNN>.txt` so we can audit whether the section-based slicing
missed any numeric rule that straddles a heading. We also produce:

  extracted/full_body_3_7.txt   - concatenated text of §3.7 only
  extracted/toc.txt             - page numbers for every §3.7 heading we detect

No rule text is rewritten — this script is read-only for the PDF.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
PDF = Path("/Users/arpit.gupta/Downloads/12SL Physicians Guide v24_UM_2056246-007_2.pdf")
PAGES_DIR = ROOT / "extracted" / "pages"
OUT_BODY = ROOT / "extracted" / "full_body_3_7.txt"
OUT_TOC = ROOT / "extracted" / "toc.txt"
HEADING_RE = re.compile(r"^\s*(3\.7(?:\.\d+){0,4})\s+([A-Z][^\n]{2,})")


def main() -> None:
    PAGES_DIR.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(str(PDF)) as pdf:
        n_pages = len(pdf.pages)
        print(f"PDF has {n_pages} pages")

        toc_rows: list[tuple[str, int, str]] = []
        body_lines: list[str] = []
        in_37 = False

        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            (PAGES_DIR / f"{i:03d}.txt").write_text(txt)
            for line in txt.splitlines():
                m = HEADING_RE.match(line.strip())
                if m:
                    sec, title = m.group(1), m.group(2).strip()
                    toc_rows.append((sec, i, title))
                    if sec.startswith("3.7"):
                        in_37 = True
            if in_37:
                body_lines.append(f"\n===== page {i:3d} =====")
                body_lines.append(txt)
            if i > 20 and i % 25 == 0:
                print(f"  processed page {i}/{n_pages}")

    OUT_TOC.write_text("\n".join(f"{s}\tp{p}\t{t}" for s, p, t in toc_rows))
    OUT_BODY.write_text("\n".join(body_lines))
    print(f"TOC entries: {len(toc_rows)}  ->  {OUT_TOC}")
    print(f"Body length: {len(''.join(body_lines))} chars  ->  {OUT_BODY}")


if __name__ == "__main__":
    main()
