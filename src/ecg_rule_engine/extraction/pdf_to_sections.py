"""Segment the GE 12SL Physician's Guide PDF into per-disease sections.

Strategy
========

Section 3.7 ("Criteria – Rules for Interpretation") and its sub-sections are
where the disease criteria live. Subsection headings look like:

    3.7.2.5 Low Voltage QRS
    3.7.2.7 Brugada
    3.7.3.11.2 EARLY REPOLARIZATION

We detect lines starting with a dotted numeric prefix (two or more dots) at
the *top of the printed body* of a page and treat each such line as a section
boundary. Each section then collects every line from its first page until the
next section starts.

Output
======

- `segment_pdf(pdf_path)` returns a list of `PdfSection` with:
    section_id:   "3.7.2.5"
    title:        "Low Voltage QRS"
    start_page:   (1-indexed, matches the printed page + PDF's internal page)
    end_page:
    text:         full joined text of the section (from heading to next heading)

- `write_sections(sections, out_dir)` writes `<section_id>__<slug>.txt` files,
  each prefixed with a small header so the LLM extractor sees the metadata.

Note: this does NOT filter by "is this a disease"; it returns every 3.7.* leaf
section. The extraction driver later decides which ones to transcribe into DSL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

# section number like 3, 3.7, 3.7.2, 3.7.2.5, 3.7.3.11.2 — require >= 1 dot
# so we skip single-digit chapter numbers. Most disease subsections have >= 2 dots.
_SECTION_RE = re.compile(r"^(?P<num>\d+(?:\.\d+){1,5})\s+(?P<title>[A-Za-z][^\n]*?)\s*$")


@dataclass
class PdfSection:
    section_id: str
    title: str
    start_page: int  # 1-indexed
    end_page: int
    text: str

    def slug(self) -> str:
        t = re.sub(r"[^A-Za-z0-9]+", "_", self.title).strip("_").lower()
        return t or "untitled"

    def header(self) -> str:
        return (
            f"# Section: {self.section_id} — {self.title}\n"
            f"# Pages: {self.start_page}–{self.end_page}\n"
            f"# Source: GE 12SL Physician's Guide\n"
        )


def _iter_page_lines(pdf_path: str | Path):
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            for line in text.splitlines():
                yield i + 1, line.rstrip()


def segment_pdf(
    pdf_path: str | Path,
    *,
    prefix: str = "3.7",
    min_depth: int = 3,
) -> list[PdfSection]:
    """Extract all sections under `prefix` (default: section 3.7) with depth >= min_depth.

    Only lines whose section number starts with `prefix` AND have at least
    `min_depth` dotted components are kept (e.g. `3.7.2.5` has depth 4).
    """
    current: dict | None = None
    sections: list[PdfSection] = []
    buffer: list[str] = []
    for page_num, line in _iter_page_lines(pdf_path):
        m = _SECTION_RE.match(line.strip())
        if m:
            num = m.group("num")
            title = m.group("title").strip()
            # Skip obvious TOC / figure / table lines ending in a dot-page-number
            if re.search(r"\.\.+\s*\d+\s*$", title) or re.search(r"\s\d+\s*$", title) and "........." in title:
                continue
            depth = num.count(".") + 1
            if num.startswith(prefix) and depth >= min_depth:
                # close previous
                if current is not None:
                    sections.append(PdfSection(
                        section_id=current["num"],
                        title=current["title"],
                        start_page=current["start_page"],
                        end_page=page_num,
                        text="\n".join(buffer).strip(),
                    ))
                    buffer = []
                current = {"num": num, "title": title, "start_page": page_num}
                continue
        if current is not None:
            buffer.append(line)

    if current is not None:
        sections.append(PdfSection(
            section_id=current["num"],
            title=current["title"],
            start_page=current["start_page"],
            end_page=current["start_page"],  # best-effort; end updated below
            text="\n".join(buffer).strip(),
        ))

    # Set end_page of each section to start_page of next -1 (where available)
    for i, sec in enumerate(sections[:-1]):
        sections[i] = PdfSection(
            section_id=sec.section_id,
            title=sec.title,
            start_page=sec.start_page,
            end_page=max(sec.start_page, sections[i + 1].start_page - 1),
            text=sec.text,
        )

    # Deduplicate by section_id (keep the first real occurrence; TOC matches are
    # usually very short so filter those that have <200 chars of body text)
    seen: dict[str, PdfSection] = {}
    for sec in sections:
        existing = seen.get(sec.section_id)
        if existing is None or len(sec.text) > len(existing.text):
            seen[sec.section_id] = sec
    deduped = sorted(seen.values(), key=lambda s: tuple(int(x) for x in s.section_id.split(".")))
    return deduped


def write_sections(sections: list[PdfSection], out_dir: str | Path) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sec in sections:
        name = f"{sec.section_id}__{sec.slug()}.txt"
        fp = d / name
        fp.write_text(sec.header() + "\n" + sec.text, encoding="utf-8")
        written.append(fp)
    return written


__all__ = ["PdfSection", "segment_pdf", "write_sections"]
