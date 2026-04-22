"""Build a structured rule catalog from extracted/sections/*.txt.

Goal: For every §3.7.* section, produce a dict containing the verbatim
excerpt plus parsed fields so downstream tooling / YAML authoring can cite
the PDF exactly.

Output (deterministic):
  extracted/catalog/catalog.yaml     - all sections, machine-readable
  extracted/catalog/catalog.md       - human-readable index with excerpts
  extracted/catalog/thresholds.csv   - flat table of every numeric threshold
                                        found (with lead/sex context when
                                        detectable), one row per (section, n, unit)
  extracted/catalog/skip_conditions.csv - one row per detected "skipped/suppress"
                                        clause

Nothing is invented:
  - Numeric thresholds are pulled by regex only from verbatim section text.
  - Units (µV, mV, ms, bpm, years, deg) are preserved as written.
  - Raw context excerpts are stored alongside so any downstream use retains
    the exact surrounding wording and can be hand-verified.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ROOT / "extracted" / "sections"
OUT = ROOT / "extracted" / "catalog"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

ACRONYM_RE = re.compile(r"^\s*Acronyms?:\s*([^\n]+)", re.IGNORECASE | re.MULTILINE)
PAGES_RE = re.compile(r"^# Pages:\s*([0-9]+)(?:–|-)([0-9]+)", re.MULTILINE)
TITLE_RE = re.compile(r"^# Section:\s*([^\n]+)", re.MULTILINE)

# Numbers + units. Order matters (longer units first).
UNIT_ALTERNATIVES = [
    "µV", "microvolts", "microvolt",
    "millivolts", "millivolt", "mV",
    "milliseconds", "millisecond", "msec", "ms",
    "bpm", "beats per minute", "beats/minute",
    "degrees", "deg", "°",
    "years", "year", "y/o", "yrs", "yr",
]
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)(?:\s*)(" + "|".join(re.escape(u) for u in UNIT_ALTERNATIVES) + r")(?![A-Za-z])",
    re.IGNORECASE,
)

# Section-skip / suppression clauses
SKIP_PATTERNS = [
    re.compile(r"(skipped if[^.]*\.)", re.IGNORECASE),
    re.compile(r"(suppress[^.]*\.)",     re.IGNORECASE),
    re.compile(r"(overridden[^.]*\.)",    re.IGNORECASE),
    re.compile(r"(not stated if[^.]*\.)", re.IGNORECASE),
]

BULLET_RE = re.compile(r"^(?:[•\-\*]|\d+[.)])\s+(.+)$", re.MULTILINE)

# A conservative leads vocabulary
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6", "V4r")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ThresholdHit:
    section: str
    value: float
    unit_raw: str
    unit_canonical: str
    context: str  # ~160 char slice around the hit


@dataclass
class SkipHit:
    section: str
    text: str


@dataclass
class SectionCatalog:
    section: str            # e.g. 3.7.2.8.1.2
    title: str              # e.g. LEFT BUNDLE BRANCH BLOCK
    pages_first: int | None
    pages_last: int | None
    path: str               # path to verbatim text
    acronyms: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    thresholds: list[ThresholdHit] = field(default_factory=list)
    skip_conditions: list[SkipHit] = field(default_factory=list)
    raw_excerpt: str = ""


# ---------------------------------------------------------------------------
# Per-section parser
# ---------------------------------------------------------------------------

UNIT_CANONICAL = {
    "µV": "uV", "microvolts": "uV", "microvolt": "uV",
    "mV": "mV", "millivolts": "mV", "millivolt": "mV",
    "ms": "ms", "msec": "ms", "milliseconds": "ms", "millisecond": "ms",
    "bpm": "bpm", "beats per minute": "bpm", "beats/minute": "bpm",
    "deg": "deg", "degrees": "deg", "°": "deg",
    "years": "years", "year": "years", "yrs": "years", "yr": "years", "y/o": "years",
}


def parse_section(path: Path) -> SectionCatalog:
    text = path.read_text()
    m_title = TITLE_RE.search(text)
    full_title = m_title.group(1).strip() if m_title else path.stem
    # "3.7.2.8.1.2 — LEFT BUNDLE BRANCH BLOCK"
    if "—" in full_title:
        section, title = [s.strip() for s in full_title.split("—", 1)]
    else:
        section, title = full_title, full_title
    pages_first = pages_last = None
    m_pages = PAGES_RE.search(text)
    if m_pages:
        pages_first = int(m_pages.group(1))
        pages_last = int(m_pages.group(2))

    # Strip the header block; keep body only for threshold mining
    body = text.split("Source: GE 12SL Physician's Guide", 1)[-1].strip()

    # Acronyms
    acronyms: list[str] = []
    m_acr = ACRONYM_RE.search(body)
    if m_acr:
        acronyms = [a.strip() for a in re.split(r"[,;]", m_acr.group(1)) if a.strip()]

    # Bullets (•, -, 1., etc.)
    bullets = [b.strip() for b in BULLET_RE.findall(body)]

    # Thresholds
    thr: list[ThresholdHit] = []
    for m in UNIT_RE.finditer(body):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        unit_raw = m.group(2)
        unit_can = UNIT_CANONICAL.get(unit_raw.lower(), UNIT_CANONICAL.get(unit_raw, unit_raw))
        a = max(0, m.start() - 80)
        b = min(len(body), m.end() + 80)
        ctx = re.sub(r"\s+", " ", body[a:b]).strip()
        thr.append(ThresholdHit(
            section=section,
            value=val,
            unit_raw=unit_raw,
            unit_canonical=unit_can,
            context=ctx,
        ))

    # Skip/suppress/override clauses
    skips: list[SkipHit] = []
    for pat in SKIP_PATTERNS:
        for m in pat.finditer(body):
            skips.append(SkipHit(
                section=section,
                text=re.sub(r"\s+", " ", m.group(1)).strip(),
            ))

    return SectionCatalog(
        section=section,
        title=title,
        pages_first=pages_first,
        pages_last=pages_last,
        path=str(path.relative_to(ROOT)),
        acronyms=acronyms,
        bullets=bullets,
        thresholds=thr,
        skip_conditions=skips,
        raw_excerpt=body,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def section_sort_key(sec: str) -> tuple[int, ...]:
    parts = sec.split(".")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    return tuple(out)


def main() -> None:
    files = sorted(SECTIONS.glob("*.txt"))
    cats: list[SectionCatalog] = [parse_section(p) for p in files]
    cats.sort(key=lambda c: section_sort_key(c.section))

    # --- catalog.yaml ---
    payload = []
    for c in cats:
        d = asdict(c)
        d["thresholds"] = [asdict(t) for t in c.thresholds]
        d["skip_conditions"] = [asdict(s) for s in c.skip_conditions]
        payload.append(d)
    (OUT / "catalog.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, width=100)
    )

    # --- thresholds.csv ---
    with (OUT / "thresholds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "title", "value", "unit_raw", "unit_canonical", "context"])
        for c in cats:
            for t in c.thresholds:
                w.writerow([c.section, c.title, t.value, t.unit_raw, t.unit_canonical, t.context])

    # --- skip_conditions.csv ---
    with (OUT / "skip_conditions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "title", "clause"])
        for c in cats:
            for s in c.skip_conditions:
                w.writerow([c.section, c.title, s.text])

    # --- catalog.md (human readable index) ---
    md = ["# GE 12SL Physician's Guide — Rule Catalog",
          "",
          "Generated from `extracted/sections/*.txt` by `scripts/build_rule_catalog.py`. "
          "Every excerpt is verbatim; no interpretation.",
          "",
          f"Total sections parsed: **{len(cats)}**",
          "",
          "| § | Title | Pages | Acronyms | #thresholds | #skip clauses |",
          "|---|---|---|---|---:|---:|"]
    for c in cats:
        pg = f"{c.pages_first}-{c.pages_last}" if c.pages_first else "?"
        md.append(
            f"| {c.section} | {c.title} | {pg} | {', '.join(c.acronyms) or '—'} | "
            f"{len(c.thresholds)} | {len(c.skip_conditions)} |"
        )
    md.append("")
    md.append("---\n\n## Section details\n")
    for c in cats:
        md.append(f"### §{c.section} — {c.title}")
        md.append(f"- Pages: {c.pages_first}-{c.pages_last}" if c.pages_first else "")
        md.append(f"- Acronyms: {', '.join(c.acronyms) or '_(none)_'}")
        if c.thresholds:
            md.append("")
            md.append("**Numeric thresholds found**")
            md.append("")
            md.append("| value | unit | context |")
            md.append("|---:|---|---|")
            for t in c.thresholds:
                # Escape pipes in context
                ctx = t.context.replace("|", "\\|")
                md.append(f"| {t.value} | {t.unit_canonical} | {ctx} |")
        if c.skip_conditions:
            md.append("")
            md.append("**Skip / suppress / override clauses**")
            for s in c.skip_conditions:
                md.append(f"- {s.text}")
        md.append("")
        md.append("**Bullets (verbatim)**")
        for b in c.bullets[:30]:
            md.append(f"- {b}")
        md.append("")
        md.append("**Verbatim excerpt**")
        md.append("")
        md.append("```")
        md.append(c.raw_excerpt.strip()[:3000])
        if len(c.raw_excerpt) > 3000:
            md.append(f"... [truncated {len(c.raw_excerpt)-3000} chars]")
        md.append("```")
        md.append("")
    (OUT / "catalog.md").write_text("\n".join(md))

    print(f"Sections: {len(cats)}")
    total_thr = sum(len(c.thresholds) for c in cats)
    total_skip = sum(len(c.skip_conditions) for c in cats)
    print(f"Thresholds: {total_thr}")
    print(f"Skip clauses: {total_skip}")
    print(f"Outputs -> {OUT}")


if __name__ == "__main__":
    main()
