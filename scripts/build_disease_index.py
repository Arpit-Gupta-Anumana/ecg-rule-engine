"""Build a master index mapping every §3.7 section to:
- the abnormalities / statements it defines (with acronyms),
- our current YAML rule coverage (if any),
- whether the manual section is purely-qualitative (no numeric thresholds)
  or quantitative (has thresholds in µV/mV/ms/bpm/years/deg).

Output:
  extracted/catalog/disease_index.md   - Markdown checklist
  extracted/catalog/disease_index.csv  - Flat CSV for filtering/sorting
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "extracted" / "catalog" / "catalog.yaml"
RULES = ROOT / "rules"
OUT_MD = ROOT / "extracted" / "catalog" / "disease_index.md"
OUT_CSV = ROOT / "extracted" / "catalog" / "disease_index.csv"


# Map YAML rule files -> set of §sections they implement
# (based on `manual_source` field in each rule).
def build_rule_coverage() -> dict[str, list[str]]:
    cov: dict[str, list[str]] = {}
    for rf in sorted(RULES.glob("*.yaml")):
        try:
            d = yaml.safe_load(rf.read_text())
        except Exception as exc:
            print(f"WARN: could not parse {rf.name}: {exc}")
            continue
        if not isinstance(d, dict):
            continue
        src = str(d.get("manual_source", ""))
        secs = []
        for tok in src.replace("and", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            # extract 3.7.x.y.z style
            import re
            m = re.search(r"3\.7(?:\.\d+){1,4}", tok)
            if m:
                secs.append(m.group(0))
        variants = d.get("variants") or []
        cov.setdefault(rf.name, []).extend(secs)
        for v in variants:
            if isinstance(v, dict):
                vp = v.get("manual_page")
                vsrc = v.get("manual_source", "")
                if isinstance(vsrc, str):
                    import re
                    for m in re.finditer(r"3\.7(?:\.\d+){1,4}", vsrc):
                        cov[rf.name].append(m.group(0))
    return cov


def main() -> None:
    cat = yaml.safe_load(CATALOG.read_text())
    cov_by_file = build_rule_coverage()
    # invert: section -> list[rule files that cite it]
    sec_to_rules: dict[str, list[str]] = {}
    for rf, secs in cov_by_file.items():
        for s in secs:
            sec_to_rules.setdefault(s, []).append(rf)

    rows: list[dict[str, str]] = []
    for c in cat:
        sec = c["section"]
        rules_citing = sorted(set(sec_to_rules.get(sec, [])))
        # Child coverage: if our section's ancestor was cited, we count as
        # covered too (e.g., q_wave_mi cites §3.7.2.10 which covers .10.1-.10.6).
        parts = sec.split(".")
        for i in range(1, len(parts)):
            ancestor = ".".join(parts[:i + 1])
            if ancestor == sec:
                continue
            for r in sec_to_rules.get(ancestor, []):
                if r not in rules_citing:
                    rules_citing.append(r)
        # Descendant coverage: a parent container like §3.7.2.10 should be
        # considered covered when any child §3.7.2.10.x has a rule.
        descendants = [s for s in sec_to_rules if s.startswith(sec + ".")]
        for d in descendants:
            for r in sec_to_rules[d]:
                if r not in rules_citing:
                    rules_citing.append(r)
        n_thr = len(c.get("thresholds") or [])
        n_skip = len(c.get("skip_conditions") or [])
        rows.append({
            "section": sec,
            "title": c.get("title", ""),
            "pages": f"{c.get('pages_first')}-{c.get('pages_last')}",
            "acronyms": ", ".join(c.get("acronyms") or []),
            "thresholds": str(n_thr),
            "skip_clauses": str(n_skip),
            "has_numeric": "yes" if n_thr else "no",
            "rules": ", ".join(rules_citing) or "",
            "coverage": "COVERED" if rules_citing else "MISSING",
        })

    # --- CSV ---
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- Markdown ---
    md = ["# Disease / Abnormality Coverage Index",
          "",
          "Every §3.7 subsection in the GE 12SL Physician's Guide, with extracted "
          "acronyms, threshold counts, skip-condition counts, and which YAML rule (if "
          "any) currently implements it.",
          "",
          "- **COVERED** = at least one rule file in `rules/` cites this section in "
          "`manual_source`.",
          "- **MISSING** = no YAML rule yet.",
          "- **has_numeric=no** = the manual text describes the condition qualitatively "
          "only (no µV/mV/ms/bpm/years/deg thresholds); any rule we write must document "
          "operationalization in `manual_departure_notes`.",
          "",
          "| § | Title | Pages | Acronyms | #thr | #skip | Coverage | Rule file(s) |",
          "|---|---|---|---|---:|---:|---|---|"]
    for r in rows:
        md.append(
            f"| {r['section']} | {r['title']} | {r['pages']} | {r['acronyms'] or '—'} | "
            f"{r['thresholds']} | {r['skip_clauses']} | "
            f"{'✅ ' + 'COVERED' if r['coverage'] == 'COVERED' else '⬜ MISSING'} | "
            f"{r['rules'] or '—'} |"
        )
    md.append("")

    # summary block
    covered = sum(1 for r in rows if r["coverage"] == "COVERED")
    missing = len(rows) - covered
    with_numeric = sum(1 for r in rows if r["has_numeric"] == "yes")
    md.insert(2, f"**Totals:** {len(rows)} sections — {covered} covered, {missing} missing; "
                 f"{with_numeric} have explicit numeric thresholds.\n")
    OUT_MD.write_text("\n".join(md))

    print(f"{len(rows)} sections indexed")
    print(f"Covered: {covered}  Missing: {missing}")
    print(f"-> {OUT_MD}")
    print(f"-> {OUT_CSV}")


if __name__ == "__main__":
    main()
