"""Step-by-step walkthrough of the pipeline on a single real ECG.

Intended as a teaching aid: for each pipeline stage, prints the raw inputs,
the relevant code path that handled them, and the outputs, so you can read
this top-to-bottom and see exactly how a manual criterion becomes a firing
decision on a real recording.

Usage:
    python scripts/walkthrough_lvh.py data/sapphire_xml/lvh_demo/demo_lvh.xml
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

from lxml import etree
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table

from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
from ecg_rule_engine.engine.trace import format_trace
import pandas as pd

from ecg_rule_engine.features.sapphire_xml import parse_file

console = Console()


def _section(title: str) -> None:
    console.print()
    console.print(Rule(f"[bold cyan]{title}[/]"))


def _subsection(title: str) -> None:
    console.print(f"\n[bold yellow]- {title}[/]")


def main(xml_path: str) -> None:
    xml_file = Path(xml_path)
    rule_file = Path("rules/lvh.yaml")

    # -----------------------------------------------------------------
    # Stage 0 - Context: what are we doing?
    # -----------------------------------------------------------------
    _section("STAGE 0  -  What the pipeline does, at a glance")
    console.print(textwrap.dedent("""
    Every ECG goes through five stages:

        [bold]1.[/bold] XML  ->  Features           (features/sapphire_xml.py)
        [bold]2.[/bold] YAML ->  Disease object     (dsl/loader.py + dsl/schema.py)
        [bold]3.[/bold] Features + Disease -> fire? (engine/evaluator.py)
        [bold]4.[/bold] fire + GT  -> metrics       (eval/metrics.py)
        [bold]5.[/bold] metrics    -> markdown      (eval/report.py)

    We'll use ONE real ECG:  [green]""" + str(xml_file) + """[/green]
    And ONE disease:         [green]Left Ventricular Hypertrophy (LVH)[/green]
    """))

    # -----------------------------------------------------------------
    # Stage 1 - Raw XML: look inside the file
    # -----------------------------------------------------------------
    _section("STAGE 1  -  Look at the raw XML")

    raw = xml_file.read_text(encoding="utf-8", errors="ignore")
    console.print(f"File size: {len(raw):,} bytes (all on one long line - typical for GE Sapphire exports).")

    _subsection("Where the demographics live")
    m = re.search(r"<gender[^>]*V=\"([^\"]+)\"", raw)
    age = re.search(r"<patientAge[^>]*V=\"([^\"]+)\"", raw)
    pid = re.search(r"<id V=\"([^\"]+)\"/><\/identifier>", raw)
    console.print(f"  <gender V=\"{m.group(1) if m else '?'}\"/>")
    console.print(f"  <patientAge V=\"{age.group(1) if age else '?'}\" U=\"a\"/>")
    console.print(f"  patient_id  -> [green]{pid.group(1) if pid else '?'}[/green]")

    _subsection("Where the global measurements live (inside medianTemplate/num/global/)")
    for tag in ("QRS_Duration", "QT_Interval", "PR_Interval", "R_Axis",
                "qtcBazettCorrection", "ventricularRate"):
        mm = re.search(rf"<{tag}[^>]*V=\"([^\"]+)\"[^>]*U=\"([^\"]+)\"", raw)
        if mm:
            console.print(f"  <{tag} V=\"{mm.group(1)}\" U=\"{mm.group(2)}\"/>")

    _subsection("Where the per-lead amplitudes live (one <measurementMatrix> block per lead)")
    # Leads can use attributes like lead="aVL" or leadName="aVL"; amplitudes may
    # live inside <LeadMeasurements> / <measurementMatrix>; grab the smallest
    # window that contains both the lead label and the closest R_Amp / S_Amp.
    for lead in ("aVL", "V1", "V3", "V5", "V6"):
        pat_block = re.search(
            rf"(?:lead|leadName)=\"{lead}\"(.*?)(?:</LeadMeasurements>|</measurementMatrix>|<(?:LeadMeasurements|measurementMatrix))",
            raw, flags=re.DOTALL,
        )
        blk = pat_block.group(1) if pat_block else ""
        r = re.search(r"<R_Amp[^>]*V=\"([-\d.]+)\"[^>]*U=\"([^\"]+)\"", blk)
        s = re.search(r"<S_Amp[^>]*V=\"([-\d.]+)\"[^>]*U=\"([^\"]+)\"", blk)
        ra = f"R={r.group(1)}{r.group(2)}" if r else "R=?"
        sa = f"S={s.group(1)}{s.group(2)}" if s else "S=?"
        console.print(f"  lead=\"{lead}\":  {ra}  {sa}")
    console.print("  [dim](regex above is approximate; the parser uses lxml + XPath, see below)[/dim]")

    _subsection("Where GE's own interpretation statements live")
    stmts = re.findall(r"<statement[^>]*>.*?<leftStatement V=\"([^\"]+)\"", raw)
    if not stmts:
        stmts = re.findall(r"<leftStatement V=\"([^\"]+)\"", raw)
    for s in stmts[:15]:
        console.print(f"  - [italic]{s}[/italic]")
    console.print(f"  (total: {len(stmts)} statements)")

    # -----------------------------------------------------------------
    # Stage 2 - Parse XML into a flat feature dict
    # -----------------------------------------------------------------
    _section("STAGE 2  -  Parse XML -> features DataFrame")

    console.print(textwrap.dedent("""
    [dim]features/sapphire_xml.py[/dim] walks the XML tree with lxml and maps each
    GE element name into our canonical feature registry:

        QRS_Duration (ms)   ->  QRS_ms
        R_Amp    at aVL (uV) ->  R_aVL_mV   (divided by 1000)
        S_Amp    at V3  (uV) ->  S_V3_mV
        ...

    The output is one row per ECG (ecg_id derived from the filename), with
    every feature as a column, plus:
        patient_id, sex, age_years, ge_statements_joined, rhythm_*_flag, ...
    """))

    rec = parse_file(str(xml_file))
    df = pd.DataFrame([rec.to_row()]).set_index("ecg_id")
    console.print(f"[green]parse_file() returned a SapphireRecord; flattened to a 1-row "
                  f"DataFrame with shape {df.shape}[/green]")

    _subsection("Demographics that ended up on the row")
    for col in ("patient_id", "sex", "age_years"):
        if col in df.columns:
            console.print(f"  {col:20s} = {df.iloc[0][col]}")

    _subsection("Global measurements picked up (already converted to ms / mV / bpm)")
    for col in ["QRS_ms", "QT_ms", "QTc_ms", "PR_ms", "R_axis_deg",
                "ventricular_rate_bpm", "atrial_rate_bpm"]:
        if col in df.columns:
            v = df.iloc[0][col]
            console.print(f"  {col:25s} = {v}")

    _subsection("Per-lead amplitudes we actually need for LVH (mV)")
    needed = ["R_aVL_mV", "S_V1_mV", "S_V2_mV", "S_V3_mV",
              "R_V5_mV", "R_V6_mV", "Q_V5_mV", "Q_V6_mV"]
    tbl = Table(show_header=True, header_style="bold")
    tbl.add_column("Feature"); tbl.add_column("Value", justify="right")
    for col in needed:
        if col in df.columns:
            tbl.add_row(col, f"{df.iloc[0][col]:.3f}" if df.iloc[0][col] == df.iloc[0][col] else "NaN")
        else:
            tbl.add_row(col, "[red]missing[/red]")
    console.print(tbl)

    _subsection("GE's own statements (extracted into one string on the row)")
    stmts_col = str(df.iloc[0].get("ge_statements_joined", ""))
    for line in stmts_col.split(" | ")[:10]:
        console.print(f"  - {line}")

    # -----------------------------------------------------------------
    # Stage 3 - Parse the YAML rule into a Disease object
    # -----------------------------------------------------------------
    _section("STAGE 3  -  Parse rules/lvh.yaml -> Disease object")

    console.print(textwrap.dedent(f"""
    [dim]dsl/loader.py[/dim] reads the YAML file, and [dim]dsl/schema.py[/dim]
    (pydantic) validates every clause against the feature registry - so a
    typo like 'R_aVl_mV' is rejected at load time, not silently ignored.

    The file [green]{rule_file}[/green] contains several named criteria
    ('variants'), each a tree of all_of / any_of / threshold nodes.
    """))

    console.print(Panel(
        Syntax(rule_file.read_text(), "yaml", line_numbers=True, word_wrap=True),
        title=str(rule_file),
        border_style="blue",
    ))

    disease = load_disease_yaml(rule_file)
    console.print(f"[green]Loaded disease: {disease.disease}  with variants:[/green] "
                  + ", ".join(v.name for v in disease.variants))
    console.print(f"Exclusions (gates at the disease level): {disease.exclusions}")

    # -----------------------------------------------------------------
    # Stage 4 - Build EvalContext from the row
    # -----------------------------------------------------------------
    _section("STAGE 4  -  Build EvalContext  (features + sex + age)")

    row = df.iloc[0]
    feats = {k: float(v) for k, v in row.items()
             if isinstance(v, (int, float)) and v == v}  # drop NaNs / strings
    sex = str(row.get("sex", "U"))
    age = int(row["age_years"]) if row.get("age_years") == row.get("age_years") else None

    ctx = EvalContext(features=feats, sex=sex, age_years=age)
    eff = ctx.effective_features()
    console.print(textwrap.dedent(f"""
    [dim]engine/evaluator.py : EvalContext.effective_features()[/dim] copies the
    raw feature dict and injects three auto-populated flags derived from
    sex / age so rule expressions can reference them directly:

        sex_M_flag   = {eff.get('sex_M_flag')}
        sex_F_flag   = {eff.get('sex_F_flag')}
        age_years    = {eff.get('age_years')}

    Raw features dict: {len(feats)} numeric entries.
    """))

    # -----------------------------------------------------------------
    # Stage 5 - Evaluate the disease
    # -----------------------------------------------------------------
    _section("STAGE 5  -  Evaluate each variant  ->  disease fire?")

    result = evaluate_disease(disease, ctx)
    console.print(textwrap.dedent(f"""
    [dim]evaluate_disease()[/dim] walks every variant top-down.  For each clause
    it records a TraceNode that says:

        node kind   ('threshold', 'all_of', 'any_of', 'point_score', ...)
        fired       (True/False)
        detail      (feature name, value, op, threshold, result)
        children    (nested TraceNodes)
        missing_features  (features the rule needed but weren't in context)

    At the end it OR's the variants (unless a disease-level exclusion fired)
    and returns a [bold]DiseaseResult[/bold]:

        disease         = {result.disease}
        fired           = {result.fired}
        excluded_by     = {result.excluded_by}
        fired_variants  = {result.fired_variants()}
    """))

    _subsection("Per-variant fire traces (one trace tree per variant)")
    for vr in result.variant_results:
        header = f"{vr.variant_name}  ->  {'FIRED' if vr.fired else 'did not fire'}"
        console.print(Panel(format_trace(vr.trace),
                            title=header,
                            border_style="green" if vr.fired else "yellow",
                            subtitle="engine/trace.py : format_trace"))

    # -----------------------------------------------------------------
    # Stage 6 - Compare to GE's own statement (cross-check stage)
    # -----------------------------------------------------------------
    _section("STAGE 6  -  Cross-check: what did GE itself say?")

    ge_lvh = any("left ventricular hypertrophy" in s.lower()
                 for s in stmts_col.split(" | "))
    table = Table()
    table.add_column("Source"); table.add_column("LVH decision")
    table.add_row("Our rule engine",   "[green]POSITIVE[/green]" if result.fired else "[red]NEGATIVE[/red]")
    table.add_row("GE 12SL on-device", "[green]POSITIVE[/green]" if ge_lvh     else "[red]NEGATIVE[/red]")
    console.print(table)
    console.print(textwrap.dedent("""
    On a full dataset, [dim]eval/ge_crosscheck.py[/dim] does this for every ECG and
    reports agreement rate + Cohen's kappa between our engine and GE - a
    sanity check even without clinician labels.
    """))

    # -----------------------------------------------------------------
    # Stage 7 - What the end-to-end evaluate CLI would do next
    # -----------------------------------------------------------------
    _section("STAGE 7  -  What happens across MANY ECGs")

    console.print(textwrap.dedent("""
    When you run:

        [cyan]ecg-rule-eval evaluate --rules rules/ --xml data/sapphire_xml/
                          --gt data/gt.csv  --out out/[/cyan]

    the CLI [dim]src/ecg_rule_engine/cli.py : cmd_evaluate()[/dim] does:

        1. parse_dir(xml/)           -> one row per ECG in a DataFrame
        2. load_gt_csv(gt.csv)       -> clinician labels (one row per
                                        (ecg_id, disease))
        3. merge_features_and_gt()   -> big wide DataFrame with features,
                                        demographics, and gt__<disease> cols
        4. split_by_patient()        -> deterministic train/val/test by
                                        hashing patient_id  (never leaks
                                        a patient across splits)
        5. run_engine()              -> for every disease in rules/:
                                            evaluate_disease on every row
                                            collect fired_<variant> cols,
                                            disease_fired col, and
                                            trace JSON
        6. metrics_table()           -> sens / spec / PPV / NPV / F1 /
                                        accuracy, each with a 200-sample
                                        bootstrap 95% CI, per variant AND
                                        the disease-level OR
        7. subgroup_metrics()        -> same metrics stratified by sex and
                                        age bucket - catches e.g. "Cornell
                                        under-fires in women"
        8. train_k_of_n()            -> sweep k=1..N on the training set,
                                        pick the k that maximises F1 on
                                        the validation set
        9. train_cart()              -> shallow CART on variant fires for
                                        a small visual tree
       10. ge_crosscheck()           -> agreement + kappa vs GE
       11. render_disease_report()   -> writes out/reports/<DISEASE>.md
                                        with every table, CI, tree, and
                                        a handful of fire-trace examples

    That final markdown is exactly what a clinician reviews to decide
    whether a given variant of a 12SL criterion is good enough to ship.
    """))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "data/sapphire_xml/lvh_demo/demo_lvh.xml")
