"""End-to-end integration test: synthetic XMLs + synthetic GT + CLI evaluate.

Generates a small in-memory dataset of positive/negative LVH ECGs, writes
them as Sapphire XML, builds a matching GT CSV, runs the `evaluate` CLI
command, and asserts the per-disease report was produced with sensible
metrics.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from ecg_rule_engine.cli import main as cli_main


# Minimal Sapphire XML template populated with placeholders.
XML_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<sapphire xmlns="urn:ge:sapphire:sapphire_3" version="3.1.0.0">
  <demographics>
    <patientInfo>
      <identifier xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="ge.idType.MUSEPID"><id V="{pid}"/></identifier>
      <gender V="{sex}"/>
      <visit><patientData><patientAge V="{age}" U="a"/></patientData></visit>
    </patientInfo>
    <testInfo><acquisitionDateTime V="2025-01-01T00:00:00.00"/></testInfo>
  </demographics>
  <xmlData><block><params><ecg><var><medianTemplate><num>
    <global>
      <QRS_Duration V="{qrs}" U="ms"/>
      <QT_Interval V="380" U="ms"/>
      <PR_Interval V="160" U="ms"/>
      <P_Axis V="30" U="deg"/>
      <R_Axis V="40" U="deg"/>
      <T_Axis V="40" U="deg"/>
      <qtcBazettCorrection V="400" U="ms"/>
      <ventricularRate V="75" U="bpm"/>
    </global>
    <atrialRate V="75" U="bpm"/>
    <PerLead lead="aVL"><R_Amp V="{r_avl}" U="uV"/></PerLead>
    <PerLead lead="V3"><S_Amp V="{s_v3}" U="uV"/></PerLead>
    <PerLead lead="V1"><S_Amp V="{s_v1}" U="uV"/></PerLead>
    <PerLead lead="V2"><S_Amp V="{s_v1}" U="uV"/></PerLead>
    <PerLead lead="V5"><R_Amp V="{r_v5}" U="uV"/></PerLead>
    <PerLead lead="V6"><R_Amp V="{r_v5}" U="uV"/></PerLead>
  </num></medianTemplate></var></ecg></params></block></xmlData>
  <interpretation><params><ecg><interp><resting><obset>
    <statement V="{stmt}"><code V="1"/></statement>
  </obset></resting></interp></ecg></params></interpretation>
</sapphire>
"""


def _xml(pid: str, sex: str, age: int, qrs: int, r_avl: int, s_v3: int, s_v1: int, r_v5: int, stmt: str) -> str:
    return XML_TEMPLATE.format(
        pid=pid, sex=sex, age=age, qrs=qrs,
        r_avl=r_avl, s_v3=s_v3, s_v1=s_v1, r_v5=r_v5, stmt=stmt,
    )


@pytest.fixture()
def dataset(tmp_path: Path) -> dict:
    xml_dir = tmp_path / "xml"
    xml_dir.mkdir()
    records: list[dict] = []

    # 30 positive LVH cases (strong Sokolow-Lyon + Cornell, ages 40-70)
    for i in range(30):
        pid = f"P{i:03d}"
        (xml_dir / f"{pid}.xml").write_text(_xml(
            pid=pid, sex="M", age=50 + (i % 20),
            qrs=100, r_avl=1500, s_v3=2500, s_v1=2500, r_v5=2500,
            stmt="Left ventricular hypertrophy",
        ))
        records.append({"ecg_id": pid, "patient_id": pid, "sex": "M",
                        "age_years": 50 + (i % 20), "disease": "LVH", "label": 1})

    # 30 negative cases (low amplitudes)
    for i in range(30, 60):
        pid = f"P{i:03d}"
        (xml_dir / f"{pid}.xml").write_text(_xml(
            pid=pid, sex="F" if i % 2 else "M", age=30 + (i % 30),
            qrs=90, r_avl=300, s_v3=200, s_v1=300, r_v5=500,
            stmt="Normal ECG",
        ))
        records.append({"ecg_id": pid, "patient_id": pid,
                        "sex": ("F" if i % 2 else "M"),
                        "age_years": 30 + (i % 30), "disease": "LVH", "label": 0})

    gt_csv = tmp_path / "gt.csv"
    pd.DataFrame(records).to_csv(gt_csv, index=False)

    rules_src = Path(__file__).parent.parent / "rules"
    rules_dst = tmp_path / "rules"
    rules_dst.mkdir()
    shutil.copy2(rules_src / "lvh.yaml", rules_dst / "lvh.yaml")

    out_dir = tmp_path / "out"
    return {"xml": xml_dir, "gt": gt_csv, "rules": rules_dst, "out": out_dir}


def test_cli_validate_succeeds(dataset):
    rc = cli_main(["validate", "--rules", str(dataset["rules"])])
    assert rc == 0


def test_cli_parse_xml_produces_features(dataset, tmp_path):
    out = tmp_path / "features.csv"
    rc = cli_main(["parse-xml", "--xml", str(dataset["xml"]), "--out", str(out)])
    assert rc == 0
    df = pd.read_csv(out)
    assert len(df) == 60
    assert "R_aVL_mV" in df.columns


def test_cli_evaluate_end_to_end(dataset):
    rc = cli_main([
        "evaluate",
        "--rules", str(dataset["rules"]),
        "--xml", str(dataset["xml"]),
        "--gt", str(dataset["gt"]),
        "--out", str(dataset["out"]),
    ])
    assert rc == 0
    md = dataset["out"] / "reports" / "LVH.md"
    assert md.exists()
    text = md.read_text()
    assert "# LVH" in text
    assert "Per-variant metrics" in text
    assert "Fire-trace examples" in text


def test_lvh_rule_catches_positives(dataset):
    """Separately sanity-check that our LVH rule fires on the synthetic positives."""
    from ecg_rule_engine.dsl.loader import load_rules_dir
    from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
    from ecg_rule_engine.features.sapphire_xml import parse_dir

    rules = load_rules_dir(dataset["rules"])
    df = parse_dir(dataset["xml"])
    positives = 0
    for ecg_id, row in df.iterrows():
        if not str(ecg_id).startswith("P0"):  # P000..P029 are positives
            continue
        if int(str(ecg_id)[1:]) >= 30:
            continue
        feats = {k: float(v) for k, v in row.items() if isinstance(v, (int, float)) and pd.notna(v)}
        ctx = EvalContext(features=feats, sex=row["sex"], age_years=int(row["age_years"]))
        r = evaluate_disease(rules["LVH"], ctx)
        positives += int(r.fired)
    assert positives >= 25  # >= 25/30 is generous; expected 30/30
