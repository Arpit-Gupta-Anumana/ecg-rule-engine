"""Tests for the Sapphire XML parser using a tiny in-memory synthetic XML."""

from __future__ import annotations

from pathlib import Path

import pytest

from ecg_rule_engine.features.sapphire_xml import parse_file


SYNTHETIC_XML = """<?xml version="1.0" encoding="utf-8"?>
<sapphire xmlns="urn:ge:sapphire:sapphire_3" version="3.1.0.0">
  <demographics>
    <patientInfo>
      <identifier xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:type="ge.idType.MUSEPID">
        <id V="PID123"/>
      </identifier>
      <gender V="M"/>
      <visit><patientData><patientAge V="55" U="a"/></patientData></visit>
    </patientInfo>
    <testInfo>
      <acquisitionDateTime V="2025-05-09T12:22:49.00"/>
    </testInfo>
  </demographics>
  <xmlData>
    <block>
      <params>
        <ecg>
          <var>
            <medianTemplate>
              <num>
                <global>
                  <P_Onset V="140" U="ms"/>
                  <P_Offset V="240" U="ms"/>
                  <QRS_Duration V="130" U="ms"/>
                  <QT_Interval V="400" U="ms"/>
                  <PR_Interval V="160" U="ms"/>
                  <P_Axis V="30" U="deg"/>
                  <R_Axis V="60" U="deg"/>
                  <T_Axis V="40" U="deg"/>
                  <qtcBazettCorrection V="420" U="ms"/>
                  <qtcFredericiaCorrection V="410" U="ms"/>
                  <qtcFraminghamCorrection V="415" U="ms"/>
                  <ventricularRate V="75" U="bpm"/>
                </global>
                <atrialRate V="75" U="bpm"/>
                <PerLead lead="I">
                  <R_Amp V="800" U="uV"/>
                  <S_Amp V="0" U="uV"/>
                  <Q_Amp V="50" U="uV"/>
                  <Q_Dur V="20" U="ms"/>
                </PerLead>
                <PerLead lead="AVL">
                  <R_Amp V="2900" U="uV"/>
                </PerLead>
                <PerLead lead="V3">
                  <S_Amp V="1800" U="uV"/>
                </PerLead>
                <PerLead lead="V5">
                  <R_Amp V="1500" U="uV"/>
                </PerLead>
                <PerLead lead="V6">
                  <R_Amp V="2000" U="uV"/>
                </PerLead>
                <PerLead lead="V1">
                  <S_Amp V="1600" U="uV"/>
                </PerLead>
              </num>
            </medianTemplate>
          </var>
        </ecg>
      </params>
    </block>
  </xmlData>
  <interpretation>
    <params>
      <ecg>
        <interp>
          <resting>
            <obset>
              <statement V="Normal sinus rhythm">
                <code V="22"/>
                <reason V="sinus rate 60-100/min"/>
              </statement>
              <statement V="Left ventricular hypertrophy">
                <code V="150"/>
              </statement>
            </obset>
          </resting>
        </interp>
      </ecg>
    </params>
  </interpretation>
</sapphire>
"""


@pytest.fixture()
def xml_path(tmp_path: Path) -> Path:
    p = tmp_path / "ecg1.xml"
    p.write_text(SYNTHETIC_XML)
    return p


def test_demographics(xml_path: Path):
    rec = parse_file(xml_path)
    assert rec.patient_id == "PID123"
    assert rec.sex == "M"
    assert rec.age_years == 55
    assert rec.acquisition_datetime == "2025-05-09T12:22:49.00"


def test_globals_mapped(xml_path: Path):
    rec = parse_file(xml_path)
    f = rec.features
    assert f["QRS_ms"] == 130
    assert f["QT_ms"] == 400
    assert f["PR_ms"] == 160
    assert f["QRS_axis_deg"] == 60
    assert f["QTc_Bazett_ms"] == 420
    assert f["ventricular_rate_bpm"] == 75
    assert f["atrial_rate_bpm"] == 75
    assert f["P_ms"] == 100  # 240 - 140


def test_per_lead_uv_to_mv(xml_path: Path):
    rec = parse_file(xml_path)
    f = rec.features
    # AVL -> aVL normalization
    assert f["R_aVL_mV"] == pytest.approx(2.9)
    assert f["R_I_mV"] == pytest.approx(0.8)
    assert f["S_V3_mV"] == pytest.approx(1.8)
    assert f["S_V1_mV"] == pytest.approx(1.6)
    assert f["R_V5_mV"] == pytest.approx(1.5)
    assert f["R_V6_mV"] == pytest.approx(2.0)
    assert f["Q_I_ms"] == 20


def test_sokolow_lyon_hand_computed(xml_path: Path):
    """max(S_V1,S_V2)+max(R_V5,R_V6) >= 3.5 → LVH (we only have V1 so S = 1.6)."""
    rec = parse_file(xml_path)
    # S_V1=1.6 (S_V2 absent so max = 1.6), R_V5=1.5, R_V6=2.0 -> max=2.0, sum=3.6 >= 3.5
    from ecg_rule_engine.dsl.expr import parse_expr
    expr = parse_expr("max(S_V1_mV, S_V1_mV) + max(R_V5_mV, R_V6_mV)")
    assert expr.eval(rec.features) >= 3.5


def test_ge_statements_extracted(xml_path: Path):
    rec = parse_file(xml_path)
    texts = [s["text"] for s in rec.ge_statements]
    assert "Normal sinus rhythm" in texts
    assert "Left ventricular hypertrophy" in texts


def test_ge_flags_derived(xml_path: Path):
    rec = parse_file(xml_path)
    assert rec.ge_flags["rhythm_sinus_flag"] == 1
    assert rec.ge_flags["rhythm_afib_flag"] == 0
    assert rec.ge_flags["lbbb_flag"] == 0
