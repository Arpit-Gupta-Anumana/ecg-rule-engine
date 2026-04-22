# ECG Rule Engine

> **Faithful, explainable implementation of the GE Marquette 12SL Physician's Guide (v24) as a deterministic rule engine — evaluated on 21,799 real 12-lead ECGs.**

---

## Table of Contents

- [Overview](#overview)
- [Why This Matters](#why-this-matters)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Rule DSL (Domain-Specific Language)](#rule-dsl-domain-specific-language)
- [Expression Language](#expression-language)
- [Coverage](#coverage)
- [Evaluation Results](#evaluation-results)
- [How It Works End-to-End](#how-it-works-end-to-end)
- [Installation](#installation)
- [Usage](#usage)
- [Running on Your Own Data](#running-on-your-own-data)
- [Feature Dictionary](#feature-dictionary)
- [Design Principles](#design-principles)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [Technology Stack](#technology-stack)

---

## Overview

The **GE Marquette 12SL Physician's Guide** is a ~100-page reference manual that describes the algorithmic logic behind the 12SL ECG interpretation program used in GE Healthcare devices worldwide. The guide contains hundreds of "if-else-and-or" statements across dozens of cardiac conditions — written in dense, unstructured prose with inconsistent formatting, mixed units, and no formal syntax.

This project **systematically converts** that prose into a machine-readable, executable rule engine. Every rule:

- Is transcribed verbatim from the manual into a validated YAML format
- Cites its exact manual page and section
- Documents any operationalization or departure from the manual's text
- Produces a full human-readable **fire trace** explaining exactly which conditions matched (or didn't) and with what measured values

The system was validated against **21,799 ECG records** from the PTB-XL dataset using GE 12SL-computed features and the GE 12SL algorithm's own diagnostic statements as ground truth.

---

## Why This Matters

| Problem | Our Solution |
|---------|-------------|
| The 12SL manual is hard to interpret — even for clinicians | Structured, machine-readable YAML rules with plain-English annotations |
| "Black box" ECG interpretation — no one knows *why* a diagnosis was flagged | Every prediction includes a **fire trace** that shows each clause, threshold, and measured value |
| No way to validate the algorithm against clinical data at scale | Automated evaluation pipeline across 21,799 ECGs with full metrics (Sensitivity, Specificity, PPV, NPV, F1, ROC AUC) |
| Manual text has no consistent format or units | All rules are unit-normalized, validated by Pydantic schema, and cross-referenced to source pages |

---

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         5-PHASE PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  PHASE 1: EXTRACTION                                                    │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐                  │
│  │  12SL PDF │───>│ PDF Segmenter │───>│ LLM-Assisted │──> YAML Rules   │
│  │  (Manual) │    │ (pdfplumber)  │    │ Transcription│    (56 files)   │
│  └──────────┘    └───────────────┘    └──────────────┘                  │
│                                                                         │
│  PHASE 2: RULE ENGINE                                                   │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐            │
│  │ YAML Rules   │───>│ Pydantic DSL  │───>│  Deterministic │──> Boolean │
│  │ (validated)  │    │ Schema        │    │  Evaluator     │   + Trace  │
│  └──────────────┘    └───────────────┘    └────────────────┘            │
│                                                                         │
│  PHASE 3: FEATURE INGESTION                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐            │
│  │ PTB-XL CSV   │───>│ Feature       │───>│  Canonical     │            │
│  │ (788 cols)   │    │ Mapper        │    │  Feature Dict  │            │
│  └──────────────┘    └───────────────┘    └────────────────┘            │
│                                                                         │
│  PHASE 4: EVALUATION                                                    │
│  ┌──────────────┐    ┌───────────────┐    ┌────────────────┐            │
│  │ Rule Preds   │───>│ vs. Ground    │───>│  Full Metrics  │            │
│  │ (per ECG)    │    │ Truth (12SL)  │    │  CSV + Report  │            │
│  └──────────────┘    └───────────────┘    └────────────────┘            │
│                                                                         │
│  PHASE 5: REPORTING                                                     │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │ Per-disease metrics, per-variant fire rates, confusion   │           │
│  │ matrices, fire traces, summary reports                   │           │
│  └──────────────────────────────────────────────────────────┘           │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
ecg_rule_engine/
├── src/ecg_rule_engine/
│   ├── dsl/                    # Rule DSL: Pydantic schema, YAML loader, expression parser
│   │   ├── schema.py           #   Disease/Variant/Clause models with validation
│   │   ├── loader.py           #   YAML → Disease object with schema validation
│   │   └── expr.py             #   Safe math expression language (max, min, abs, +, -, *, /)
│   ├── engine/                 # Deterministic rule evaluator
│   │   ├── evaluator.py        #   EvalContext + clause evaluation → DiseaseResult + TraceNode
│   │   └── trace.py            #   Fire trace formatting (human-readable explanation)
│   ├── features/               # Feature registration and extraction
│   │   ├── registry.py         #   Canonical feature name registry (138+ features)
│   │   ├── sapphire_xml.py     #   GE Sapphire XML parser
│   │   ├── ptbxl.py            #   PTB-XL waveform feature extractor
│   │   ├── waveform_features.py#   Low-level waveform processing
│   │   └── ge_output.py        #   GE output cross-reference utilities
│   ├── extraction/             # PDF → YAML rule extraction pipeline
│   │   ├── pdf_to_sections.py  #   PDF segmenter (pdfplumber)
│   │   ├── llm_transcribe.py   #   LLM-assisted rule transcription
│   │   ├── roundtrip_check.py  #   Verification: YAML → load → re-serialize
│   │   └── driver.py           #   End-to-end extraction orchestration
│   ├── eval/                   # Evaluation and reporting
│   │   ├── metrics.py          #   Sens, Spec, PPV, NPV, F1, ROC AUC, bootstrap CIs
│   │   ├── runner.py           #   Batch evaluation runner
│   │   ├── gt_loader.py        #   Ground truth loading and mapping
│   │   ├── ge_crosscheck.py    #   GE interpretation cross-check
│   │   ├── report.py           #   Report generation
│   │   └── distill.py          #   Optional: shallow CART / RuleFit distillation
│   └── cli.py                  # Command-line interface
├── rules/                      # 56 YAML rule files (one per disease/condition)
│   ├── lvh.yaml                #   Left ventricular hypertrophy (4 variants)
│   ├── lbbb.yaml               #   Left bundle branch block
│   ├── rbbb.yaml               #   Right bundle branch block
│   ├── afib.yaml               #   Atrial fibrillation
│   ├── q_wave_mi.yaml          #   Q-wave myocardial infarction (5 region variants)
│   ├── sinus_rhythm.yaml       #   Sinus rhythm (4 rate variants)
│   ├── ...                     #   + 50 more adult and pediatric rules
│   └── ped_*.yaml              #   17 pediatric-specific rule files
├── scripts/                    # Evaluation and utility scripts
│   ├── eval_12sl_features.py   #   Main evaluation: rules vs. 21,799 PTB-XL ECGs
│   ├── extract_pdf_sweep.py    #   Full PDF extraction sweep
│   ├── build_rule_catalog.py   #   Rule inventory and catalog builder
│   ├── walkthrough_lvh.py      #   Educational: step-by-step LVH evaluation
│   └── ...                     #   + debugging and spot-check utilities
├── reports/                    # Generated evaluation reports
│   ├── 12sl_full_metrics.csv   #   Per-disease metrics on 21,799 ECGs
│   └── 12sl_variant_fires.csv  #   Per-variant fire counts
├── tests/                      # pytest test suite
│   ├── test_dsl_schema.py      #   DSL schema validation tests
│   ├── test_evaluator.py       #   Engine evaluation tests
│   ├── test_expr.py            #   Expression parser tests
│   └── test_integration_e2e.py #   End-to-end integration tests
├── pyproject.toml              # Project configuration and dependencies
└── README.md                   # This file
```

---

## Rule DSL (Domain-Specific Language)

Every cardiac condition is encoded in a single YAML file with a strict, validated schema:

```yaml
# rules/lvh.yaml — Left Ventricular Hypertrophy
disease: LVH
manual_source: "12SL Physicians Guide v24, §3.7.2.9.2 p.64-65"
description: >
  Left ventricular hypertrophy. The 12SL program incorporates four commonly used
  LVH criteria: R in aVL, Sokolow-Lyon voltage, Cornell product, and Romhilt-Estes
  point score.
exclusions: [LBBB, RBBB, WPW, Paced]
sex_specific: true
variants:
  - name: R_in_aVL
    manual_page: 64
    description: "R amplitude in aVL > 1.1 mV (manual: > 1100 uV)."
    clause:
      kind: threshold
      expr: "R_aVL_mV"
      op: ">"
      value: 1.1

  - name: Sokolow_Lyon
    manual_page: 64
    clause:
      kind: all_of
      clauses:
        - kind: threshold
          expr: "max(S_V1_mV, S_V2_mV) + max(R_V5_mV, R_V6_mV)"
          op: ">"
          value: 3.5
        - kind: threshold
          expr: "age_years"
          op: ">="
          value: 30

  - name: Cornell_product
    manual_page: 64
    clause:
      kind: all_of
      clauses:
        - kind: threshold
          expr: "(R_aVL_mV + S_V3_mV + 0.6 * sex_F_flag) * QRS_ms"
          op: ">"
          value: 244
        - kind: threshold
          expr: "age_years"
          op: ">="
          value: 30

  - name: Romhilt_Estes
    manual_page: 64
    clause:
      kind: point_score
      threshold_positive: 5
      items:
        - points: 3
          when: { kind: threshold, expr: "max(R_I_mV, ...)", op: ">=", value: 2.0 }
        - points: 3
          when: { kind: any_of, clauses: [...] }
        - points: 2
          when: { kind: threshold, expr: "QRS_axis_deg", op: "<=", value: -30 }
        - points: 1
          when: { kind: threshold, expr: "QRS_ms", op: ">=", value: 90 }
```

### Clause Types

| Clause Kind | Description | Example |
|-------------|-------------|---------|
| `threshold` | Single comparison: `expr op value` | `QRS_ms >= 120` |
| `all_of` | Boolean AND across child clauses | Wide QRS AND deep S in V1 AND upright R in I |
| `any_of` | Boolean OR across child clauses | Deep S in V1 OR QS morphology in V1 |
| `not_` | Boolean negation | Not in atrial fibrillation |
| `point_score` | Weighted sum; true if score >= threshold | Romhilt-Estes (5 criteria, threshold = 5 pts) |

### Threshold Value Types

| Type | Description | Example |
|------|-------------|---------|
| Scalar | Fixed numeric threshold | `value: 3.5` |
| Sex-specific | Different thresholds for M/F | `value: { M: 2.8, F: 2.0 }` |
| Age-banded | Piecewise by age | `value: { bands: [{ max_age: 40, value: 3.0 }, ...] }` |

---

## Expression Language

Rule expressions use a **safe, auditable math DSL** (no arbitrary code execution):

```
expr    := term (('+' | '-') term)*
term    := factor (('*' | '/') factor)*
factor  := NUMBER | FEATURE_NAME | func(args) | (expr) | -factor
func    := max | min | abs | sum | avg
```

**Examples from actual rules:**

| Expression | Used In |
|-----------|---------|
| `max(S_V1_mV, S_V2_mV) + max(R_V5_mV, R_V6_mV)` | LVH Sokolow-Lyon |
| `(R_aVL_mV + S_V3_mV + 0.6 * sex_F_flag) * QRS_ms` | LVH Cornell Product |
| `R_I_mV - S_I_mV` | LBBB lateral dominance |
| `S_V1_mV - R_V1_mV` | LBBB V1 negativity |
| `abs(QRS_axis_deg - T_axis_deg)` | Abnormal QRS-T angle |

Every feature name is validated at load time against a **138-feature registry** covering amplitudes, durations, intervals, axes, and rates across all 12 leads.

---

## Coverage

### 56 Rule Files | 133 Variants | 3 Categories

**Adult Contour Rules (21 files):**
LVH, RVH, LBBB, RBBB, Incomplete Bundle Blocks, Hemiblocks (LAFB/LPFB), Nonspecific IVCD, WPW, Q-Wave MI (5 regions), ST Elevation Injury, ST Depression Ischemia, T-Wave Ischemia, Prolonged QT, Low Voltage QRS, Atrial Enlargement (LAE/RAE/BAE), Biventricular Hypertrophy, Pulmonary Disease Pattern, QRS Axis Deviation, Poor R-Wave Progression, Pericarditis/Early Repolarization, Brugada Pattern, Abnormal QRS-T Angle, Electrode Reversals, Nonspecific ST Elevation/Depression, Nonspecific T-Wave

**Adult Rhythm Rules (13 files):**
Sinus Rhythm (NSR/Bradycardia/Tachycardia/Marked Bradycardia), Atrial Fibrillation, Atrial Flutter, AV Block (1st/2nd/3rd degree), PR Interval (1st degree block, Short PR), Pacing, Sinus Arrhythmia, Ectopic Atrial Rhythm, Junctional Rhythm, Ectopy (PVC/PAC), Undetermined Rhythm, Acute MI/STEMI

**Pediatric Rules (17 files):**
Pediatric-specific variants for LVH, RVH, conduction abnormalities, atrial enlargement, axis deviation, prolonged QT, ST/T-wave changes, WPW, Brugada, dextrocardia, low voltage, MI, and more — with age-adjusted thresholds per the manual.

---

## Evaluation Results

### Dataset

**PTB-XL** — 21,799 twelve-lead ECGs with **GE 12SL-computed features** (788 measurements per ECG) and **GE 12SL diagnostic statements** as ground truth.

### Top-Performing Rules (by F1 Score)

| Disease | GT Positives | Sensitivity | Specificity | PPV | NPV | F1 | ROC AUC |
|---------|-------------|-------------|-------------|-----|-----|-------|---------|
| **PR Interval** | 1,343 | **100.0%** | **99.99%** | **99.8%** | **100.0%** | **99.9%** | 0.999 |
| **Sinus Rhythm** | 19,634 | **98.9%** | **93.0%** | **99.2%** | **90.2%** | **99.1%** | 0.960 |
| **Low Voltage QRS** | 356 | **100.0%** | **99.0%** | 62.2% | **100.0%** | **76.7%** | 0.995 |
| **Q-Wave MI** | 4,132 | 78.7% | 89.0% | 62.6% | 94.7% | **69.7%** | 0.838 |
| **LBBB** | 566 | **96.3%** | **97.8%** | 54.4% | **99.9%** | **69.5%** | 0.971 |
| **Prolonged QT** | 636 | 65.6% | **98.8%** | 61.2% | **99.0%** | **63.3%** | 0.822 |
| **RBBB** | 721 | 52.7% | **99.3%** | 70.6% | **98.4%** | **60.4%** | 0.760 |
| **LVH** | 4,108 | 38.7% | **99.98%** | **99.8%** | 87.5% | **55.8%** | 0.693 |
| **Atrial Enlargement** | 588 | 46.1% | **99.3%** | 63.8% | **98.5%** | **53.5%** | 0.727 |
| **RVH** | 44 | 50.0% | **99.9%** | 48.9% | **99.9%** | **49.4%** | 0.749 |
| **WPW** | 49 | 42.9% | **99.9%** | 42.0% | **99.9%** | **42.4%** | 0.714 |

### Conditions Requiring Beat-Level Features (0% Sensitivity — Expected)

These rules require features not extractable from static measurements:

| Disease | GT Positives | Reason for 0% Sensitivity |
|---------|-------------|---------------------------|
| Atrial Fibrillation | 1,396 | Needs RR-irregularity (beat-to-beat RR intervals) |
| Pacing | 222 | Needs pacing spike detection (high-frequency analysis) |
| Atrial Flutter | 151 | Needs sawtooth morphology detection |
| Sinus Arrhythmia | 1,473 | Needs beat-to-beat RR variability |
| Ectopy (PVC/PAC) | 2,260 | Needs individual beat classification |
| Junctional Rhythm | 135 | Needs P-wave absence/retrograde detection per beat |

*These 0% results are scientifically honest — they reflect genuine feature-set limitations, not rule errors.*

### Full Metrics

Complete results for all 33 evaluated diseases are in [`reports/12sl_full_metrics.csv`](reports/12sl_full_metrics.csv). Per-variant fire rates are in [`reports/12sl_variant_fires.csv`](reports/12sl_variant_fires.csv).

---

## How It Works End-to-End

### Step 1: A Rule is Loaded

```python
from ecg_rule_engine.dsl.loader import load_disease_yaml
disease = load_disease_yaml("rules/lvh.yaml")
# → Disease(disease="LVH", exclusions=["LBBB","RBBB","WPW","Paced"], variants=[...])
```

### Step 2: An ECG's Features Are Prepared

```python
features = {
    "R_aVL_mV": 1.35,        # R wave amplitude in aVL = 1.35 mV
    "S_V1_mV": 2.10,         # S wave depth in V1 = 2.10 mV
    "R_V5_mV": 2.80,         # R wave amplitude in V5 = 2.80 mV
    "QRS_ms": 96,             # QRS duration = 96 ms
    "QRS_axis_deg": -15,      # QRS axis = -15 degrees
    "lbbb_flag": 0.0,         # No LBBB present (exclusion check)
    # ... (138 features total)
}
```

### Step 3: The Rule Engine Evaluates

```python
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease
ctx = EvalContext(features=features, sex="M", age_years=55)
result = evaluate_disease(disease, ctx)
# → DiseaseResult(disease="LVH", fired=True, fired_variants=["R_in_aVL"])
```

### Step 4: The Fire Trace Explains *Why*

```
LVH: FIRED (1 of 4 variants matched)
  Exclusions: CLEAR (lbbb_flag=0, rbbb_flag=0, wpw_flag=0, paced_flag=0)

  ✓ R_in_aVL:
    R_aVL_mV = 1.35 > 1.1  → TRUE

  ✗ Sokolow_Lyon:
    all_of (1/2 matched)
      max(S_V1_mV=2.1, S_V2_mV=1.8) + max(R_V5_mV=2.8, R_V6_mV=2.1) = 4.9 > 3.5  → TRUE
      age_years=55 >= 30  → TRUE
    [Note: Sokolow fires too, but R_in_aVL already triggered disease-level OR]
```

---

## Installation

```bash
# Clone and set up
cd ecg_rule_engine
python -m venv .venv
source .venv/bin/activate

# Install with all optional dependencies
pip install -e ".[dev,extract,distill]"

# Or minimal install (just the rule engine + evaluation)
pip install -e .
```

### Requirements

- Python >= 3.11
- Core: `pydantic`, `pyyaml`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pdfplumber`, `rich`, `jinja2`, `tabulate`
- Extraction (optional): `openai`, `anthropic`
- Dev (optional): `pytest`, `ruff`, `mypy`, `jupyter`

---

## Usage

### Run Full Evaluation on PTB-XL

```bash
python scripts/eval_12sl_features.py
# Outputs:
#   reports/12sl_full_metrics.csv      (per-disease metrics)
#   reports/12sl_variant_fires.csv     (per-variant fire counts)
```

### Run with a Subset

```bash
python scripts/eval_12sl_features.py --limit 500
```

### Run Tests

```bash
pytest -q
```

### Evaluate a Single ECG (Programmatic)

```python
from ecg_rule_engine.dsl.loader import load_disease_yaml
from ecg_rule_engine.engine.evaluator import EvalContext, evaluate_disease

disease = load_disease_yaml("rules/lbbb.yaml")
ctx = EvalContext(
    features={"QRS_ms": 145, "R_I_mV": 0.8, "S_I_mV": 0.1,
              "S_V1_mV": 1.5, "R_V1_mV": 0.1, "Q_V6_mV": 0.01},
    sex="M",
    age_years=65,
)
result = evaluate_disease(disease, ctx)
print(f"LBBB fired: {result.fired}")
print(f"Fired variants: {result.fired_variants()}")
```

---

## Running on Your Own Data

The universal script `scripts/run_rules.py` lets you evaluate all rules on **any CSV** containing ECG measurements — no code changes required.

### Quick Start (GE 12SL format)

If your CSV has GE 12SL column names (e.g., `R_Amp_V1`, `QRS_Dur_Global`), just run:

```bash
python scripts/run_rules.py --csv your_data.csv --out-preds predictions.csv
```

### Custom Column Names

If your CSV uses different column names, create a mapping JSON that maps our canonical feature IDs to your column names:

```json
{
    "QRS_ms":               "QRS_Duration",
    "PR_ms":                "PR_Interval",
    "ventricular_rate_bpm": "HeartRate",
    "R_V1_mV":              "R_amp_V1",
    "S_V1_mV":              {"column": "S_amp_V1", "transform": "abs"},
    "STM_V1_mV":            {"column": "ST_mid_V1", "transform": "divide_1000"},
    "age_years":            "patient_age",
    "sex_col":              "gender",
    "sex_male_value":       "M"
}
```

Then run:

```bash
python scripts/run_rules.py --csv your_data.csv --mapping your_mapping.json --out-preds predictions.csv
```

### Transform Options

For features where your CSV stores values in a different convention:

| Transform | What it does | When to use |
|-----------|-------------|-------------|
| `"abs"` | Takes absolute value | Q and S wave amplitudes stored as negative in your CSV |
| `"divide_1000"` | Divides by 1000 | Amplitudes stored in microvolts instead of millivolts |
| `"negate"` | Multiplies by -1 | Sign convention is inverted |

### All Options

```bash
python scripts/run_rules.py \
    --csv your_data.csv \
    --mapping your_mapping.json \
    --age-col patient_age \
    --sex-col gender \
    --sex-male-value "M" \
    --diseases LBBB,RBBB,LVH \
    --limit 500 \
    --out-preds predictions.csv \
    --out-traces traces.json
```

### Output

- **`--out-preds`**: CSV with one row per ECG, one column per disease (1 = fired, 0 = not fired, -1 = skipped)
- **`--out-traces`**: JSON with full fire traces for every ECG (clause-by-clause audit trail)

See [`examples/sample_mapping.json`](examples/sample_mapping.json) for a full example mapping file.

---

## Feature Dictionary

All 138 features used by the rule engine are documented in [`feature_dictionary.json`](feature_dictionary.json). Each entry includes:

| Field | Description |
|-------|-------------|
| `description` | Human-readable description of the feature |
| `unit` | Unit of measurement (mV, ms, degrees, bpm, flag) |
| `category` | Feature category (global_interval, per_lead_amplitude, etc.) |
| `sign` | Sign convention (positive, signed, positive magnitude, binary) |
| `example_csv_columns` | Common column names for this feature in various CSV formats |

### Feature Categories

| Category | Count | Examples |
|----------|------:|---------|
| `per_lead_amplitude` | 84 | R, Q, S, P, T, STJ, STM amplitudes across 12 leads |
| `per_lead_duration` | 24 | Q duration, QRS duration per lead |
| `global_interval` | 10 | QRS, PR, QT, QTc, RR, PP, P durations |
| `global_rate` | 2 | Ventricular rate, atrial rate |
| `global_axis` | 3 | QRS axis, P axis, T axis |
| `demographics` | 3 | age_years, sex_M_flag, sex_F_flag |
| `exclusion_flag` | 12 | LBBB, RBBB, WPW, paced, etc. |

---

## Design Principles

1. **The manual is authoritative.** Every threshold, criterion, and exclusion is transcribed from the GE 12SL Physician's Guide. We never invent criteria or adjust thresholds to improve scores.

2. **Every prediction is explainable by construction.** Fire traces show exactly which clauses matched, with what actual measured values, against what thresholds. There is no opacity.

3. **Departures are documented, not hidden.** When the manual describes qualitative morphology (e.g., "predominantly negative QRS in V1") that must be operationalized as a numeric threshold, this is explicitly documented in `manual_departure_notes` and flagged for clinician review.

4. **No circular logic.** Ground truth labels are never used as input features. Exclusion flags (e.g., "skip LVH if LBBB is present") use the algorithm's own prior diagnostic outputs — not the ground truth being evaluated against.

5. **Honest results.** If a rule requires features we don't have (beat-level RR intervals, pacing spike detection, waveform morphology), it scores 0% sensitivity — and we explain why. We don't patch rules to look better.

6. **Validated at every layer.** The Pydantic schema validates YAML structure, expression syntax, feature name existence, and variant name uniqueness at load time — before any evaluation runs.

---

## Known Limitations & Future Work

### Current Limitations

| Limitation | Impact | Path Forward |
|-----------|--------|-------------|
| No beat-level features (RR intervals, individual beat morphology) | AFib, Pacing, Ectopy, Flutter, Sinus Arrhythmia cannot fire | Add beat-level RR-interval extractor |
| No pacing spike detector | Pacing rule is a stub | High-frequency spike detection module |
| No delta-wave morphology detector | WPW relies on PR-interval proxy | Waveform morphology analysis |
| Age/sex unavailable in current CSV | Age-gated rules (Sokolow-Lyon, Cornell) and sex-specific thresholds cannot activate | Merge demographics from PTB-XL metadata |
| Pediatric rules untested | 17 rule files, 0 pediatric ECGs in dataset | Obtain pediatric ECG dataset |

### Planned Improvements

- **Threshold calibration** — Train on folds 1-8, validate on fold 9, report on fold 10 for operationalized thresholds
- **Bootstrap confidence intervals** — 95% CIs on all metrics via bootstrap resampling
- **Beat-level feature extraction** — RR intervals, beat classification, pacing spike detection
- **Demographics integration** — Age and sex from PTB-XL metadata for full rule activation
- **Interactive dashboard** — Web-based per-ECG fire trace explorer

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Rule Schema & Validation | Pydantic v2 |
| Rule Serialization | YAML (PyYAML) |
| Expression Parser | Custom recursive-descent parser |
| Data Processing | Pandas, NumPy |
| PDF Extraction | pdfplumber |
| LLM-Assisted Transcription | OpenAI / Anthropic APIs |
| Evaluation Metrics | scikit-learn, custom implementations |
| Visualization | Matplotlib |
| Formatting & CLI | Rich, Tabulate, Jinja2 |
| Testing | pytest |
| Linting | Ruff |
| Type Checking | mypy |

---

*This is a research and evaluation tool, not a clinical product. All rules are derived from the GE Marquette 12SL Physician's Guide v24 for research purposes.*
