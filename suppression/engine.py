"""
Suppression Engine
==================
Loads suppression_rules.yaml, takes raw model predictions + measurements,
returns a suppressed (cleaned) prediction set with a human-readable trace
showing exactly which rules fired and why.

Pipeline order:
  Pass 1  – Concordance gates (measurement vs. label agreement)
  Pass 2  – Cross-head suppression (label A kills label B)
  Pass 3  – Intra-head assembly (conduction, chamber, ectopy, MI)
  Pass 4  – Rhythm assembly (base rhythm + HR → display name)
  Pass 5  – Screening mode (optional high-specificity filter)

Usage (as library):
    from suppression.engine import apply_suppression
    result = apply_suppression(predictions, measurements)
    print(result["final_labels"])
    print(result["trace"])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# ── Load rules once at import ───────────────────────────────────────────────
_RULES_PATH = Path(__file__).parent / "suppression_rules.yaml"
with open(_RULES_PATH) as f:
    RULES: dict[str, Any] = yaml.safe_load(f)


def apply_suppression(
    predictions: dict[str, int | float],
    measurements: dict[str, float | None],
    *,
    scores: dict[str, float] | None = None,
    screening_mode: bool = False,
) -> dict[str, Any]:
    """
    Run the full suppression pipeline.

    Parameters
    ----------
    predictions : dict
        label → 1/0 (or probability).  1 = model says positive.
    measurements : dict
        Continuous values:
            heart_rate, pr_interval, qrs_duration, axis, qt_interval, qtc
        Missing key → None (rule won't fire).
    scores : dict, optional
        label → confidence.  Used for mutual-exclusion tie-breaking.
    screening_mode : bool
        If True, extra suppressions for low-specificity labels.

    Returns
    -------
    dict with:
        final_labels  – surviving display strings
        suppressed    – label → reason
        assembled     – label → display string (if renamed)
        trace         – human-readable log
        active_raw    – label → True for surviving raw labels
    """
    active: dict[str, bool] = {k: bool(v) for k, v in predictions.items()}
    scores = scores or {}
    suppressed: dict[str, str] = {}
    assembled: dict[str, str] = {}
    trace: list[str] = []

    hr = measurements.get("heart_rate")
    pr = measurements.get("pr_interval")
    qrs = measurements.get("qrs_duration")
    axis = measurements.get("axis")

    # ── helpers ──────────────────────────────────────────────────────────
    def is_on(label: str) -> bool:
        return active.get(label, False)

    def kill(label: str, reason: str):
        if active.get(label):
            active[label] = False
            suppressed[label] = reason
            trace.append(f"  SUPPRESS  {label}  ←  {reason}")

    def check_cond(cond: dict) -> bool:
        ok = True
        if "min_bpm" in cond and (hr is None or hr < cond["min_bpm"]):
            ok = False
        if "max_bpm" in cond and (hr is None or hr >= cond["max_bpm"]):
            ok = False
        if "min_qrs" in cond and (qrs is None or qrs < cond["min_qrs"]):
            ok = False
        if "max_qrs" in cond and (qrs is None or qrs >= cond["max_qrs"]):
            ok = False
        return ok

    # ====================================================================
    # PASS 1 — Concordance gates
    # ====================================================================
    trace.append("─── PASS 1: Concordance gates ───")
    for rule in RULES.get("concordance_rules", []):
        label = rule["label"]
        if not is_on(label):
            continue
        for req in rule.get("requires", []):
            if "pr_gte" in req and (pr is None or pr < req["pr_gte"]):
                kill(label, f"concordance: PR={pr} < {req['pr_gte']} ms")
            if "bpm_gte" in req and (hr is None or hr < req["bpm_gte"]):
                kill(label, f"concordance: HR={hr} < {req['bpm_gte']} bpm")
            if "qrs_gte" in req and (qrs is None or qrs < req["qrs_gte"]):
                kill(label, f"concordance: QRS={qrs} < {req['qrs_gte']} ms")
            if "qrs_lt" in req and qrs is not None and qrs >= req["qrs_lt"]:
                kill(label, f"concordance: QRS={qrs} >= {req['qrs_lt']} ms")
            if "rhythm_in" in req:
                if not any(is_on(r) for r in req["rhythm_in"]):
                    kill(label, f"concordance: no active rhythm in {req['rhythm_in']}")

    # ====================================================================
    # PASS 2 — Cross-head suppression
    # ====================================================================
    trace.append("─── PASS 2: Cross-head suppression ───")
    for rule in RULES.get("cross_head_rules", []):
        triggers = rule.get("when_any", [rule["when"]] if "when" in rule else [])
        if not any(is_on(t) for t in triggers):
            continue
        cond = rule.get("condition", {})
        if cond and not check_cond(cond):
            continue
        ref = rule.get("manual_ref", "")
        who = ", ".join(t for t in triggers if is_on(t))
        for target in rule.get("suppress", []):
            kill(target, f"cross-head: {who} ON [{ref}]")

    # ====================================================================
    # PASS 3 — Intra-head assembly
    # ====================================================================
    trace.append("─── PASS 3: Intra-head assembly ───")

    # ── 3a. Conduction ──────────────────────────────────────────────────
    cond_asm = RULES.get("conduction_assembly", {})

    # BBB completeness
    for item in cond_asm.get("bbb_completeness", []):
        label = item["label"]
        if not is_on(label):
            continue
        if qrs is not None and qrs >= 120:
            assembled[label] = item["when_qrs_wide"]
            trace.append(f"  ASSEMBLE  {label} → {assembled[label]}  (QRS={qrs} >= 120)")
        elif qrs is not None:
            assembled[label] = item["when_qrs_narrow"]
            trace.append(f"  ASSEMBLE  {label} → {assembled[label]}  (QRS={qrs} < 120)")

    # Mutual exclusion (LBBB vs RBBB)
    me = cond_asm.get("mutual_exclusion", [])
    on_me = [l for l in me if is_on(l)]
    if len(on_me) > 1:
        best = max(on_me, key=lambda l: scores.get(l, 0.0))
        for l in on_me:
            if l != best:
                kill(l, f"mutual exclusion: {best} had higher score")

    # Trifascicular FIRST (most specific: RBBB + fascicular block + 1° AVB)
    tri_fired = False
    for tri in cond_asm.get("trifascicular", []):
        comps = tri["components"]
        if not all(is_on(c) for c in comps):
            continue
        if tri.get("requires_qrs_wide") and (qrs is None or qrs < 120):
            continue
        if tri.get("requires_pr_long") and (pr is None or pr < 210):
            continue
        emitted = tri["emits"]
        active[emitted] = True
        assembled[emitted] = emitted
        trace.append(f"  ASSEMBLE  {' + '.join(comps)} → {emitted}")
        for s in tri.get("suppresses", []):
            kill(s, f"absorbed into {emitted}")
        tri_fired = True

    # Bifascicular only if trifascicular didn't fire
    if not tri_fired:
        for bif in cond_asm.get("bifascicular", []):
            comps = bif["components"]
            if not all(is_on(c) for c in comps):
                continue
            if bif.get("requires_qrs_wide") and (qrs is None or qrs < 120):
                continue
            emitted = bif["emits"]
            active[emitted] = True
            assembled[emitted] = emitted
            trace.append(f"  ASSEMBLE  {' + '.join(comps)} → {emitted}")
            for s in bif.get("suppresses", []):
                kill(s, f"absorbed into {emitted}")

    # NICD gate
    nicd_cfg = cond_asm.get("nicd", {})
    if nicd_cfg and is_on("NICD"):
        blocked = any(is_on(b) for b in nicd_cfg.get("blocked_by", []))
        blocked_r = any(is_on(b) for b in nicd_cfg.get("blocked_by_rhythm", []))
        qrs_ok = qrs is not None and qrs > nicd_cfg.get("requires_qrs_gt_ms", 0)
        if blocked or blocked_r or not qrs_ok:
            parts = []
            if blocked:
                parts.append("BBB/fascicular active")
            if blocked_r:
                parts.append("paced/WPW rhythm")
            if not qrs_ok:
                parts.append(f"QRS={qrs} <= {nicd_cfg.get('requires_qrs_gt_ms')} ms")
            kill("NICD", f"NICD gate: {', '.join(parts)}")

    # ── 3b. Chamber assembly ────────────────────────────────────────────
    chamber = RULES.get("chamber_assembly", {})

    # BAE = LAE + RAE
    bae = chamber.get("biatrial", {})
    if bae:
        comps = bae.get("components", [])
        if all(is_on(c) for c in comps):
            emitted = bae["emits"]
            active[emitted] = True
            assembled[emitted] = emitted
            trace.append(f"  ASSEMBLE  {' + '.join(comps)} → {emitted}")
            for s in bae.get("suppresses", []):
                kill(s, f"absorbed into {emitted}")

    # P-wave visibility gate
    p_rhythms = chamber.get("p_wave_rhythms", [])
    if not any(is_on(r) for r in p_rhythms):
        for label in chamber.get("suppress_without_p", []):
            kill(label, f"no P-wave rhythm active (need {p_rhythms})")

    # ── 3c. Ectopy rules ───────────────────────────────────────────────
    ectopy = RULES.get("ectopy_rules", {})
    for dep in ectopy.get("dependency", []):
        label = dep["label"]
        if is_on(label):
            needs = dep.get("requires_any", [])
            if needs and not any(is_on(n) for n in needs):
                kill(label, f"requires one of {needs}")

    ect_me = ectopy.get("mutual_exclusion", [])
    on_ect = [l for l in ect_me if is_on(l)]
    if len(on_ect) > 1:
        best = max(on_ect, key=lambda l: scores.get(l, 0.0))
        for l in on_ect:
            if l != best:
                kill(l, f"ectopy mutual exclusion: {best} wins")

    # ── 3d. MI assembly ─────────────────────────────────────────────────
    mi = RULES.get("mi_assembly", {})
    for combo in mi.get("combined_territories", []):
        comps = combo["components"]
        if all(is_on(c) for c in comps):
            emitted = combo["emits"]
            active[emitted] = True
            assembled[emitted] = emitted
            trace.append(f"  ASSEMBLE  {' + '.join(comps)} → {emitted}")
            for s in combo.get("suppresses", []):
                kill(s, f"absorbed into {emitted}")

    for rule in mi.get("acute_suppresses_old", []):
        if any(is_on(a) for a in rule.get("if_any_acute", [])):
            targets = rule["suppresses"]
            if isinstance(targets, str):
                targets = [targets]
            for s in targets:
                kill(s, "acute MI present → old MI suppressed")

    # ====================================================================
    # PASS 4 — Rhythm assembly
    # ====================================================================
    trace.append("─── PASS 4: Rhythm assembly ───")
    rhythm_cfg = RULES.get("rhythm_assembly", {})

    for entry in rhythm_cfg.get("bases", []):
        base = entry["base"]
        if not is_on(base):
            continue
        for rc in entry.get("rate_conditions", []):
            has_rate_gate = "min_bpm" in rc or "max_bpm" in rc
            if has_rate_gate and hr is None:
                continue

            hr_ok = True
            if "min_bpm" in rc and hr < rc["min_bpm"]:
                hr_ok = False
            if "max_bpm" in rc and hr >= rc["max_bpm"]:
                hr_ok = False

            qrs_ok = True
            if "min_qrs" in rc and (qrs is None or qrs < rc["min_qrs"]):
                qrs_ok = False
            if "max_qrs" in rc and (qrs is None or qrs >= rc["max_qrs"]):
                qrs_ok = False

            if hr_ok and qrs_ok:
                assembled[base] = rc["label"]
                trace.append(f"  ASSEMBLE  {base} (HR={hr}, QRS={qrs}) → {rc['label']}")
                break

    # Rhythm overrides
    for ov in rhythm_cfg.get("overrides", []):
        pair = ov.get("if_both", [])
        if len(pair) == 2 and all(is_on(p) for p in pair):
            if check_cond(ov.get("condition", {})):
                kill(ov["suppress"], f"rhythm override: {ov['keep']} wins")

    # ====================================================================
    # PASS 5 — Screening mode (optional)
    # ====================================================================
    if screening_mode:
        trace.append("─── PASS 5: Screening mode ───")
        for label in RULES.get("screening_mode", {}).get("suppress_when_on", []):
            kill(label, "screening mode: low-specificity label suppressed")

    # ====================================================================
    # Final output
    # ====================================================================
    final_labels = []
    for label, on in active.items():
        if on:
            final_labels.append(assembled.get(label, label))

    trace.append("─── FINAL ───")
    trace.append(f"  Active labels: {final_labels}")
    trace.append(f"  Suppressed:    {list(suppressed.keys())}")

    return {
        "final_labels": final_labels,
        "suppressed": suppressed,
        "assembled": assembled,
        "trace": trace,
        "active_raw": {k: v for k, v in active.items() if v},
    }
