"""Roundtrip verification: DSL YAML -> prose -> compared to the manual section.

Pipeline:

    original manual section
           |
           v
    transcribe_section (LLM #1)  -->  DSL YAML
           |
           v
    dsl_to_prose (LLM #2)        -->  regenerated English
           |
           v
    compare_to_original          -->  similarity + diff + flagged for review

`dsl_to_prose` uses a DIFFERENT model family from the extractor when possible,
so extractor biases do not mask themselves.

Similarity is computed with difflib on a normalized token stream; anything
below `min_ratio` (default 0.55) OR any YAML that fails DSL validation is
moved into the review queue.
"""

from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from ..dsl.schema import Disease
from .llm_transcribe import _call_anthropic, _call_openai

Provider = Literal["openai", "anthropic"]


PROSE_SYSTEM = """\
You are an inverse transcriber. You take a structured YAML Rule DSL describing
ECG criteria and regenerate faithful English prose that a clinician could read.
Do NOT add any claim that is not directly encoded in the YAML. Do NOT simplify.
If a threshold has sex-specific values, state both. If there are exclusions,
state them. If a criterion is point-scored, list every item and its points.
Keep the prose terse — the goal is fidelity, not readability."""


PROSE_USER_TEMPLATE = """\
YAML to rewrite into prose:

{yaml_text}
"""


@dataclass
class RoundtripResult:
    ratio: float
    regenerated_prose: str
    diff: str
    dsl_valid: bool
    flagged_for_review: bool
    reasons: list[str]


# --- text normalization for comparison ---------------------------------------

_WS_RE = re.compile(r"\s+")
_HEADER_RE = re.compile(r"^#.*$", re.MULTILINE)
_CITATION_RE = re.compile(r"\[\d+(?:,\s*\d+)*\]")  # strip footnote refs like [11, 19]
_PAGE_MARK_RE = re.compile(r"\d+/\d+\s*$", re.MULTILINE)


def _normalize(text: str) -> str:
    text = _HEADER_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = _PAGE_MARK_RE.sub("", text)
    text = re.sub(r"2056246-007 Revision \d+.*", "", text)
    text = re.sub(r"Physician Guide.*", "", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=_normalize(a), b=_normalize(b), autojunk=False).ratio()


def _unified_diff(a: str, b: str) -> str:
    a_lines = _normalize(a).split(". ")
    b_lines = _normalize(b).split(". ")
    return "\n".join(difflib.unified_diff(a_lines, b_lines, lineterm="", n=2))


def dsl_to_prose(
    yaml_text: str,
    *,
    provider: Provider = "anthropic",
    model: str | None = None,
) -> str:
    user = PROSE_USER_TEMPLATE.format(yaml_text=yaml_text)
    if provider == "openai":
        return _call_openai(PROSE_SYSTEM, user, model or "gpt-4.1-mini")
    if provider == "anthropic":
        return _call_anthropic(PROSE_SYSTEM, user, model or "claude-3-5-sonnet-latest")
    raise ValueError(f"Unknown provider: {provider}")


def roundtrip_check(
    original_text: str,
    yaml_text: str,
    *,
    min_ratio: float = 0.55,
    prose_provider: Provider = "anthropic",
    prose_model: str | None = None,
) -> RoundtripResult:
    reasons: list[str] = []
    dsl_valid = True
    try:
        data = yaml.safe_load(yaml_text)
        Disease.model_validate(data)
    except Exception as e:  # noqa: BLE001
        dsl_valid = False
        reasons.append(f"DSL invalid: {e}")

    try:
        prose = dsl_to_prose(yaml_text, provider=prose_provider, model=prose_model)
    except Exception as e:  # noqa: BLE001
        reasons.append(f"prose regen failed: {e}")
        return RoundtripResult(
            ratio=0.0, regenerated_prose="", diff="",
            dsl_valid=dsl_valid, flagged_for_review=True, reasons=reasons,
        )

    r = _ratio(original_text, prose)
    diff = _unified_diff(original_text, prose)
    flagged = (not dsl_valid) or (r < min_ratio)
    if r < min_ratio:
        reasons.append(f"similarity {r:.2f} < threshold {min_ratio}")
    return RoundtripResult(
        ratio=r,
        regenerated_prose=prose,
        diff=diff,
        dsl_valid=dsl_valid,
        flagged_for_review=flagged,
        reasons=reasons,
    )


def write_review_artifacts(
    section_id: str,
    section_text: str,
    yaml_text: str,
    result: RoundtripResult,
    out_dir: str | Path,
) -> Path:
    """Write the review bundle for a single section and return its directory."""
    d = Path(out_dir) / section_id.replace(".", "_")
    d.mkdir(parents=True, exist_ok=True)
    (d / "original.txt").write_text(section_text, encoding="utf-8")
    (d / "draft.yaml").write_text(yaml_text, encoding="utf-8")
    (d / "regenerated_prose.txt").write_text(result.regenerated_prose, encoding="utf-8")
    (d / "diff.txt").write_text(result.diff, encoding="utf-8")
    (d / "status.txt").write_text(
        f"ratio={result.ratio:.3f}\n"
        f"dsl_valid={result.dsl_valid}\n"
        f"flagged_for_review={result.flagged_for_review}\n"
        + "\n".join(f"reason: {r}" for r in result.reasons),
        encoding="utf-8",
    )
    return d


__all__ = ["dsl_to_prose", "roundtrip_check", "RoundtripResult", "write_review_artifacts"]
