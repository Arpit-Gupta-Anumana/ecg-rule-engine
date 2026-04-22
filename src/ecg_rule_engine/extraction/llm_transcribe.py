"""LLM-assisted transcription of a PDF section into Rule DSL YAML.

The LLM is used strictly as a *transcriber*: it reads the manual text and emits
a draft YAML that conforms to `ecg_rule_engine.dsl.schema.Disease`. We enforce
schema via JSON Schema (provided to the model) and then re-validate the output
with pydantic on our side. Unknown features / malformed expressions are hard
errors — the YAML is rejected, moved to a review queue, and a human fixes it.

Providers (pluggable):
- `openai`    via env OPENAI_API_KEY
- `anthropic` via env ANTHROPIC_API_KEY

Call `transcribe_section(text, provider="openai", model=...)`.
"""

from __future__ import annotations

import json
import os
import textwrap
from dataclasses import dataclass
from typing import Literal

import yaml

from ..dsl.loader import RuleLoadError
from ..dsl.schema import Disease
from ..features.registry import all_feature_names

Provider = Literal["openai", "anthropic"]


# The system prompt: explicit, deterministic rules for the model.
SYSTEM_PROMPT = """\
You are a transcriber. You translate ECG interpretation criteria from the GE
Marquette 12SL Physician's Guide into a strict YAML "Rule DSL". You do NOT
invent criteria. You do NOT simplify. You do NOT drop conditions. If the manual
text is ambiguous, emit a comment in a top-level `description:` field noting
the ambiguity, and encode the most literal reading.

Output rules:
- Emit ONLY a YAML document conforming to the provided JSON Schema.
- Units in the DSL: amplitudes in mV, durations in ms, axes in degrees.
  The manual often uses uV and you MUST convert (e.g. "> 1100 µV" -> 1.1 mV).
- Feature names MUST come from the allowed list you are given. Do not invent names.
- Every variant MUST cite `manual_page` (use the page numbers in the header of
  the input text).
- For sex-specific thresholds use {M: ..., F: ...}. Do NOT bake sex into the
  expression; keep the expression sex-agnostic and put sex differences in `value`.
- Include `exclusions:` for any disease the manual says to suppress (e.g. RBBB,
  LBBB, WPW, pacing). Use short canonical names: LBBB, RBBB, WPW, VentricularPacing.
- If a criterion is a point-scoring test (e.g. Romhilt-Estes), use `kind: point_score`.
- If you encounter something you cannot encode faithfully, emit the YAML with a
  top-level `description:` field starting with "UNSUPPORTED:" explaining why,
  and OMIT that variant.
"""


USER_TEMPLATE = """\
Transcribe the following section of the GE 12SL Physician's Guide into Rule DSL YAML.

=== ALLOWED FEATURE NAMES (use these ONLY) ===
{features}

=== SECTION TEXT ===
{section_text}

Emit the YAML now. No prose, no markdown fences — just the YAML document.
"""


@dataclass
class TranscriptionResult:
    yaml_text: str
    disease: Disease | None
    error: str | None


def _disease_json_schema() -> dict:
    """Return the pydantic JSON schema for a Disease."""
    return Disease.model_json_schema()


def _build_user_prompt(section_text: str) -> str:
    feats = ", ".join(sorted(all_feature_names()))
    # keep prompt size bounded
    feats_wrapped = "\n".join(textwrap.wrap(feats, width=110))
    return USER_TEMPLATE.format(features=feats_wrapped, section_text=section_text)


# --- provider adapters -------------------------------------------------------

def _call_openai(system: str, user: str, model: str) -> str:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def _call_anthropic(system: str, user: str, model: str) -> str:
    import anthropic  # type: ignore

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    return "".join(parts)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        # remove first and last fence lines
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


# --- public API --------------------------------------------------------------

def transcribe_section(
    section_text: str,
    *,
    provider: Provider = "openai",
    model: str | None = None,
) -> TranscriptionResult:
    """Ask an LLM to transcribe a section into Rule DSL YAML.

    Returns the raw YAML text, a validated `Disease` if parseable, and any error.
    """
    system = SYSTEM_PROMPT + "\n\n=== JSON SCHEMA (informational) ===\n" + json.dumps(
        _disease_json_schema()
    )
    user = _build_user_prompt(section_text)

    if provider == "openai":
        raw = _call_openai(system, user, model or "gpt-4.1-mini")
    elif provider == "anthropic":
        raw = _call_anthropic(system, user, model or "claude-3-5-sonnet-latest")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    yaml_text = _strip_fences(raw)

    # Try to validate
    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return TranscriptionResult(yaml_text=yaml_text, disease=None, error=f"YAML parse error: {e}")

    if not isinstance(data, dict):
        return TranscriptionResult(
            yaml_text=yaml_text, disease=None,
            error=f"Top-level YAML must be a mapping, got {type(data).__name__}",
        )

    try:
        disease = Disease.model_validate(data)
    except Exception as e:  # pydantic ValidationError etc
        return TranscriptionResult(yaml_text=yaml_text, disease=None, error=f"DSL validation: {e}")

    return TranscriptionResult(yaml_text=yaml_text, disease=disease, error=None)


__all__ = ["transcribe_section", "TranscriptionResult", "Provider", "SYSTEM_PROMPT"]
