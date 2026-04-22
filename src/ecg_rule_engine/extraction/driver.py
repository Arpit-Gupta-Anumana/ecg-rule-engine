"""Drive the PDF -> DSL pipeline over all sections of the 12SL manual.

Per section it:
1. calls `transcribe_section` (LLM#1) to produce draft YAML,
2. re-validates against our DSL schema locally (belt-and-suspenders),
3. calls `roundtrip_check` (LLM#2) to verify the YAML round-trips to prose
   that resembles the original section,
4. writes:
     - a `draft/*.yaml` file (even if invalid, for debugging),
     - a `rules/*.yaml` file ONLY if DSL validates AND roundtrip ratio >= min_ratio,
     - a `review_queue/<section>/` folder for anything flagged.

Idempotent: skips sections that already have a final YAML unless `--force` is
passed.

Budget-aware: accepts a `max_sections` kwarg so pilot runs don't blow through
API quota.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from .llm_transcribe import TranscriptionResult, transcribe_section
from .pdf_to_sections import PdfSection, segment_pdf
from .roundtrip_check import RoundtripResult, roundtrip_check, write_review_artifacts

log = logging.getLogger(__name__)


@dataclass
class DriverOutcome:
    section_id: str
    title: str
    accepted: bool
    final_yaml_path: Path | None
    review_path: Path | None
    transcription_error: str | None
    ratio: float | None


def _slug(section: PdfSection) -> str:
    return f"{section.section_id.replace('.', '_')}__{section.slug()}"


def process_section(
    section: PdfSection,
    *,
    out_rules: Path,
    out_draft: Path,
    out_review: Path,
    provider: str = "openai",
    model: str | None = None,
    prose_provider: str = "anthropic",
    prose_model: str | None = None,
    min_ratio: float = 0.55,
    force: bool = False,
) -> DriverOutcome:
    slug = _slug(section)
    final_path = out_rules / f"{slug}.yaml"
    draft_path = out_draft / f"{slug}.yaml"

    if final_path.exists() and not force:
        log.info("Skip %s (already finalized).", section.section_id)
        return DriverOutcome(section.section_id, section.title, True, final_path, None, None, None)

    section_blob = section.header() + "\n" + section.text

    tres: TranscriptionResult = transcribe_section(
        section_blob, provider=provider, model=model,
    )
    out_draft.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(tres.yaml_text, encoding="utf-8")

    if tres.disease is None:
        review_dir = write_review_artifacts(
            section.section_id,
            section_blob,
            tres.yaml_text,
            RoundtripResult(
                ratio=0.0, regenerated_prose="", diff="",
                dsl_valid=False, flagged_for_review=True,
                reasons=[tres.error or "unknown transcription error"],
            ),
            out_review,
        )
        return DriverOutcome(
            section.section_id, section.title, False, None, review_dir, tres.error, None,
        )

    rr: RoundtripResult = roundtrip_check(
        original_text=section_blob,
        yaml_text=tres.yaml_text,
        min_ratio=min_ratio,
        prose_provider=prose_provider,
        prose_model=prose_model,
    )

    if rr.flagged_for_review:
        review_dir = write_review_artifacts(
            section.section_id, section_blob, tres.yaml_text, rr, out_review,
        )
        return DriverOutcome(
            section.section_id, section.title, False, None, review_dir, None, rr.ratio,
        )

    # All checks passed — write final
    out_rules.mkdir(parents=True, exist_ok=True)
    # Re-dump through pydantic for canonical formatting
    canonical = yaml.safe_dump(
        tres.disease.model_dump(mode="json", exclude_none=True),
        sort_keys=False,
    )
    final_path.write_text(canonical, encoding="utf-8")
    return DriverOutcome(
        section.section_id, section.title, True, final_path, None, None, rr.ratio,
    )


def run(
    pdf_path: str,
    *,
    out_root: str,
    provider: str = "openai",
    model: str | None = None,
    prose_provider: str = "anthropic",
    prose_model: str | None = None,
    min_ratio: float = 0.55,
    max_sections: int | None = None,
    only_sections: list[str] | None = None,
    force: bool = False,
) -> list[DriverOutcome]:
    root = Path(out_root)
    out_rules = root / "rules"
    out_draft = root / "draft"
    out_review = root / "review_queue"

    sections = segment_pdf(pdf_path)
    if only_sections:
        sections = [s for s in sections if s.section_id in set(only_sections)]
    if max_sections is not None:
        sections = sections[:max_sections]

    outcomes: list[DriverOutcome] = []
    for sec in sections:
        try:
            out = process_section(
                sec,
                out_rules=out_rules,
                out_draft=out_draft,
                out_review=out_review,
                provider=provider,
                model=model,
                prose_provider=prose_provider,
                prose_model=prose_model,
                min_ratio=min_ratio,
                force=force,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("Section %s failed: %s", sec.section_id, e)
            out = DriverOutcome(sec.section_id, sec.title, False, None, None, str(e), None)
        outcomes.append(out)
        status = "OK  " if out.accepted else "FLAG"
        ratio = f"{out.ratio:.2f}" if out.ratio is not None else "  -  "
        log.info("%s  %s  %s  %s", status, sec.section_id, ratio, sec.title)
    return outcomes


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full PDF -> DSL pipeline.")
    p.add_argument("--pdf", required=True)
    p.add_argument("--out-root", required=True, help="Directory under which rules/, draft/, review_queue/ are written")
    p.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--model", default=None)
    p.add_argument("--prose-provider", default="anthropic", choices=["openai", "anthropic"])
    p.add_argument("--prose-model", default=None)
    p.add_argument("--min-ratio", type=float, default=0.55)
    p.add_argument("--max-sections", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None,
                   help="Only process these section IDs (e.g. 3.7.2.9.2)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    outcomes = run(
        pdf_path=args.pdf,
        out_root=args.out_root,
        provider=args.provider,
        model=args.model,
        prose_provider=args.prose_provider,
        prose_model=args.prose_model,
        min_ratio=args.min_ratio,
        max_sections=args.max_sections,
        only_sections=args.only,
        force=args.force,
    )
    accepted = sum(1 for o in outcomes if o.accepted)
    print(f"\n{accepted}/{len(outcomes)} sections accepted; {len(outcomes) - accepted} in review queue.")


if __name__ == "__main__":
    main()
