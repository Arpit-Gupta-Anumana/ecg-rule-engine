#!/usr/bin/env bash
# Orchestrate the full 12SL extraction. Requires OPENAI_API_KEY and
# ANTHROPIC_API_KEY in the environment.
#
# Stages:
#   1. Segment the PDF into per-section text files (idempotent, no API calls).
#   2. Run the LLM transcription + round-trip QA driver on every section.
#      Flagged sections land in extracted/review_queue/; validated YAMLs land
#      in extracted/rules/.
#   3. After human review, promote extracted/rules/*.yaml into the project
#      rules/ directory with the `promote_rule.py` helper.
#
# Usage:
#   scripts/extract_all_rules.sh [MAX_SECTIONS]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PDF="${PDF:-$HOME/Downloads/12SL Physicians Guide v24_UM_2056246-007_2.pdf}"
OUT="$ROOT/extracted"
MAX="${1:-}"  # optional: pilot cap

if [[ ! -f "$PDF" ]]; then
    echo "PDF not found: $PDF" >&2
    exit 1
fi

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY}"
: "${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY}"

echo "[1/3] Segmenting PDF into sections..."
python "$ROOT/scripts/segment_pdf.py" --pdf "$PDF" --out "$OUT/sections" > /dev/null

echo "[2/3] Running LLM extraction + roundtrip QA..."
ARGS=(
    --pdf "$PDF"
    --out-root "$OUT"
    --provider openai
    --prose-provider anthropic
    --min-ratio 0.55
)
if [[ -n "$MAX" ]]; then
    ARGS+=(--max-sections "$MAX")
fi
python -m ecg_rule_engine.extraction.driver "${ARGS[@]}"

echo "[3/3] Done. Inspect:"
echo "  $OUT/rules/         (validated YAMLs)"
echo "  $OUT/review_queue/  (flagged — open notebooks/02_extraction_qa.ipynb)"
echo ""
echo "After you sign off, copy the validated YAMLs into $ROOT/rules/ or use"
echo "scripts/promote_rule.py to move them one at a time."
