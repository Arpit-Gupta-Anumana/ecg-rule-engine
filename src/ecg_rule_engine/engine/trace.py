"""Human-readable formatters for rule fire traces."""

from __future__ import annotations

from ..dsl.schema import Disease, Variant
from .evaluator import DiseaseResult, TraceNode, VariantResult


def format_trace(node: TraceNode, indent: int = 0) -> str:
    prefix = "  " * indent
    marker = "[x]" if node.fired else "[ ]"
    note = f"  ({node.note})" if node.note else ""
    lines = [f"{prefix}{marker} {node.kind}: {node.detail}{note}"]
    if node.missing_features:
        lines.append(f"{prefix}    missing: {', '.join(node.missing_features)}")
    for child in node.children:
        lines.append(format_trace(child, indent + 1))
    return "\n".join(lines)


def format_variant(result: VariantResult, variant: Variant | None = None) -> str:
    header = f"Variant {result.variant_name}: {'FIRED' if result.fired else 'not fired'}"
    if variant is not None and variant.manual_page is not None:
        header += f"  (manual p.{variant.manual_page})"
    return header + "\n" + format_trace(result.trace, indent=1)


def format_disease_result(
    result: DiseaseResult,
    disease: Disease | None = None,
) -> str:
    lines: list[str] = []
    header = f"Disease {result.disease}: {'FIRED' if result.fired else 'not fired'}"
    if disease is not None:
        header += f"  [{disease.manual_source}]"
    lines.append(header)
    if result.excluded_by:
        lines.append(f"  Excluded by: {', '.join(result.excluded_by)}")
    variants_by_name = {v.name: v for v in (disease.variants if disease else [])}
    for vr in result.variant_results:
        lines.append(format_variant(vr, variants_by_name.get(vr.variant_name)))
    return "\n".join(lines)


__all__ = ["format_trace", "format_variant", "format_disease_result"]
