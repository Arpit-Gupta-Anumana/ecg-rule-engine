"""YAML -> validated `Disease` loader for the Rule DSL."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import yaml
from pydantic import ValidationError

from .schema import Disease


class RuleLoadError(ValueError):
    """Raised when a YAML rule file cannot be loaded or validated."""


def load_disease_yaml(path: str | Path) -> Disease:
    """Load a single disease YAML file, returning a validated `Disease`."""
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise RuleLoadError(f"YAML parse failed for {p}: {e}") from e
    if not isinstance(raw, dict):
        raise RuleLoadError(f"{p}: top-level YAML must be a mapping, got {type(raw).__name__}")
    try:
        return Disease.model_validate(raw)
    except ValidationError as e:
        raise RuleLoadError(f"Schema validation failed for {p}:\n{e}") from e


def load_rules_dir(rules_dir: str | Path) -> dict[str, Disease]:
    """Load every *.yaml / *.yml file in a directory into a {disease_name: Disease} map.

    Raises if two files declare the same `disease:` name.
    """
    d = Path(rules_dir)
    if not d.is_dir():
        raise RuleLoadError(f"Rules directory does not exist: {d}")
    out: dict[str, Disease] = {}
    paths: Iterable[Path] = sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml")))
    for p in paths:
        disease = load_disease_yaml(p)
        if disease.disease in out:
            raise RuleLoadError(
                f"Duplicate disease name {disease.disease!r} in {p} "
                f"(already defined elsewhere in {d})"
            )
        out[disease.disease] = disease
    return out


__all__ = ["load_disease_yaml", "load_rules_dir", "RuleLoadError"]
