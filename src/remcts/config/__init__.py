"""Packaged YAML configurations.

``load_config("remcts_known")`` and ``load_config("path/to/my.yaml")`` both work: a value that is
not an existing file is looked up among the packaged configs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

from remcts.core.state import SearchConfig

CONFIG_DIR = Path(__file__).resolve().parent


def available_configs() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def resolve_config_path(name_or_path: str | Path) -> Path:
    path = Path(name_or_path)
    if path.is_file():
        return path
    packaged = CONFIG_DIR / (path.name if path.suffix in {".yaml", ".yml"} else f"{path.name}.yaml")
    if packaged.is_file():
        return packaged
    raise FileNotFoundError(f"Config {name_or_path!r} not found. Packaged configs: {', '.join(available_configs())}")


def load_config_mapping(
    name_or_path: str | Path,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Read a YAML config and apply section-level overrides such as ``{"search": {"path_memory_k": 0}}``."""
    mapping = yaml.safe_load(resolve_config_path(name_or_path).read_text(encoding="utf-8")) or {}
    mapping = copy.deepcopy(mapping)
    for section, values in (overrides or {}).items():
        mapping[section] = {**(mapping.get(section) or {}), **values}
    return mapping


def load_config(
    name_or_path: str | Path,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> SearchConfig:
    return SearchConfig.from_mapping(load_config_mapping(name_or_path, overrides))
