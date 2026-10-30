"""Read API profiles from a local credentials file (``llm.credentials_file``).

Accepted layouts:

* one profile as ``key: value`` or ``key=value`` lines, optionally mixed with bare values
  (an API key, a ``http(s)://`` base URL and a model name);
* several profiles separated by blank lines or introduced by ``[alias]`` headers, with an
  optional ``default_alias`` line;
* JSON: a profile object, a list of profiles, or an object that holds profiles either under
  a container key (``apis``, ``profiles``, ...) or keyed by alias.

A profile has the fields ``alias``, ``api_key``, ``base_url``, ``model_name`` and
``enable_thinking``. When a file defines several profiles, the requested alias (or the
file's ``default_alias``) selects one of them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import LLMClientError

_FIELD_SPELLINGS = {
    "api_key": {"api_key", "apikey", "key", "token", "llm_api_key", "remcts_api_key", "openai_api_key"},
    "base_url": {"base_url", "api_base", "api_base_url", "url", "endpoint"},
    "model_name": {"model", "model_name", "llm_model", "llm_model_name"},
    "alias": {"alias", "api_alias", "name", "profile"},
    "default_alias": {"default", "default_alias", "active_alias", "selected_alias"},
    "enable_thinking": {"enable_thinking", "thinking"},
}
_FIELDS = {spelling: field for field, spellings in _FIELD_SPELLINGS.items() for spelling in spellings}
_CREDENTIAL_FIELDS = {"api_key", "base_url", "model_name"}
_PROFILE_CONTAINERS = {"apis", "api_profiles", "profiles", "providers", "credentials"}


def load_credentials_file(file_path: str | None, alias: str | None = None) -> dict[str, str]:
    """Return the selected profile, or ``{}`` when no file is configured or it does not exist."""
    if not file_path:
        return {}
    path = Path(os.path.expandvars(os.path.expanduser(file_path)))
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    if raw[0] == "{" or (raw[0] == "[" and not _contains_section_headers(raw)):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"Invalid JSON credentials file: {path}") from exc
        if not isinstance(parsed, (dict, list)):
            raise LLMClientError(f"Credentials file must contain an object or a list: {path}")
        return _select_json_profile(parsed, alias, path)

    if _contains_section_headers(raw):
        profiles, default_alias = _parse_sectioned_profiles(raw)
    else:
        profiles, default_alias = _parse_blank_separated_profiles(raw)
    if profiles:
        return _select_profile(profiles, alias or default_alias, path)

    single = _parse_key_values(raw)
    if single:
        for key, value in _parse_bare_values(_bare_lines(raw)).items():
            single.setdefault(key, value)
    else:
        single = _parse_bare_values(raw)
    _check_single_profile_alias(single, alias, path)
    return single


def _field(key: Any) -> str | None:
    return _FIELDS.get(str(key).strip().lower().replace("-", "_"))


def _clean(value: str) -> str:
    return value.strip().strip(",").strip().strip('"').strip("'")


def _split_key_value(line: str) -> tuple[str, str] | None:
    separator = "=" if "=" in line else ":" if ":" in line else ""
    if not separator:
        return None
    key, value = line.split(separator, 1)
    return key, value


def _content_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]


def _parse_key_values(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _content_lines(raw):
        pair = _split_key_value(line)
        if pair is None:
            continue
        field, value = _field(pair[0]), _clean(pair[1])
        if field and value:
            values[field] = value
    return values


def _parse_bare_values(raw: str) -> dict[str, str]:
    """``sk-key [base_url] [model]`` on one line, or one value per line."""
    lines = _content_lines(raw)
    if not lines:
        return {}
    parts = lines[0].split() if len(lines) == 1 else lines
    values = {"api_key": _clean(parts[0])}
    for part in parts[1:]:
        value = _clean(part)
        if value.startswith(("http://", "https://")):
            values["base_url"] = value
        elif value and "model_name" not in values:
            values["model_name"] = value
    return values


def _bare_lines(raw: str) -> str:
    """Lines of ``raw`` that are not recognised ``key: value`` lines."""
    kept = []
    for line in _content_lines(raw):
        pair = _split_key_value(line)
        if pair is not None and _field(pair[0]):
            continue
        kept.append(line)
    return "\n".join(kept)


def _contains_section_headers(raw: str) -> bool:
    return any(_is_section_header(line.strip()) for line in raw.splitlines())


def _is_section_header(line: str) -> bool:
    if len(line) < 3 or not line.startswith("[") or not line.endswith("]"):
        return False
    name = line[1:-1].strip()
    return bool(name) and not any(char in name for char in '{}":,')


def _has_profile_content(profile: dict[str, str]) -> bool:
    return bool(profile.get("alias")) and any(key in profile for key in _CREDENTIAL_FIELDS)


def _is_default_alias_only(values: dict[str, str]) -> bool:
    return bool(values.get("default_alias")) and not any(key in values for key in ("alias", *_CREDENTIAL_FIELDS))


def _parse_sectioned_profiles(raw: str) -> tuple[list[dict[str, str]], str | None]:
    profiles: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    default_alias: str | None = None
    for line in _content_lines(raw):
        if _is_section_header(line):
            if current and _has_profile_content(current):
                profiles.append(current)
            current = {"alias": _clean(line[1:-1])}
            continue
        values = _parse_key_values(line)
        if not values:
            continue
        if _is_default_alias_only(values):
            default_alias = values["default_alias"]
            continue
        current = current or {}
        current.update(values)
    if current and _has_profile_content(current):
        profiles.append(current)
    return profiles, default_alias


def _parse_blank_separated_profiles(raw: str) -> tuple[list[dict[str, str]], str | None]:
    profiles: list[dict[str, str]] = []
    default_alias: str | None = None
    for block in _blocks(raw):
        values = _parse_key_values(block)
        if not values:
            continue
        if _is_default_alias_only(values):
            default_alias = values["default_alias"]
        elif values.get("alias") and _has_profile_content(values):
            profiles.append(values)
    return profiles, default_alias


def _blocks(raw: str) -> list[str]:
    blocks: list[list[str]] = [[]]
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            if blocks[-1]:
                blocks.append([])
        elif not line.startswith("#"):
            blocks[-1].append(line)
    return ["\n".join(block) for block in blocks if block]


def _normalize_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    if isinstance(mapping.get("llm"), dict):
        nested = _normalize_mapping(mapping["llm"])
        return {**nested, **_normalize_mapping({k: v for k, v in mapping.items() if k != "llm"})}
    values: dict[str, str] = {}
    for key, value in mapping.items():
        field = _field(key)
        if field and value is not None:
            values[field] = _clean(str(value))
    return values


def _profiles_from_container(container: Any) -> list[dict[str, str]]:
    if isinstance(container, list):
        return [_normalize_mapping(item) for item in container if isinstance(item, dict)]
    if isinstance(container, dict):
        return [_keyed_profile(key, value) for key, value in container.items() if isinstance(value, dict)]
    raise LLMClientError("Credentials profile container must be a list or an object.")


def _keyed_profile(key: Any, value: dict[str, Any]) -> dict[str, str]:
    profile = _normalize_mapping(value)
    profile.setdefault("alias", _clean(str(key)))
    return profile


def _has_credential_field(mapping: dict[str, Any]) -> bool:
    return any(_field(key) in _CREDENTIAL_FIELDS for key in mapping)


def _default_alias(mapping: dict[str, Any]) -> str | None:
    for key, value in mapping.items():
        if _field(key) == "default_alias" and value is not None:
            return _clean(str(value))
    return None


def _select_json_profile(parsed: dict[str, Any] | list[Any], alias: str | None, path: Path) -> dict[str, str]:
    if isinstance(parsed, list):
        return _select_profile(_profiles_from_container(parsed), alias, path)

    if isinstance(parsed.get("llm"), dict) and not _has_credential_field(parsed):
        nested = dict(parsed["llm"])
        for key, value in parsed.items():
            if _field(key) == "default_alias" and key not in nested:
                nested[key] = value
        return _select_json_profile(nested, alias, path)

    default_alias = _default_alias(parsed)
    for key, value in parsed.items():
        if str(key).strip().lower().replace("-", "_") in _PROFILE_CONTAINERS:
            return _select_profile(_profiles_from_container(value), alias or default_alias, path)

    keyed = [
        _keyed_profile(key, value)
        for key, value in parsed.items()
        if _field(key) != "default_alias" and isinstance(value, dict)
    ]
    if keyed and not _has_credential_field(parsed):
        return _select_profile(keyed, alias or default_alias, path)

    single = _normalize_mapping(parsed)
    _check_single_profile_alias(single, alias, path)
    return _without_default_alias(single)


def _without_default_alias(profile: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in profile.items() if key != "default_alias"}


def _format_aliases(profiles: list[dict[str, str]]) -> str:
    aliases = sorted(profile["alias"] for profile in profiles if profile.get("alias"))
    return ", ".join(aliases) if aliases else "none"


def _select_profile(entries: list[dict[str, str]], alias: str | None, path: Path) -> dict[str, str]:
    profiles = [_without_default_alias(entry) for entry in entries if _has_profile_content(entry)]
    if not profiles:
        return {}
    selected = _clean(alias or "")
    if selected:
        for profile in profiles:
            if profile.get("alias") == selected:
                return profile
        raise LLMClientError(
            f"API alias {selected!r} was not found in {path}. Available aliases: {_format_aliases(profiles)}."
        )
    if len(profiles) == 1:
        return profiles[0]
    raise LLMClientError(
        f"Multiple API profiles were found in {path}. Pass --api-alias or configure llm.api_alias. "
        f"Available aliases: {_format_aliases(profiles)}."
    )


def _check_single_profile_alias(profile: dict[str, str], alias: str | None, path: Path) -> None:
    selected = _clean(alias or "")
    if not selected or not profile or profile.get("alias") == selected:
        return
    available = _format_aliases([profile]) if profile.get("alias") else "none"
    raise LLMClientError(
        f"API alias {selected!r} was requested, but {path} does not define it. Available aliases: {available}."
    )
