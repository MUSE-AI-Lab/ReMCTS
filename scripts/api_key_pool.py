from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Any


QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@dataclass(frozen=True)
class ApiProfile:
    provider: str
    api_key: str
    base_url: str
    model_name: str
    fingerprint: str
    index: int

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "fingerprint": self.fingerprint,
            "index": self.index,
        }


def load_key_pool(path: Path | str | None, model_aliases: list[str] | None = None) -> dict[str, list[ApiProfile]]:
    if not path:
        return {}
    key_path = Path(path).expanduser()
    if not key_path.exists():
        return {}
    raw = key_path.read_text(encoding="utf-8", errors="replace")
    keys_by_provider: dict[str, list[str]] = {"qwen": [], "deepseek": []}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.search(r"sk-[A-Za-z0-9_-]+", stripped)
        if not match:
            continue
        provider = provider_from_text(stripped)
        if provider not in keys_by_provider:
            continue
        keys_by_provider[provider].append(match.group(0))

    aliases = model_aliases or ["qwen3-8b", "qwen3-32b", "deepseek-v4-flash"]
    pool: dict[str, list[ApiProfile]] = {}
    for alias in aliases:
        provider = provider_for_model(alias)
        base_url = base_url_for_provider(provider)
        profiles = []
        for index, api_key in enumerate(keys_by_provider.get(provider, [])):
            profiles.append(
                ApiProfile(
                    provider=provider,
                    api_key=api_key,
                    base_url=base_url,
                    model_name=alias,
                    fingerprint=fingerprint_key(api_key),
                    index=index,
                )
            )
        if profiles:
            pool[alias] = profiles
    return pool


def provider_from_text(text: str) -> str:
    lowered = text.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "qwen" in lowered:
        return "qwen"
    return "unknown"


def provider_for_model(model_alias: str) -> str:
    lowered = model_alias.lower()
    if lowered.startswith("deepseek"):
        return "deepseek"
    return "qwen"


def base_url_for_provider(provider: str) -> str:
    if provider == "deepseek":
        return DEEPSEEK_BASE_URL
    return QWEN_BASE_URL


def fingerprint_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()[:8]


def select_profile(
    pool: dict[str, list[ApiProfile]],
    model_alias: str,
    task_index: int,
    attempt: int = 0,
) -> ApiProfile | None:
    profiles = pool.get(model_alias) or []
    if not profiles:
        return None
    base = max(int(task_index), 0)
    offset = max(int(attempt), 0)
    return profiles[(base + offset) % len(profiles)]


def sanitize_rotation_stats(stats: dict[str, Any] | None) -> dict[str, Any]:
    stats = dict(stats or {})
    stats.pop("api_key", None)
    return stats

