"""API key pool for the experiment drivers.

A key-pool file holds one provider-labelled key per line, e.g. ``qwen: sk-...`` or
``deepseek: sk-...``. Tasks are spread over the keys of their provider by a stable hash of
``(label, task index)``, and a task moves to the next key after a retryable request failure.
Only short SHA-256 fingerprints of the keys are written to reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import re
from typing import Any

QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]+")
_RETRYABLE_MARKERS = (
    "429", "408", "500", "502", "503", "504", "rate limit", "timeout", "timed out",
    "temporarily", "connection", "remote end closed", "incomplete", "reset",
)


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


def provider_for_model(model_alias: str) -> str | None:
    alias = model_alias.lower()
    if alias.startswith("deepseek"):
        return "deepseek"
    if alias.startswith("qwen"):
        return "qwen"
    return None


def base_url_for_model(model_alias: str) -> str | None:
    """Endpoint of the providers used in the paper; None for other models (keep the config's URL)."""
    return {"deepseek": DEEPSEEK_BASE_URL, "qwen": QWEN_BASE_URL}.get(provider_for_model(model_alias) or "")


def fingerprint_key(api_key: str) -> str:
    return sha256(api_key.encode("utf-8")).hexdigest()[:8]


def _provider_from_line(line: str) -> str | None:
    lowered = line.lower()
    if "deepseek" in lowered:
        return "deepseek"
    if "qwen" in lowered:
        return "qwen"
    return None


def load_key_pool(path: str | Path | None, model_alias: str) -> list[ApiProfile]:
    """All profiles in ``path`` usable for ``model_alias`` (empty if the file is missing)."""
    if not path:
        return []
    return list(_load_key_pool(str(Path(path).expanduser()), model_alias))


@lru_cache(maxsize=None)
def _load_key_pool(path: str, model_alias: str) -> tuple[ApiProfile, ...]:
    key_path = Path(path)
    provider = provider_for_model(model_alias)
    if provider is None or not key_path.is_file():
        return ()
    keys = []
    for line in key_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        match = _KEY_PATTERN.search(line)
        if line and not line.startswith("#") and match and _provider_from_line(line) == provider:
            keys.append(match.group(0))
    return tuple(
        ApiProfile(
            provider=provider,
            api_key=key,
            base_url=base_url_for_model(model_alias) or QWEN_BASE_URL,
            model_name=model_alias,
            fingerprint=fingerprint_key(key),
            index=index,
        )
        for index, key in enumerate(keys)
    )


def stable_key_slot(label: str, task_index: int) -> int:
    return int(sha256(f"{label}:{int(task_index)}".encode("utf-8")).hexdigest()[:12], 16)


def select_profile(profiles: list[ApiProfile], slot: int, attempt: int = 0) -> ApiProfile | None:
    if not profiles:
        return None
    return profiles[(max(int(slot), 0) + max(int(attempt), 0)) % len(profiles)]


def is_retryable_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)
