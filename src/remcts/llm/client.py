from __future__ import annotations

from abc import ABC, abstractmethod
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def batch_generate(self, prompts: list[str], **kwargs) -> list[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def reset_usage(self) -> None:
        return None

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_sec": 0.0,
            "records": [],
        }


class MockLLMClient(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        if "score" in prompt.lower() and "json" in prompt.lower():
            return '{"score": 0.2, "reason": "mock semantic score"}'
        return """```python
def solution(*args, **kwargs):
    raise NotImplementedError(\"MockLLMClient placeholder\")
```"""


class LLMClientError(RuntimeError):
    pass


_API_KEY_FIELDS = {
    "api_key",
    "apikey",
    "key",
    "token",
    "llm_api_key",
    "remcts_api_key",
    "openai_api_key",
}
_BASE_URL_FIELDS = {"base_url", "api_base", "api_base_url", "url", "endpoint"}
_MODEL_FIELDS = {"model", "model_name", "llm_model", "llm_model_name"}
_ALIAS_FIELDS = {"alias", "api_alias", "name", "profile"}
_DEFAULT_ALIAS_FIELDS = {"default", "default_alias", "active_alias", "selected_alias"}
_PROFILE_CONTAINER_FIELDS = {"apis", "api_profiles", "profiles", "providers", "credentials"}
_ENABLE_THINKING_FIELDS = {"enable_thinking", "thinking"}


class OpenAICompatibleLLMClient(BaseLLMClient):
    def __init__(
        self,
        model_name: str,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "REMCTS_API_KEY",
        api_alias: str | None = None,
        enable_thinking: bool | None = None,
        timeout_sec: int = 60,
        max_retries: int = 2,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
        self.api_key_env = api_key_env
        self.api_alias = api_alias
        self.enable_thinking = enable_thinking
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self._usage_records: list[dict[str, Any]] = []
        if not self.api_key:
            raise LLMClientError(
                f"Missing API key. Set {api_key_env}, OPENAI_API_KEY, "
                "or configure llm.credentials_file with an api_key."
            )

    def reset_usage(self) -> None:
        self._usage_records = []

    def usage_snapshot(self) -> dict[str, Any]:
        prompt_tokens = sum(_safe_int(record.get("prompt_tokens")) for record in self._usage_records)
        completion_tokens = sum(_safe_int(record.get("completion_tokens")) for record in self._usage_records)
        total_tokens = sum(_safe_int(record.get("total_tokens")) for record in self._usage_records)
        if total_tokens == 0:
            total_tokens = prompt_tokens + completion_tokens
        elapsed_sec = sum(float(record.get("elapsed_sec", 0.0)) for record in self._usage_records)
        return {
            "model": self.model_name,
            "api_alias": self.api_alias,
            "calls": len(self._usage_records),
            "successful_calls": sum(1 for record in self._usage_records if record.get("status") == "success"),
            "failed_calls": sum(1 for record in self._usage_records if record.get("status") != "success"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "elapsed_sec": round(elapsed_sec, 6),
            "records": list(self._usage_records),
        }

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful coding assistant. "
                    "Return only complete executable code unless the user explicitly asks for JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
        }
        enable_thinking = self.enable_thinking
        if enable_thinking is None and self.model_name.lower().startswith("qwen3"):
            enable_thinking = False
        payload.update(_build_thinking_payload(self.model_name, enable_thinking))
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            return self._chat_completion(payload)
        except LLMClientError as exc:
            message = str(exc).lower()
            if "max_tokens" in message and max_tokens is not None:
                fallback = dict(payload)
                fallback.pop("max_tokens", None)
                fallback["max_completion_tokens"] = max_tokens
                return self._chat_completion(fallback)
            if "temperature" in message and temperature is not None:
                fallback = dict(payload)
                fallback.pop("temperature", None)
                return self._chat_completion(fallback)
            raise

    def _chat_completion(self, payload: dict[str, Any]) -> str:
        url = f"{self.base_url}/chat/completions"
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "remcts/0.1 OpenAI-Compatible-Client",
        }
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            request = Request(url, data=data, headers=headers, method="POST")
            started = time.perf_counter()
            try:
                with urlopen(request, timeout=self.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                body = json.loads(raw)
                elapsed = time.perf_counter() - started
                self._record_usage(payload, body, elapsed_sec=elapsed, attempt=attempt, status="success")
                return _extract_chat_content(body)
            except HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="replace")
                last_error = LLMClientError(f"LLM HTTP {exc.code}: {_sanitize_error(raw_error)}")
                elapsed = time.perf_counter() - started
                self._record_usage(
                    payload,
                    None,
                    elapsed_sec=elapsed,
                    attempt=attempt,
                    status=f"http_{exc.code}",
                    error=str(last_error),
                )
                if not _should_retry_http_status(exc.code):
                    break
            except (
                URLError,
                TimeoutError,
                ConnectionResetError,
                IncompleteRead,
                RemoteDisconnected,
                json.JSONDecodeError,
                KeyError,
                IndexError,
            ) as exc:
                last_error = exc
                elapsed = time.perf_counter() - started
                self._record_usage(
                    payload,
                    None,
                    elapsed_sec=elapsed,
                    attempt=attempt,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}",
                )
            if attempt < self.max_retries:
                time.sleep(_retry_delay(attempt))
        raise LLMClientError(str(last_error or "Unknown LLM client error"))

    def _record_usage(
        self,
        payload: dict[str, Any],
        body: dict[str, Any] | None,
        *,
        elapsed_sec: float,
        attempt: int,
        status: str,
        error: str | None = None,
    ) -> None:
        usage = _extract_usage(body or {})
        record: dict[str, Any] = {
            "call_index": len(self._usage_records) + 1,
            "model": str(payload.get("model") or self.model_name),
            "api_alias": self.api_alias,
            "status": status,
            "attempt": attempt,
            "elapsed_sec": round(elapsed_sec, 6),
            **usage,
        }
        if error:
            record["error"] = error[:1000]
        self._usage_records.append(record)


def _build_thinking_payload(model_name: str, enable_thinking: bool | None) -> dict[str, Any]:
    if enable_thinking is None:
        return {}

    model_name_lower = model_name.lower()
    if model_name_lower.startswith("deepseek-v4"):
        return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}

    # GPT-5 reasoning models exposed via Chat Completions commonly expect
    # reasoning_effort instead of provider-specific enable_thinking flags.
    # For disabling thinking, use the lowest-effort official value accepted by
    # OpenAI-compatible providers we target here.
    if model_name_lower.startswith("gpt-5"):
        if enable_thinking:
            return {}
        return {"reasoning_effort": "none"}

    return {"enable_thinking": enable_thinking}


def _extract_chat_content(body: dict[str, Any]) -> str:
    choice = body["choices"][0]
    if "message" in choice:
        content = choice["message"].get("content", "")
    else:
        content = choice.get("text", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or ""
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _sanitize_error(raw_error: str) -> str:
    try:
        body = json.loads(raw_error)
    except json.JSONDecodeError:
        return raw_error[:1000]
    message = body.get("error", {}).get("message") if isinstance(body.get("error"), dict) else None
    return str(message or body)[:1000]


def _should_retry_http_status(status_code: int) -> bool:
    if status_code in {408, 409, 425, 429}:
        return True
    return status_code >= 500


def _retry_delay(attempt: int) -> float:
    return min(12.0, 0.75 * (2**attempt))


def _extract_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _first_int(
        usage,
        "prompt_tokens",
        "input_tokens",
        "inputTokenCount",
        "input_token_count",
    )
    completion_tokens = _first_int(
        usage,
        "completion_tokens",
        "output_tokens",
        "outputTokenCount",
        "output_token_count",
    )
    total_tokens = _first_int(
        usage,
        "total_tokens",
        "totalTokenCount",
        "total_token_count",
    )
    if total_tokens == 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _first_int(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return _safe_int(value)
    return 0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def create_llm_client(config: Any) -> BaseLLMClient:
    if config.llm_model_name.lower() == "mock":
        return MockLLMClient()
    requested_alias = getattr(config, "llm_api_alias", None)
    credentials = _load_credentials_file(getattr(config, "llm_credentials_file", None), alias=requested_alias)
    return OpenAICompatibleLLMClient(
        model_name=credentials.get("model_name") or config.llm_model_name,
        base_url=credentials.get("base_url") or config.llm_base_url,
        api_key=credentials.get("api_key"),
        api_key_env=config.llm_api_key_env,
        api_alias=credentials.get("alias") or requested_alias,
        enable_thinking=(
            _optional_bool(str(config.llm_enable_thinking))
            if getattr(config, "llm_enable_thinking", None) is not None
            else _optional_bool(credentials.get("enable_thinking"))
        ),
        timeout_sec=config.llm_timeout_sec,
        max_retries=config.llm_max_retries,
    )


def _load_credentials_file(file_path: str | None, alias: str | None = None) -> dict[str, str]:
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
        return _select_json_credentials(parsed, alias, path)

    text_profiles, default_alias = _parse_text_credential_profiles(raw)
    if text_profiles:
        return _select_credentials_profile(text_profiles, alias or default_alias, path)

    parsed = _parse_key_value_credentials(raw)
    if parsed:
        plain = _parse_plain_credentials(_plain_credentials_only(raw))
        for key, value in plain.items():
            parsed.setdefault(key, value)
        _validate_single_credentials_alias(parsed, alias, path)
        return parsed
    plain = _parse_plain_credentials(raw)
    _validate_single_credentials_alias(plain, alias, path)
    return plain


def _select_json_credentials(parsed: dict[str, Any] | list[Any], alias: str | None, path: Path) -> dict[str, str]:
    if isinstance(parsed, list):
        return _select_credentials_profile(_entries_from_profile_container(parsed), alias, path)

    if isinstance(parsed.get("llm"), dict) and not _has_credential_field(parsed):
        nested = dict(parsed["llm"])
        for key, value in parsed.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if normalized_key in _DEFAULT_ALIAS_FIELDS and key not in nested:
                nested[key] = value
        return _select_json_credentials(nested, alias, path)

    default_alias = _extract_default_alias(parsed)
    for key, value in parsed.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in _PROFILE_CONTAINER_FIELDS:
            return _select_credentials_profile(
                _entries_from_profile_container(value),
                alias or default_alias,
                path,
            )

    nested_profiles: list[dict[str, str]] = []
    for key, value in parsed.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in _DEFAULT_ALIAS_FIELDS:
            continue
        if isinstance(value, dict):
            profile = _normalize_credentials_mapping(value)
            profile.setdefault("alias", _clean_credential_value(str(key)))
            nested_profiles.append(profile)

    if nested_profiles and not _has_credential_field(parsed):
        return _select_credentials_profile(nested_profiles, alias or default_alias, path)

    credentials = _normalize_credentials_mapping(parsed)
    _validate_single_credentials_alias(credentials, alias, path)
    return _strip_internal_credential_fields(credentials)


def _entries_from_profile_container(container: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    if isinstance(container, list):
        for item in container:
            if not isinstance(item, dict):
                continue
            entries.append(_normalize_credentials_mapping(item))
        return entries

    if isinstance(container, dict):
        for key, value in container.items():
            if not isinstance(value, dict):
                continue
            profile = _normalize_credentials_mapping(value)
            profile.setdefault("alias", _clean_credential_value(str(key)))
            entries.append(profile)
        return entries

    raise LLMClientError("Credentials profile container must be a list or an object.")


def _parse_text_credential_profiles(raw: str) -> tuple[list[dict[str, str]], str | None]:
    if _contains_section_headers(raw):
        return _parse_sectioned_credentials(raw)
    return _parse_blank_separated_credentials(raw)


def _contains_section_headers(raw: str) -> bool:
    return any(_is_section_header(line.strip()) for line in raw.splitlines())


def _is_section_header(line: str) -> bool:
    if len(line) < 3 or not line.startswith("[") or not line.endswith("]"):
        return False
    section_name = line[1:-1].strip()
    return bool(section_name) and not any(char in section_name for char in '{}":,')


def _parse_sectioned_credentials(raw: str) -> tuple[list[dict[str, str]], str | None]:
    profiles: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    default_alias: str | None = None

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _is_section_header(line):
            if current and _has_profile_content(current):
                profiles.append(current)
            current = {"alias": _clean_credential_value(line[1:-1])}
            continue

        fields = _parse_key_value_credentials(line)
        if not fields:
            continue
        if _is_default_alias_only(fields):
            default_alias = fields["default_alias"]
            continue
        if current is None:
            current = {}
        current.update(fields)

    if current and _has_profile_content(current):
        profiles.append(current)
    return profiles, default_alias


def _parse_blank_separated_credentials(raw: str) -> tuple[list[dict[str, str]], str | None]:
    profiles: list[dict[str, str]] = []
    default_alias: str | None = None
    blocks: list[list[str]] = []
    current: list[str] = []

    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        current.append(line)
    if current:
        blocks.append(current)

    for block in blocks:
        fields = _parse_key_value_credentials("\n".join(block))
        if not fields:
            continue
        if _is_default_alias_only(fields):
            default_alias = fields["default_alias"]
            continue
        if fields.get("alias") and _has_profile_content(fields):
            profiles.append(fields)

    return profiles, default_alias


def _select_credentials_profile(entries: list[dict[str, str]], alias: str | None, path: Path) -> dict[str, str]:
    profiles = [_strip_internal_credential_fields(entry) for entry in entries if _has_profile_content(entry)]
    selected_alias = _clean_credential_value(alias or "")
    if not profiles:
        return {}

    if selected_alias:
        for profile in profiles:
            if profile.get("alias") == selected_alias:
                return profile
        raise LLMClientError(
            f"API alias {selected_alias!r} was not found in {path}. "
            f"Available aliases: {_format_available_aliases(profiles)}."
        )

    if len(profiles) == 1:
        return profiles[0]

    raise LLMClientError(
        f"Multiple API profiles were found in {path}. "
        "Pass --api-alias or configure llm.api_alias. "
        f"Available aliases: {_format_available_aliases(profiles)}."
    )


def _validate_single_credentials_alias(credentials: dict[str, str], alias: str | None, path: Path) -> None:
    selected_alias = _clean_credential_value(alias or "")
    if not selected_alias or not credentials:
        return
    actual_alias = credentials.get("alias")
    if actual_alias == selected_alias:
        return
    available = _format_available_aliases([credentials]) if actual_alias else "none"
    raise LLMClientError(
        f"API alias {selected_alias!r} was requested, but {path} does not define it. "
        f"Available aliases: {available}."
    )


def _has_credential_field(mapping: dict[str, Any]) -> bool:
    for key in mapping:
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in _API_KEY_FIELDS | _BASE_URL_FIELDS | _MODEL_FIELDS:
            return True
    return False


def _has_profile_content(mapping: dict[str, str]) -> bool:
    return bool(mapping.get("alias")) and any(key in mapping for key in ("api_key", "base_url", "model_name"))


def _is_default_alias_only(mapping: dict[str, str]) -> bool:
    return bool(mapping.get("default_alias")) and not any(
        key in mapping for key in ("alias", "api_key", "base_url", "model_name")
    )


def _extract_default_alias(mapping: dict[str, Any]) -> str | None:
    for key, value in mapping.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in _DEFAULT_ALIAS_FIELDS and value is not None:
            return _clean_credential_value(str(value))
    return None


def _strip_internal_credential_fields(credentials: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in credentials.items() if key != "default_alias"}


def _format_available_aliases(profiles: list[dict[str, str]]) -> str:
    aliases = [profile.get("alias", "") for profile in profiles if profile.get("alias")]
    return ", ".join(sorted(aliases)) if aliases else "none"


def _parse_key_value_credentials(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        separator = "=" if "=" in line else ":" if ":" in line else ""
        if not separator:
            continue
        key, value = line.split(separator, 1)
        normalized_key = key.strip().lower().replace("-", "_")
        cleaned_value = _clean_credential_value(value)
        if not cleaned_value:
            continue
        if normalized_key in _API_KEY_FIELDS:
            values["api_key"] = cleaned_value
        elif normalized_key in _BASE_URL_FIELDS:
            values["base_url"] = cleaned_value
        elif normalized_key in _MODEL_FIELDS:
            values["model_name"] = cleaned_value
        elif normalized_key in _ALIAS_FIELDS:
            values["alias"] = cleaned_value
        elif normalized_key in _DEFAULT_ALIAS_FIELDS:
            values["default_alias"] = cleaned_value
        elif normalized_key in _ENABLE_THINKING_FIELDS:
            values["enable_thinking"] = cleaned_value
    return values


def _parse_plain_credentials(raw: str) -> dict[str, str]:
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return {}
    if len(lines) == 1:
        parts = lines[0].split()
    else:
        parts = lines

    values: dict[str, str] = {"api_key": _clean_credential_value(parts[0])}
    for part in parts[1:]:
        value = _clean_credential_value(part)
        if value.startswith(("http://", "https://")):
            values["base_url"] = value
        elif value and "model_name" not in values:
            values["model_name"] = value
    return values


def _plain_credentials_only(raw: str) -> str:
    lines: list[str] = []
    known_fields = (
        _API_KEY_FIELDS
        | _BASE_URL_FIELDS
        | _MODEL_FIELDS
        | _ALIAS_FIELDS
        | _DEFAULT_ALIAS_FIELDS
        | _ENABLE_THINKING_FIELDS
    )
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        separator = "=" if "=" in stripped else ":" if ":" in stripped else ""
        if separator:
            key = stripped.split(separator, 1)[0].strip().lower().replace("-", "_")
            if key in known_fields:
                continue
        lines.append(stripped)
    return "\n".join(lines)


def _normalize_credentials_mapping(mapping: dict[str, Any]) -> dict[str, str]:
    if isinstance(mapping.get("llm"), dict):
        nested = _normalize_credentials_mapping(mapping["llm"])
        current = _normalize_credentials_mapping({k: v for k, v in mapping.items() if k != "llm"})
        return {**nested, **current}

    values: dict[str, str] = {}
    for key, value in mapping.items():
        if value is None:
            continue
        normalized_key = str(key).strip().lower().replace("-", "_")
        cleaned_value = _clean_credential_value(str(value))
        if normalized_key in _API_KEY_FIELDS:
            values["api_key"] = cleaned_value
        elif normalized_key in _BASE_URL_FIELDS:
            values["base_url"] = cleaned_value
        elif normalized_key in _MODEL_FIELDS:
            values["model_name"] = cleaned_value
        elif normalized_key in _ALIAS_FIELDS:
            values["alias"] = cleaned_value
        elif normalized_key in _DEFAULT_ALIAS_FIELDS:
            values["default_alias"] = cleaned_value
        elif normalized_key in _ENABLE_THINKING_FIELDS:
            values["enable_thinking"] = cleaned_value
    return values


def _clean_credential_value(value: str) -> str:
    return value.strip().strip(",").strip().strip('"').strip("'")


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None
