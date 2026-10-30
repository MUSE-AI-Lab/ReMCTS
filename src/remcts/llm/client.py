from __future__ import annotations

from abc import ABC, abstractmethod
from http.client import IncompleteRead, RemoteDisconnected
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .credentials import load_credentials_file
from .errors import LLMClientError

__all__ = [
    "BaseLLMClient",
    "LLMClientError",
    "MockLLMClient",
    "OpenAICompatibleLLMClient",
    "create_llm_client",
]

SYSTEM_PROMPT = (
    "You are a careful coding assistant. "
    "Return only complete executable code unless the user explicitly asks for JSON."
)
_USAGE_COUNTERS = ("prompt_tokens", "completion_tokens", "total_tokens")


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

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
    """Offline client for smoke tests: emits a placeholder program and a fixed semantic score."""

    def generate(self, prompt: str, **kwargs) -> str:
        if "score" in prompt.lower() and "json" in prompt.lower():
            return '{"score": 0.2, "reason": "mock semantic score"}'
        return """```python
def solution(*args, **kwargs):
    raise NotImplementedError(\"MockLLMClient placeholder\")
```"""


class OpenAICompatibleLLMClient(BaseLLMClient):
    """Minimal Chat Completions client (standard library only) with retries and usage records."""

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
        records = self._usage_records
        totals = {key: sum(_safe_int(record.get(key)) for record in records) for key in _USAGE_COUNTERS}
        if totals["total_tokens"] == 0:
            totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
        successful = sum(1 for record in records if record.get("status") == "success")
        return {
            "model": self.model_name,
            "api_alias": self.api_alias,
            "calls": len(records),
            "successful_calls": successful,
            "failed_calls": len(records) - successful,
            **totals,
            "elapsed_sec": round(sum(float(record.get("elapsed_sec", 0.0)) for record in records), 6),
            "records": list(records),
        }

    def generate(self, prompt: str, **kwargs) -> str:
        temperature = kwargs.get("temperature")
        max_tokens = kwargs.get("max_tokens")
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        enable_thinking = self.enable_thinking
        if enable_thinking is None and self.model_name.lower().startswith("qwen3"):
            enable_thinking = False
        payload.update(_thinking_payload(self.model_name, enable_thinking))
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        try:
            return self._chat_completion(payload)
        except LLMClientError as exc:
            # Some providers reject ``max_tokens`` or ``temperature``; retry once without them.
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
                    body = json.loads(response.read().decode("utf-8"))
                self._record_usage(payload, body, time.perf_counter() - started, attempt, "success")
                return _extract_chat_content(body)
            except HTTPError as exc:
                raw_error = exc.read().decode("utf-8", errors="replace")
                last_error = LLMClientError(f"LLM HTTP {exc.code}: {_sanitize_error(raw_error)}")
                self._record_usage(
                    payload, None, time.perf_counter() - started, attempt, f"http_{exc.code}", str(last_error)
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
                self._record_usage(
                    payload, None, time.perf_counter() - started, attempt, "error", f"{type(exc).__name__}: {exc}"
                )
            if attempt < self.max_retries:
                time.sleep(_retry_delay(attempt))
        raise LLMClientError(str(last_error or "Unknown LLM client error"))

    def _record_usage(
        self,
        payload: dict[str, Any],
        body: dict[str, Any] | None,
        elapsed_sec: float,
        attempt: int,
        status: str,
        error: str | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "call_index": len(self._usage_records) + 1,
            "model": str(payload.get("model") or self.model_name),
            "api_alias": self.api_alias,
            "status": status,
            "attempt": attempt,
            "elapsed_sec": round(elapsed_sec, 6),
            **_extract_usage(body or {}),
        }
        if error:
            record["error"] = error[:1000]
        self._usage_records.append(record)


def _thinking_payload(model_name: str, enable_thinking: bool | None) -> dict[str, Any]:
    if enable_thinking is None:
        return {}
    model = model_name.lower()
    if model.startswith("deepseek-v4"):
        return {"thinking": {"type": "enabled" if enable_thinking else "disabled"}}
    if model.startswith("gpt-5"):
        # GPT-5 reasoning models take reasoning_effort instead of a provider-specific flag.
        return {} if enable_thinking else {"reasoning_effort": "none"}
    return {"enable_thinking": enable_thinking}


def _extract_chat_content(body: dict[str, Any]) -> str:
    choice = body["choices"][0]
    content = choice["message"].get("content", "") if "message" in choice else choice.get("text", "")
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
    return status_code in {408, 409, 425, 429} or status_code >= 500


def _retry_delay(attempt: int) -> float:
    return min(12.0, 0.75 * (2**attempt))


def _extract_usage(body: dict[str, Any]) -> dict[str, int]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    prompt_tokens = _first_int(usage, "prompt_tokens", "input_tokens", "inputTokenCount", "input_token_count")
    completion_tokens = _first_int(
        usage, "completion_tokens", "output_tokens", "outputTokenCount", "output_token_count"
    )
    total_tokens = _first_int(usage, "total_tokens", "totalTokenCount", "total_token_count")
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens or prompt_tokens + completion_tokens,
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


def _optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def create_llm_client(config: Any) -> BaseLLMClient:
    """Build the client described by a ``SearchConfig`` (``llm.model_name: mock`` gives the mock client)."""
    if config.llm_model_name.lower() == "mock":
        return MockLLMClient()
    requested_alias = getattr(config, "llm_api_alias", None)
    credentials = load_credentials_file(getattr(config, "llm_credentials_file", None), alias=requested_alias)
    if getattr(config, "llm_enable_thinking", None) is not None:
        enable_thinking = _optional_bool(str(config.llm_enable_thinking))
    else:
        enable_thinking = _optional_bool(credentials.get("enable_thinking"))
    return OpenAICompatibleLLMClient(
        model_name=credentials.get("model_name") or config.llm_model_name,
        base_url=credentials.get("base_url") or config.llm_base_url,
        api_key=credentials.get("api_key"),
        api_key_env=config.llm_api_key_env,
        api_alias=credentials.get("alias") or requested_alias,
        enable_thinking=enable_thinking,
        timeout_sec=config.llm_timeout_sec,
        max_retries=config.llm_max_retries,
    )
