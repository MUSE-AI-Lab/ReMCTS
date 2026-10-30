"""Run one task of one experiment; executed in worker processes."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

from remcts.config import load_config
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient, OpenAICompatibleLLMClient, create_llm_client

from .keys import base_url_for_model, is_retryable_error, load_key_pool, select_profile
from .methods import METHODS, TaskContext
from .usage import compact_usage, merge_usages


@dataclass
class TaskPayload:
    label: str
    method: str
    runner: str
    model_alias: str
    dataset: str
    task_index: int
    key_slot: int
    config: str
    problem: dict[str, Any]
    hidden_problem: dict[str, Any] | None = None
    config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    run_output_dir: str = "outputs/runs"
    credentials_file: str | None = None
    key_rotation: bool = True
    max_key_switches: int = 4


def build_config(payload: TaskPayload) -> SearchConfig:
    config = load_config(payload.config, payload.config_overrides)
    config.llm_model_name = payload.model_alias
    config.llm_api_alias = payload.model_alias
    config.llm_base_url = base_url_for_model(payload.model_alias) or config.llm_base_url
    config.llm_credentials_file = payload.credentials_file
    config.llm_enable_thinking = False
    config.llm_max_retries = max(int(config.llm_max_retries), 5)
    config.run_output_dir = payload.run_output_dir
    return config


def build_client(payload: TaskPayload, config: SearchConfig, attempt: int) -> tuple[BaseLLMClient, dict[str, Any]]:
    """A client on the task's key from the key pool, or from ``llm.credentials_file`` / env as fallback."""
    profiles = load_key_pool(payload.credentials_file, payload.model_alias) if payload.key_rotation else []
    profile = select_profile(profiles, payload.key_slot, attempt)
    if profile is None:
        return create_llm_client(config), {"key_rotation_enabled": False, "key_switch_count": attempt}
    config.llm_credentials_file = None
    config.llm_api_alias = f"{payload.model_alias}:{profile.fingerprint}"
    client = OpenAICompatibleLLMClient(
        model_name=profile.model_name,
        base_url=profile.base_url,
        api_key=profile.api_key,
        api_alias=config.llm_api_alias,
        enable_thinking=config.llm_enable_thinking,
        timeout_sec=config.llm_timeout_sec,
        max_retries=config.llm_max_retries,
    )
    stats = {
        "key_rotation_enabled": True,
        "key_group": profile.provider,
        "key_index": profile.index,
        "key_fingerprint": profile.fingerprint,
        "key_switch_count": attempt,
    }
    return client, stats


def run_task(payload: TaskPayload) -> dict[str, Any]:
    """Run the method on one task; on a retryable API error, retry the task on the next key."""
    problem = ProblemSpec.from_dict(payload.problem)
    hidden = ProblemSpec.from_dict(payload.hidden_problem) if payload.hidden_problem else None
    started = time.perf_counter()
    usages: list[dict[str, Any]] = []
    retryable_errors = 0
    stats: dict[str, Any] = {"key_rotation_enabled": False, "key_switch_count": 0}
    attempts = max(1, int(payload.max_key_switches)) if payload.key_rotation else 1
    row: dict[str, Any] = {}
    for attempt in range(attempts):
        client = None
        try:
            config = build_config(payload)
            client, stats = build_client(payload, config, attempt)
            client.reset_usage()
            row = METHODS[payload.runner](TaskContext(client, config, problem, hidden, dict(payload.params)))
            usages.append(compact_usage(client.usage_snapshot()))
            break
        except Exception as exc:
            usages.append(compact_usage(client.usage_snapshot() if client is not None else None))
            if stats.get("key_rotation_enabled") and is_retryable_error(exc) and attempt < attempts - 1:
                retryable_errors += 1
                continue
            row = failed_row(payload.runner, exc)
            break
    return {
        "task_index": payload.task_index,
        "task_id": problem.task_id,
        "dataset": payload.dataset,
        "model_alias": payload.model_alias,
        "method": payload.method,
        **row,
        "wall_elapsed_sec": round(time.perf_counter() - started, 6),
        "llm_usage": merge_usages(usages),
        "key_rotation": {**stats, "retryable_errors": retryable_errors},
    }


def failed_row(runner: str, exc: BaseException, stop_reason: str = "request_failed") -> dict[str, Any]:
    return {
        "success": False,
        "accepted": False,
        "needs_review": True,
        "stop_reason": stop_reason,
        "iterations": 1 if runner == "direct" else 0,
        "tree_size": 0,
        "run_dir": None,
        "code": "",
        "request_failed": True,
        "error": f"{type(exc).__name__}: {exc}"[:1000],
    }
