#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import OpenAICompatibleLLMClient, create_llm_client
from remcts.llm.response_parser import extract_code
from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from remcts.tasks.humaneval import HumanEvalDataset
from remcts.tasks.mbpp import MBPPDataset

from api_key_pool import ApiProfile, load_key_pool, sanitize_rotation_stats, select_profile

MODEL_PRICES_CNY_PER_MILLION: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.952, 1.904),
    "qwen3-1.7b": (0.3, 1.2),
    "qwen3-8b": (0.5, 2.0),
    "qwen3-14b": (1.0, 4.0),
    "qwen3-32b": (2.0, 8.0),
}

REFERENCE_REMCTS_USAGE: dict[str, tuple[int, int]] = {
    "humaneval_full_known": (1_633_843, 372_884),
    "humaneval_full_no_tests": (821_943, 210_036),
    "mbpp_sanitized_full_known": (5_798_278, 1_286_010),
    "mbpp_sanitized_full_no_tests": (1_493_723, 415_300),
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: str
    data_path: Path
    remcts_config_path: Path
    hidden_eval_path: Path | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    label: str
    model_alias: str
    method: str
    dataset_spec: DatasetSpec
    reuse_from_label: str | None = None


class BudgetExceeded(RuntimeError):
    pass


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug_model(model_alias: str) -> str:
    return model_alias.replace(".", "_").replace("-", "_")


def load_dataset(dataset: str, path: Path) -> list[ProblemSpec]:
    if dataset == "humaneval":
        return list(HumanEvalDataset(str(path)).iter_tasks())
    if dataset == "mbpp":
        return list(MBPPDataset(str(path)).iter_tasks())
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_config(path: Path, api_alias: str) -> SearchConfig:
    config = SearchConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    config.llm_api_alias = api_alias
    config.llm_enable_thinking = False
    config.llm_max_retries = max(int(config.llm_max_retries), 5)
    return config


def compact_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    keys = [
        "model",
        "api_alias",
        "calls",
        "successful_calls",
        "failed_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "elapsed_sec",
    ]
    compact = {key: usage.get(key, 0) for key in keys}
    compact["elapsed_sec"] = round(float(compact.get("elapsed_sec") or 0.0), 6)
    for key in ["calls", "successful_calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens"]:
        compact[key] = int(compact.get(key) or 0)
    return compact


def aggregate_usage(task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_sec": 0.0,
    }
    by_model: dict[str, dict[str, Any]] = {}
    for task in task_rows:
        usage = task.get("llm_usage") or {}
        model = str(usage.get("model") or task.get("model_alias") or "unknown")
        model_totals = by_model.setdefault(model, dict(totals))
        for key in totals:
            value = usage.get(key, 0)
            if key == "elapsed_sec":
                totals[key] += float(value or 0.0)
                model_totals[key] += float(value or 0.0)
            else:
                totals[key] += int(value or 0)
                model_totals[key] += int(value or 0)
    totals["elapsed_sec"] = round(float(totals["elapsed_sec"]), 6)
    for model_totals in by_model.values():
        model_totals["elapsed_sec"] = round(float(model_totals["elapsed_sec"]), 6)
    return {"totals": totals, "by_model": by_model}


def estimate_cost_cny_for_model(model_alias: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = MODEL_PRICES_CNY_PER_MILLION.get(model_alias, MODEL_PRICES_CNY_PER_MILLION["qwen3-32b"])
    return round(
        prompt_tokens * input_price / 1_000_000
        + completion_tokens * output_price / 1_000_000,
        6,
    )


def hidden_summary(task_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    hidden_rows = [row for row in task_rows if "hidden_success" in row]
    if not hidden_rows:
        return None
    total = len(hidden_rows)
    accepted = sum(1 for row in hidden_rows if row.get("accepted"))
    needs_review = sum(1 for row in hidden_rows if row.get("needs_review"))
    hidden_success = sum(1 for row in hidden_rows if row.get("hidden_success"))
    accepted_and_success = sum(1 for row in hidden_rows if row.get("accepted") and row.get("hidden_success"))
    accepted_but_failed = sum(1 for row in hidden_rows if row.get("accepted") and not row.get("hidden_success"))
    return {
        "tasks": total,
        "accepted": accepted,
        "needs_review": needs_review,
        "hidden_success": hidden_success,
        "accepted_and_hidden_success": accepted_and_success,
        "accepted_but_hidden_failed": accepted_but_failed,
        "accepted_rate": round(accepted / total, 4) if total else 0.0,
        "needs_review_rate": round(needs_review / total, 4) if total else 0.0,
        "hidden_solve_rate": round(hidden_success / total, 4) if total else 0.0,
        "acceptance_precision": round(accepted_and_success / accepted, 4) if accepted else 0.0,
        "false_acceptance_rate": round(accepted_but_failed / accepted, 4) if accepted else 0.0,
    }


def build_direct_prompt(problem: ProblemSpec) -> str:
    entry_point = problem.entry_point or problem.metadata.get("entry_point") or "solution"
    return f"""
You are a careful Python coding assistant. Solve the following programming problem in one shot.

Return only complete executable Python code. Do not include Markdown fences or explanations.
The solution must define the entry function `{entry_point}` and should not rely on stdin, command-line input, or interactive prompts unless the problem explicitly requires it.

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def _problem_from_dict(data: dict[str, Any]) -> ProblemSpec:
    return ProblemSpec(
        task_id=str(data.get("task_id", "unknown")),
        prompt=str(data.get("prompt") or ""),
        entry_point=data.get("entry_point"),
        tests=str(data.get("tests") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


_KEY_POOL_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, list[ApiProfile]]] = {}


def _key_rotation_mode() -> str:
    return os.getenv("REMCTS_KEY_ROTATION", "disabled").strip().lower()


def _key_pool_file() -> str | None:
    value = os.getenv("REMCTS_KEY_POOL_FILE")
    return value.strip() if value else None


def _key_rotation_is_enabled() -> bool:
    return _key_rotation_mode() not in {"", "0", "false", "off", "disabled", "none"} and bool(_key_pool_file())


def _max_key_switches() -> int:
    try:
        return max(1, int(os.getenv("REMCTS_MAX_KEY_SWITCHES", "4")))
    except ValueError:
        return 4


def _load_runtime_key_pool(model_alias: str) -> dict[str, list[ApiProfile]]:
    path = _key_pool_file()
    if not path:
        return {}
    key = (path, (model_alias,))
    if key not in _KEY_POOL_CACHE:
        _KEY_POOL_CACHE[key] = load_key_pool(path, model_aliases=[model_alias])
    return _KEY_POOL_CACHE[key]


def _select_runtime_profile(model_alias: str, task_index: int, attempt: int) -> ApiProfile | None:
    if not _key_rotation_is_enabled():
        return None
    return select_profile(_load_runtime_key_pool(model_alias), model_alias, task_index, attempt)


def stable_key_slot(label: str, task_index: int) -> int:
    digest = sha256(f"{label}:{int(task_index)}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _config_and_client_for_task(
    config_path: Path,
    model_alias: str,
    task_index: int,
    key_attempt: int,
) -> tuple[SearchConfig, Any, dict[str, Any]]:
    config = load_config(config_path, model_alias)
    profile = _select_runtime_profile(model_alias, task_index, key_attempt)
    if profile is None:
        return config, create_llm_client(config), {
            "key_rotation_enabled": False,
            "key_switch_count": key_attempt,
        }

    config.llm_credentials_file = None
    config.llm_model_name = profile.model_name
    config.llm_base_url = profile.base_url
    config.llm_api_alias = f"{model_alias}:{profile.fingerprint}"
    client = OpenAICompatibleLLMClient(
        model_name=config.llm_model_name,
        base_url=config.llm_base_url,
        api_key=profile.api_key,
        api_alias=config.llm_api_alias,
        enable_thinking=config.llm_enable_thinking,
        timeout_sec=config.llm_timeout_sec,
        max_retries=config.llm_max_retries,
    )
    return config, client, {
        "key_rotation_enabled": True,
        "key_group": profile.provider,
        "key_index": profile.index,
        "key_fingerprint": profile.fingerprint,
        "key_switch_count": key_attempt,
    }


def _is_retryable_key_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    retry_markers = [
        "429",
        "408",
        "500",
        "502",
        "503",
        "504",
        "rate limit",
        "timeout",
        "timed out",
        "temporarily",
        "connection",
        "remote end closed",
        "incomplete",
        "reset",
    ]
    return any(marker in text for marker in retry_markers)


def _merge_compact_usages(usages: list[dict[str, Any]]) -> dict[str, Any]:
    merged = compact_usage(None)
    model = None
    api_alias = None
    for usage in usages:
        usage = compact_usage(usage)
        model = usage.get("model") or model
        api_alias = usage.get("api_alias") or api_alias
        for key in ["calls", "successful_calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens"]:
            merged[key] = int(merged.get(key) or 0) + int(usage.get(key) or 0)
        merged["elapsed_sec"] = round(float(merged.get("elapsed_sec") or 0.0) + float(usage.get("elapsed_sec") or 0.0), 6)
    merged["model"] = model or merged.get("model", 0)
    merged["api_alias"] = api_alias or merged.get("api_alias", 0)
    return merged


def _attach_rotation_stats(row: dict[str, Any], stats: dict[str, Any], retryable_errors: int) -> None:
    public_stats = sanitize_rotation_stats(stats)
    public_stats["retryable_errors"] = int(retryable_errors)
    row["key_rotation"] = public_stats


def _apply_hidden_eval(
    task_row: dict[str, Any],
    hidden_problem: ProblemSpec | None,
    code: str,
    config: SearchConfig,
) -> None:
    if hidden_problem is None or not code:
        return
    hidden_executor = PythonExecutor()
    hidden_result = run_problem_tests(
        hidden_problem,
        code,
        hidden_executor,
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )
    hidden_ok = (
        hidden_result.total_tests > 0
        and hidden_result.exec_ok
        and hidden_result.passed_tests == hidden_result.total_tests
    )
    task_row.update(
        {
            "hidden_success": bool(hidden_ok),
            "hidden_passed_tests": hidden_result.passed_tests,
            "hidden_total_tests": hidden_result.total_tests,
            "hidden_failed_cases": hidden_result.failed_cases[:8],
        }
    )


def _apply_execution_result_as_hidden_eval(task_row: dict[str, Any], execution_result: Any) -> None:
    hidden_ok = (
        execution_result.total_tests > 0
        and execution_result.exec_ok
        and execution_result.passed_tests == execution_result.total_tests
    )
    task_row.update(
        {
            "hidden_success": bool(hidden_ok),
            "hidden_passed_tests": execution_result.passed_tests,
            "hidden_total_tests": execution_result.total_tests,
            "hidden_failed_cases": execution_result.failed_cases[:8],
        }
    )


def _run_remcts_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    problem = _problem_from_dict(payload["problem"])
    method = str(payload.get("method") or "remcts")
    hidden_problem = (
        _problem_from_dict(payload["hidden_problem"])
        if payload.get("hidden_problem") is not None
        else None
    )
    task_started_all = time.perf_counter()
    attempted_usages: list[dict[str, Any]] = []
    retryable_errors = 0
    max_attempts = _max_key_switches() if _key_rotation_is_enabled() else 1
    last_error: Exception | None = None
    last_stats: dict[str, Any] = {"key_rotation_enabled": False, "key_switch_count": 0}
    for key_attempt in range(max_attempts):
        engine: SearchEngine | None = None
        try:
            config, client, rotation_stats = _config_and_client_for_task(
                Path(payload["config_path"]),
                str(payload["model_alias"]),
                int(payload.get("key_slot", payload["task_index"])),
                key_attempt,
            )
            last_stats = rotation_stats
            engine = SearchEngine(config, llm_client=client)
            if hasattr(engine.llm_client, "reset_usage"):
                engine.llm_client.reset_usage()
            result = engine.run(problem)
            attempted_usages.append(compact_usage(result.summary.get("llm_usage")))
            usage = _merge_compact_usages(attempted_usages)
            task_row: dict[str, Any] = {
                "task_index": int(payload["task_index"]),
                "task_id": problem.task_id,
                "model_alias": str(payload["model_alias"]),
                "method": method,
                "success": bool(result.success),
                "accepted": bool(result.accepted),
                "needs_review": bool(result.needs_review),
                "stop_reason": result.stop_reason,
                "iterations": result.iterations,
                "tree_size": result.tree_size,
                "run_dir": result.run_dir,
                "wall_elapsed_sec": round(time.perf_counter() - task_started_all, 6),
                "llm_usage": usage,
            }
            _attach_rotation_stats(task_row, rotation_stats, retryable_errors)
            _apply_hidden_eval(task_row, hidden_problem, result.best_code, config)
            return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))
        except Exception as exc:
            last_error = exc
            usage = compact_usage(engine.llm_client.usage_snapshot() if engine is not None else None)
            attempted_usages.append(usage)
            if _key_rotation_is_enabled() and _is_retryable_key_error(exc) and key_attempt < max_attempts - 1:
                retryable_errors += 1
                continue
            task_row = _failed_task_row(
                payload,
                problem,
                method,
                round(time.perf_counter() - task_started_all, 6),
                _merge_compact_usages(attempted_usages),
                exc,
            )
            _attach_rotation_stats(task_row, last_stats, retryable_errors)
            return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))
    task_row = _failed_task_row(
        payload,
        problem,
        method,
        round(time.perf_counter() - task_started_all, 6),
        _merge_compact_usages(attempted_usages),
        last_error or RuntimeError("key rotation exhausted"),
    )
    _attach_rotation_stats(task_row, last_stats, retryable_errors)
    return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))


def _run_direct_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    problem = _problem_from_dict(payload["problem"])
    hidden_problem = (
        _problem_from_dict(payload["hidden_problem"])
        if payload.get("hidden_problem") is not None
        else None
    )
    task_started_all = time.perf_counter()
    attempted_usages: list[dict[str, Any]] = []
    retryable_errors = 0
    max_attempts = _max_key_switches() if _key_rotation_is_enabled() else 1
    last_error: Exception | None = None
    last_stats: dict[str, Any] = {"key_rotation_enabled": False, "key_switch_count": 0}
    for key_attempt in range(max_attempts):
        client = None
        try:
            config, client, rotation_stats = _config_and_client_for_task(
                Path(payload["config_path"]),
                str(payload["model_alias"]),
                int(payload.get("key_slot", payload["task_index"])),
                key_attempt,
            )
            last_stats = rotation_stats
            client.reset_usage()
            raw = client.generate(
                build_direct_prompt(problem),
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            code = extract_code(raw).strip()
            executor = PythonExecutor()
            eval_problem = hidden_problem or problem
            exec_result = run_problem_tests(
                eval_problem,
                code,
                executor,
                timeout_sec=config.timeout_sec,
                max_output_chars=config.max_output_chars,
            )
            attempted_usages.append(compact_usage(client.usage_snapshot()))
            success = exec_result.total_tests > 0 and exec_result.exec_ok and exec_result.passed_tests == exec_result.total_tests
            task_row: dict[str, Any] = {
                "task_index": int(payload["task_index"]),
                "task_id": problem.task_id,
                "model_alias": str(payload["model_alias"]),
                "method": "direct",
                "success": bool(success),
                "accepted": bool(success),
                "needs_review": False,
                "stop_reason": "direct_hidden_tests_passed" if hidden_problem is not None and success else (
                    "direct_hidden_single_call_evaluated" if hidden_problem is not None else (
                        "direct_tests_passed" if success else "direct_single_call_evaluated"
                    )
                ),
                "iterations": 1,
                "tree_size": 1,
                "run_dir": None,
                "wall_elapsed_sec": round(time.perf_counter() - task_started_all, 6),
                "llm_usage": _merge_compact_usages(attempted_usages),
                "execution": exec_result.to_dict(),
                "code": code,
            }
            _attach_rotation_stats(task_row, rotation_stats, retryable_errors)
            if hidden_problem is not None:
                _apply_execution_result_as_hidden_eval(task_row, exec_result)
            return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))
        except Exception as exc:
            last_error = exc
            attempted_usages.append(compact_usage(client.usage_snapshot() if client is not None else None))
            if _key_rotation_is_enabled() and _is_retryable_key_error(exc) and key_attempt < max_attempts - 1:
                retryable_errors += 1
                continue
            task_row = _failed_task_row(
                payload,
                problem,
                "direct",
                round(time.perf_counter() - task_started_all, 6),
                _merge_compact_usages(attempted_usages),
                exc,
            )
            _attach_rotation_stats(task_row, last_stats, retryable_errors)
            return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))
    task_row = _failed_task_row(
        payload,
        problem,
        "direct",
        round(time.perf_counter() - task_started_all, 6),
        _merge_compact_usages(attempted_usages),
        last_error or RuntimeError("key rotation exhausted"),
    )
    _attach_rotation_stats(task_row, last_stats, retryable_errors)
    return _with_progress_line(task_row, str(payload["label"]), int(payload["task_count"]))


def _failed_task_row(
    payload: dict[str, Any],
    problem: ProblemSpec,
    method: str,
    task_elapsed: float,
    usage: dict[str, Any],
    exc: Exception,
) -> dict[str, Any]:
    return {
        "task_index": int(payload["task_index"]),
        "task_id": problem.task_id,
        "model_alias": str(payload["model_alias"]),
        "method": method,
        "success": False,
        "accepted": False,
        "needs_review": True,
        "stop_reason": "request_failed",
        "iterations": 1 if method == "direct" else 0,
        "tree_size": 0,
        "run_dir": None,
        "wall_elapsed_sec": task_elapsed,
        "llm_usage": usage,
        "request_failed": True,
        "error": f"{type(exc).__name__}: {exc}"[:1000],
    }


def _with_progress_line(task_row: dict[str, Any], label: str, task_count: int) -> dict[str, Any]:
    usage = task_row.get("llm_usage") or {}
    hidden_text = ""
    if "hidden_success" in task_row:
        hidden_text = (
            f", hidden={task_row['hidden_success']} "
            f"({task_row['hidden_passed_tests']}/{task_row['hidden_total_tests']})"
        )
    error_text = ", request_failed=True" if task_row.get("request_failed") else ""
    task_row["progress_line"] = (
        f"[task] {label} {task_row['task_index']}/{task_count} {task_row['task_id']}: "
        f"success={task_row['success']}, accepted={task_row['accepted']}, iter={task_row['iterations']}, "
        f"calls={usage.get('calls', 0)}, tokens={usage.get('total_tokens', 0)}{hidden_text}{error_text}, "
        f"wall={task_row['wall_elapsed_sec']:.2f}s"
    )
    return task_row


def _run_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    if payload["method"] == "direct":
        return _run_direct_task_worker(payload)
    return _run_remcts_task_worker(payload)


def _load_progress(progress_path: Path, tasks: list[ProblemSpec]) -> list[dict[str, Any]]:
    if not progress_path.exists():
        return []
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    rows = data.get("tasks") or []
    if not isinstance(rows, list):
        return []
    order = {problem.task_id: index for index, problem in enumerate(tasks, start=1)}
    dedup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("request_failed"):
            continue
        task_id = str(row.get("task_id") or "")
        if task_id not in order:
            continue
        row["task_index"] = int(row.get("task_index") or order[task_id])
        dedup[task_id] = row
    return sorted(dedup.values(), key=lambda item: int(item.get("task_index") or 0))


def _write_progress(progress_path: Path, spec: ExperimentSpec, task_count: int, task_rows: list[dict[str, Any]]) -> None:
    ordered_rows = sorted(task_rows, key=lambda item: int(item.get("task_index") or 0))
    progress = {
        "label": spec.label,
        "model_alias": spec.model_alias,
        "method": spec.method,
        "dataset": spec.dataset_spec.name,
        "completed": len(ordered_rows),
        "size": task_count,
        "tasks": ordered_rows,
    }
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def _finalize_spec_output(
    spec: ExperimentSpec,
    output_path: Path,
    tasks: list[ProblemSpec],
    task_rows: list[dict[str, Any]],
    started: float,
    resumed_count: int,
    workers: int,
) -> dict[str, Any]:
    task_rows = sorted(task_rows, key=lambda item: int(item.get("task_index") or 0))
    wall_elapsed = round(time.perf_counter() - started, 6)
    summed_task_wall = round(sum(float(row.get("wall_elapsed_sec") or 0.0) for row in task_rows), 6)
    usage_summary = aggregate_usage(task_rows)
    totals = usage_summary["totals"]
    hidden = hidden_summary(task_rows)
    raw_search_success_rate = (
        round(sum(1 for row in task_rows if row.get("success")) / len(task_rows), 6)
        if task_rows
        else 0.0
    )
    request_failed_count = sum(1 for row in task_rows if row.get("request_failed"))
    output = {
        "label": spec.label,
        "model_alias": spec.model_alias,
        "method": spec.method,
        "dataset": spec.dataset_spec.name,
        "dataset_type": spec.dataset_spec.dataset,
        "data_path": str(spec.dataset_spec.data_path),
        "config_path": str(spec.dataset_spec.remcts_config_path),
        "hidden_eval_path": str(spec.dataset_spec.hidden_eval_path) if spec.dataset_spec.hidden_eval_path else None,
        "reuse_from_label": spec.reuse_from_label,
        "status": "complete" if len(task_rows) == len(tasks) and request_failed_count == 0 else "partial",
        "size": len(tasks),
        "completed": len(task_rows),
        "pass_rate": hidden["hidden_solve_rate"] if hidden else raw_search_success_rate,
        "pass_rate_metric": "hidden_solve_rate" if hidden else "explicit_test_pass_rate",
        "raw_search_success_rate": raw_search_success_rate,
        "avg_iterations": round(sum(int(row.get("iterations") or 0) for row in task_rows) / len(task_rows), 6) if task_rows else 0.0,
        "avg_tree_size": round(sum(int(row.get("tree_size") or 0) for row in task_rows) / len(task_rows), 6) if task_rows else 0.0,
        "request_failed_tasks": request_failed_count,
        "wall_elapsed_sec": wall_elapsed,
        "summed_task_wall_elapsed_sec": summed_task_wall,
        "resumed_task_count": resumed_count,
        "workers": workers,
        "llm_usage": usage_summary,
        "estimated_cost_cny": estimate_cost_cny_for_model(
            spec.model_alias,
            totals["prompt_tokens"],
            totals["completion_tokens"],
        ),
        "hidden_summary": hidden,
        "tasks": task_rows,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_reused_direct_spec(
    spec: ExperimentSpec,
    report_dir: Path,
    force: bool,
) -> dict[str, Any]:
    output_path = report_dir / f"{spec.label}.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[skip] {spec.label}: {output_path}", flush=True)
            return existing

    if not spec.reuse_from_label:
        raise ValueError(f"{spec.label} does not declare reuse_from_label")

    source_path = report_dir / f"{spec.reuse_from_label}.json"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Cannot derive {spec.label}: source direct report is missing: {source_path}"
        )

    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise RuntimeError(f"Cannot derive {spec.label}: source report is not complete: {source_path}")

    tasks = load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.data_path)
    hidden_tasks = (
        {problem.task_id: problem for problem in load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.hidden_eval_path)}
        if spec.dataset_spec.hidden_eval_path
        else {}
    )
    source_rows = {str(row.get("task_id")): row for row in source.get("tasks") or []}
    config = load_config(spec.dataset_spec.remcts_config_path, spec.model_alias)
    started = time.perf_counter()
    task_rows: list[dict[str, Any]] = []
    print(
        f"[reuse] {spec.label}: source={spec.reuse_from_label}, size={len(tasks)}, "
        "api_calls=0, hidden_eval=True",
        flush=True,
    )

    for index, problem in enumerate(tasks, start=1):
        source_row = source_rows.get(problem.task_id)
        if source_row is None:
            raise RuntimeError(f"Cannot derive {spec.label}: missing source task {problem.task_id}")
        code = str(source_row.get("code") or "")
        if not code:
            raise RuntimeError(
                f"Cannot derive {spec.label}: source task {problem.task_id} has no saved code. "
                "Re-run the known direct baseline with the updated script."
            )

        task_started = time.perf_counter()
        task_row: dict[str, Any] = {
            "task_index": index,
            "task_id": problem.task_id,
            "model_alias": spec.model_alias,
            "method": "direct",
            "success": bool(source_row.get("success")),
            "accepted": bool(source_row.get("accepted")),
            "needs_review": bool(source_row.get("needs_review")),
            "stop_reason": "direct_reused_for_hidden_eval",
            "iterations": 1,
            "tree_size": 1,
            "run_dir": None,
            "wall_elapsed_sec": 0.0,
            "llm_usage": compact_usage(None),
            "reused_llm_from_label": spec.reuse_from_label,
            "source_stop_reason": source_row.get("stop_reason"),
            "code": code,
        }
        if "hidden_success" in source_row:
            task_row.update(
                {
                    "hidden_success": bool(source_row.get("hidden_success")),
                    "hidden_passed_tests": int(source_row.get("hidden_passed_tests") or 0),
                    "hidden_total_tests": int(source_row.get("hidden_total_tests") or 0),
                    "hidden_failed_cases": list(source_row.get("hidden_failed_cases") or [])[:8],
                }
            )
            task_row["success"] = bool(task_row["hidden_success"])
            task_row["accepted"] = bool(task_row["hidden_success"])
            task_row["needs_review"] = False
        else:
            _apply_hidden_eval(task_row, hidden_tasks.get(problem.task_id), code, config)
            if "hidden_success" in task_row:
                task_row["success"] = bool(task_row["hidden_success"])
                task_row["accepted"] = bool(task_row["hidden_success"])
                task_row["needs_review"] = False
        task_row["wall_elapsed_sec"] = round(time.perf_counter() - task_started, 6)
        task_rows.append(task_row)

    progress_path = report_dir / f"{spec.label}.progress.json"
    _write_progress(progress_path, spec, len(tasks), task_rows)
    output = _finalize_spec_output(
        spec,
        output_path,
        tasks,
        task_rows,
        started,
        resumed_count=0,
        workers=1,
    )
    print(f"[done] {spec.label}: {output_path}", flush=True)
    return output


def run_one_spec(
    spec: ExperimentSpec,
    report_dir: Path,
    force: bool,
    workers: int,
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> dict[str, Any]:
    if spec.reuse_from_label:
        return run_reused_direct_spec(spec, report_dir, force)

    output_path = report_dir / f"{spec.label}.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[skip] {spec.label}: {output_path}", flush=True)
            return existing

    tasks = load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.data_path)
    hidden_tasks = (
        {problem.task_id: problem for problem in load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.hidden_eval_path)}
        if spec.dataset_spec.hidden_eval_path
        else {}
    )
    progress_path = report_dir / f"{spec.label}.progress.json"
    task_rows = [] if force else _load_progress(progress_path, tasks)
    completed_task_ids = {str(row.get("task_id")) for row in task_rows}
    pending_tasks = [
        (index, problem)
        for index, problem in enumerate(tasks, start=1)
        if problem.task_id not in completed_task_ids
    ]

    existing_cost = completed_report_cost_cny(report_dir)
    next_estimate = estimate_spec_cost_cny(spec, tasks)
    projected = initial_spent_cny + existing_cost + next_estimate
    if budget_cny > 0 and projected > budget_cny - stop_margin_cny and pending_tasks:
        raise BudgetExceeded(
            f"Budget guard stopped before {spec.label}: initial={initial_spent_cny:.2f}, "
            f"completed_report={existing_cost:.2f}, next_estimate={next_estimate:.2f}, "
            f"budget={budget_cny:.2f}, margin={stop_margin_cny:.2f}."
        )

    print(
        f"[start] {spec.label}: model={spec.model_alias}, method={spec.method}, size={len(tasks)}, "
        f"pending={len(pending_tasks)}, resumed={len(task_rows)}, workers={workers}, "
        f"hidden_eval={bool(hidden_tasks)}, est_next_cny={next_estimate:.2f}, "
        f"projected_with_initial_cny={projected:.2f}",
        flush=True,
    )
    started = time.perf_counter()

    def make_payload(index: int, problem: ProblemSpec) -> dict[str, Any]:
        hidden_problem = hidden_tasks.get(problem.task_id)
        return {
            "label": spec.label,
            "model_alias": spec.model_alias,
            "method": spec.method,
            "task_index": index,
            "key_slot": stable_key_slot(spec.label, index),
            "task_count": len(tasks),
            "config_path": str(spec.dataset_spec.remcts_config_path),
            "problem": problem.to_dict(),
            "hidden_problem": hidden_problem.to_dict() if hidden_problem is not None else None,
        }

    budget_hit = False
    try:
        if workers <= 1:
            for index, problem in pending_tasks:
                task_row = _run_task_worker(make_payload(index, problem))
                task_rows.append(task_row)
                _write_progress(progress_path, spec, len(tasks), task_rows)
                print(task_row.pop("progress_line"), flush=True)
                if budget_exceeded(report_dir, spec, task_rows, budget_cny, initial_spent_cny, stop_margin_cny):
                    budget_hit = True
                    break
        elif pending_tasks:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = [pool.submit(_run_task_worker, make_payload(index, problem)) for index, problem in pending_tasks]
            try:
                for future in as_completed(futures):
                    task_row = future.result()
                    task_rows.append(task_row)
                    _write_progress(progress_path, spec, len(tasks), task_rows)
                    print(task_row.pop("progress_line"), flush=True)
                    if budget_exceeded(report_dir, spec, task_rows, budget_cny, initial_spent_cny, stop_margin_cny):
                        budget_hit = True
                        for pending_future in futures:
                            if not pending_future.done():
                                pending_future.cancel()
                        break
            finally:
                pool.shutdown(wait=True, cancel_futures=budget_hit)
        else:
            _write_progress(progress_path, spec, len(tasks), task_rows)
    finally:
        for row in task_rows:
            row.pop("progress_line", None)

    if budget_hit:
        _write_progress(progress_path, spec, len(tasks), task_rows)
        raise BudgetExceeded(f"Budget guard stopped during {spec.label}. Progress saved to {progress_path}.")

    output = _finalize_spec_output(
        spec,
        output_path,
        tasks,
        task_rows,
        started,
        resumed_count=len(completed_task_ids),
        workers=workers,
    )
    print(f"[done] {spec.label}: {output_path}", flush=True)
    return output


def budget_exceeded(
    report_dir: Path,
    current_spec: ExperimentSpec,
    current_rows: list[dict[str, Any]],
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> bool:
    if budget_cny <= 0:
        return False
    current_usage = aggregate_usage(current_rows)["totals"]
    current_cost = estimate_cost_cny_for_model(
        current_spec.model_alias,
        current_usage["prompt_tokens"],
        current_usage["completion_tokens"],
    )
    spent = initial_spent_cny + completed_report_cost_cny(report_dir) + current_cost
    return spent >= budget_cny - stop_margin_cny


def completed_report_cost_cny(report_dir: Path) -> float:
    total = 0.0
    for path in report_dir.glob("*.json"):
        if path.name == "summary.json" or path.name.endswith(".progress.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("status") != "complete":
            continue
        total += float(row.get("estimated_cost_cny") or 0.0)
    return round(total, 6)


def estimate_spec_cost_cny(spec: ExperimentSpec, tasks: list[ProblemSpec] | None = None) -> float:
    if spec.reuse_from_label:
        return 0.0
    if spec.method in {"remcts", "pure_mcts", "mcts_style"}:
        prompt_tokens, completion_tokens = REFERENCE_REMCTS_USAGE.get(spec.dataset_spec.name, (0, 0))
    else:
        loaded_tasks = tasks or load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.data_path)
        prompt_tokens = sum(max(1, len(build_direct_prompt(problem)) // 3) for problem in loaded_tasks)
        completion_tokens = len(loaded_tasks) * 700
    return estimate_cost_cny_for_model(spec.model_alias, prompt_tokens, completion_tokens)


def build_dataset_specs() -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name="humaneval_full_known",
            dataset="humaneval",
            data_path=ROOT / "data/humaneval/HumanEval.jsonl",
            remcts_config_path=ROOT / "src/remcts/config/qwen_full_k3.yaml",
            hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
        ),
        DatasetSpec(
            name="humaneval_full_no_tests",
            dataset="humaneval",
            data_path=ROOT / "data/humaneval/HumanEval_no_tests.jsonl",
            remcts_config_path=ROOT / "src/remcts/config/qwen_production_full_k3.yaml",
            hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
        ),
        DatasetSpec(
            name="mbpp_sanitized_full_known",
            dataset="mbpp",
            data_path=ROOT / "data/mbpp/mbpp_sanitized.json",
            remcts_config_path=ROOT / "src/remcts/config/qwen_full_k3.yaml",
            hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
        ),
        DatasetSpec(
            name="mbpp_sanitized_full_no_tests",
            dataset="mbpp",
            data_path=ROOT / "data/mbpp/mbpp_sanitized_no_tests.json",
            remcts_config_path=ROOT / "src/remcts/config/qwen_production_full_k3.yaml",
            hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
        ),
    ]


def direct_known_dataset_name(dataset_name: str) -> str:
    if dataset_name.endswith("_full_no_tests"):
        return dataset_name.removesuffix("_full_no_tests") + "_full_known"
    raise ValueError(f"No known direct counterpart for dataset: {dataset_name}")


def build_experiment_specs(
    models: list[str],
    include_remcts: bool,
    include_direct: bool,
    include_direct_no_tests: bool = False,
    dataset_names: set[str] | None = None,
    label_suffix: str = "",
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    dataset_specs = build_dataset_specs()
    if dataset_names:
        unknown = dataset_names - {dataset_spec.name for dataset_spec in dataset_specs}
        if unknown:
            raise ValueError(f"Unknown dataset names: {sorted(unknown)}")
        dataset_specs = [dataset_spec for dataset_spec in dataset_specs if dataset_spec.name in dataset_names]
    dataset_specs_by_name = {dataset_spec.name: dataset_spec for dataset_spec in dataset_specs}
    for model in models:
        model_slug = slug_model(model)
        label_prefix = f"{model_slug}_{slug_model(label_suffix)}" if label_suffix else model_slug
        if include_direct:
            for dataset_spec in dataset_specs:
                if dataset_spec.name.endswith("_full_no_tests") and not include_direct_no_tests:
                    source_dataset_name = direct_known_dataset_name(dataset_spec.name)
                    all_dataset_specs_by_name = {dataset_spec.name: dataset_spec for dataset_spec in build_dataset_specs()}
                    if source_dataset_name not in all_dataset_specs_by_name:
                        raise ValueError(f"Unknown source dataset spec: {source_dataset_name}")
                    source_label = f"{label_prefix}_direct_{source_dataset_name}"
                    specs.append(
                        ExperimentSpec(
                            label=f"{label_prefix}_direct_{dataset_spec.name}",
                            model_alias=model,
                            method="direct",
                            dataset_spec=dataset_spec,
                            reuse_from_label=source_label,
                        )
                    )
                    continue
                specs.append(
                    ExperimentSpec(
                        label=f"{label_prefix}_direct_{dataset_spec.name}",
                        model_alias=model,
                        method="direct",
                        dataset_spec=dataset_spec,
                    )
                )
        if include_remcts:
            for dataset_spec in dataset_specs:
                specs.append(
                    ExperimentSpec(
                        label=f"{label_prefix}_remcts_{dataset_spec.name}",
                        model_alias=model,
                        method="remcts",
                        dataset_spec=dataset_spec,
                    )
                )
    return specs


def load_completed_outputs(report_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".progress.json"):
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("label") and row.get("status") == "complete":
            rows.append(row)
    return rows


def summarize_all(
    report_dir: Path,
    rows: list[dict[str, Any]],
    started_at: str,
    *,
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> dict[str, Any]:
    totals = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_elapsed_sec": 0.0,
        "wall_elapsed_sec": 0.0,
        "estimated_cost_cny": 0.0,
    }
    table_rows: list[dict[str, Any]] = []
    for row in rows:
        usage_totals = row["llm_usage"]["totals"]
        hidden = row.get("hidden_summary")
        displayed_pass_rate = hidden["hidden_solve_rate"] if hidden else row["pass_rate"]
        pass_rate_metric = "hidden_solve_rate" if hidden else row.get("pass_rate_metric", "explicit_test_pass_rate")
        table_row = {
            "label": row["label"],
            "model_alias": row["model_alias"],
            "method": row["method"],
            "dataset": row["dataset"],
            "file": str(report_dir / f"{row['label']}.json"),
            "size": row["size"],
            "completed": row.get("completed", row["size"]),
            "pass_rate": displayed_pass_rate,
            "pass_rate_metric": pass_rate_metric,
            "raw_search_success_rate": row.get("raw_search_success_rate", row["pass_rate"]),
            "avg_iterations": row["avg_iterations"],
            "avg_tree_size": row["avg_tree_size"],
            "wall_elapsed_sec": row["wall_elapsed_sec"],
            "summed_task_wall_elapsed_sec": row.get("summed_task_wall_elapsed_sec", row["wall_elapsed_sec"]),
            "resumed_task_count": row.get("resumed_task_count", 0),
            "workers": row.get("workers", 1),
            "request_failed_tasks": row.get("request_failed_tasks", 0),
            "calls": usage_totals["calls"],
            "successful_calls": usage_totals["successful_calls"],
            "failed_calls": usage_totals["failed_calls"],
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "llm_elapsed_sec": usage_totals["elapsed_sec"],
            "estimated_cost_cny": row["estimated_cost_cny"],
            "hidden_summary": hidden,
        }
        table_rows.append(table_row)
        totals["calls"] += usage_totals["calls"]
        totals["successful_calls"] += usage_totals["successful_calls"]
        totals["failed_calls"] += usage_totals["failed_calls"]
        totals["prompt_tokens"] += usage_totals["prompt_tokens"]
        totals["completion_tokens"] += usage_totals["completion_tokens"]
        totals["total_tokens"] += usage_totals["total_tokens"]
        totals["llm_elapsed_sec"] += usage_totals["elapsed_sec"]
        totals["wall_elapsed_sec"] += row["wall_elapsed_sec"]
        totals["estimated_cost_cny"] += row["estimated_cost_cny"]
    totals["llm_elapsed_sec"] = round(totals["llm_elapsed_sec"], 6)
    totals["wall_elapsed_sec"] = round(totals["wall_elapsed_sec"], 6)
    totals["estimated_cost_cny"] = round(totals["estimated_cost_cny"], 6)
    budget = {
        "budget_cny": budget_cny,
        "initial_spent_cny": initial_spent_cny,
        "stop_margin_cny": stop_margin_cny,
        "report_estimated_cost_cny": totals["estimated_cost_cny"],
        "projected_total_spent_cny": round(initial_spent_cny + totals["estimated_cost_cny"], 6),
        "remaining_before_margin_cny": round(budget_cny - initial_spent_cny - totals["estimated_cost_cny"], 6)
        if budget_cny > 0
        else None,
    }
    summary = {
        "report_dir": str(report_dir),
        "stage": "qwen3 8B/14B/32B full ReMCTS and direct baselines",
        "started_at_utc": started_at,
        "finished_at_utc": utc_stamp(),
        "candidate_budget": {
            "max_iterations": 8,
            "branch_factor": 5,
            "max_candidates": 40,
            "path_memory_k": 3,
            "max_depth": 8,
            "exploration_c": 1.4,
            "no_improve_patience": 8,
            "direct_baseline_calls_per_task": 1,
            "direct_final_metric": "hidden_solve_rate only",
            "known_policy": "search may use visible tests; final submission is evaluated with hidden tests",
            "no_tests_policy": "search does not use visible tests; final submission is evaluated with hidden tests",
            "direct_no_tests_policy": "reuse known direct generations for hidden eval; no extra API calls",
            "enable_thinking": False,
        },
        "price_assumption_cny_per_million": {
            model: {"input": prices[0], "output": prices[1]}
            for model, prices in MODEL_PRICES_CNY_PER_MILLION.items()
        },
        "budget": budget,
        "rows": table_rows,
        "totals": totals,
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_summary_md(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Qwen3 8B/14B/32B Full Experiment Summary")
    lines.append("")
    lines.append(f"- Report dir: `{summary['report_dir']}`")
    lines.append("- Methods: `remcts` = LATS-like ReMCTS k=3; `direct` = one API request per task, no multi-turn dialogue")
    lines.append("- Direct final metric: hidden solve only; if a hidden evaluator exists, direct does not use visible tests as the final score")
    lines.append("- Known policy: visible tests may be used during search, but final submission quality is reported with hidden evaluation")
    lines.append("- No-tests policy: search does not use visible tests; final submission quality is reported with hidden evaluation")
    lines.append("- Direct no_tests policy: reuse the same known direct generations for hidden eval; no extra API calls unless explicitly requested")
    lines.append("- Budget: `max_iterations=8`, `branch_factor=5`, candidate budget 40")
    lines.append("- ReMCTS memory: `path_memory_k=3`, `global_memory_top_t=2`")
    lines.append("- Thinking: disabled (`enable_thinking=false`)")
    lines.append("- Retry: qwen configs use at least 5 client retries for retryable request failures")
    lines.append("")
    lines.append("## Budget")
    lines.append("")
    budget = summary["budget"]
    lines.append(
        "- budget=CNY {budget_cny:.2f}, initial_spent=CNY {initial_spent_cny:.2f}, "
        "report_cost=CNY {report_estimated_cost_cny:.6f}, projected_total=CNY {projected_total_spent_cny:.6f}, "
        "remaining_before_margin=CNY {remaining_before_margin_cny}".format(**budget)
    )
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append(
        "| Model | Method | Dataset | Size | Workers | Resumed | Solve Rate | Rate Metric | Avg Iter | Avg Tree | Wall Time | Calls | Input Tok | Output Tok | Total Tok | Failed Tasks | Est. CNY |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["rows"]:
        metric = "hidden solve" if row["pass_rate_metric"] == "hidden_solve_rate" else "explicit tests"
        lines.append(
            "| {model_alias} | {method} | {dataset} | {size} | {workers} | {resumed_task_count} | "
            "{pass_rate:.4f} | {metric} | {avg_iterations:.4f} | {avg_tree_size:.2f} | "
            "{wall_elapsed_sec:.2f}s | {calls} | {prompt_tokens} | {completion_tokens} | {total_tokens} | "
            "{request_failed_tasks} | {estimated_cost_cny:.6f} |".format(metric=metric, **row)
        )
    lines.append("")
    totals = summary["totals"]
    lines.append("## Totals")
    lines.append("")
    lines.append(
        "- calls={calls}, input={prompt_tokens}, output={completion_tokens}, total={total_tokens}, "
        "cost=CNY {estimated_cost_cny:.6f}, llm_elapsed={llm_elapsed_sec:.2f}s, wall_elapsed={wall_elapsed_sec:.2f}s".format(
            **totals
        )
    )
    hidden_rows = [row for row in summary["rows"] if row.get("hidden_summary")]
    if hidden_rows:
        lines.append("")
        lines.append("## Hidden Eval")
        lines.append("")
        lines.append("| Model | Method | Dataset | Tasks | Hidden Solve | Accepted | Needs Review | Acceptance Precision | False Acceptance |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in hidden_rows:
            hidden = row["hidden_summary"]
            lines.append(
                "| {model_alias} | {method} | {dataset} | {tasks} | {hidden_success}/{tasks} ({hidden_rate}) | "
                "{accepted}/{tasks} ({accepted_rate}) | {needs_review}/{tasks} ({review_rate}) | "
                "{precision} | {false_rate} |".format(
                    model_alias=row["model_alias"],
                    method=row["method"],
                    dataset=row["dataset"],
                    tasks=hidden["tasks"],
                    hidden_success=hidden["hidden_success"],
                    hidden_rate=percent(hidden["hidden_solve_rate"]),
                    accepted=hidden["accepted"],
                    accepted_rate=percent(hidden["accepted_rate"]),
                    needs_review=hidden["needs_review"],
                    review_rate=percent(hidden["needs_review_rate"]),
                    precision=percent(hidden["acceptance_precision"]),
                    false_rate=percent(hidden["false_acceptance_rate"]),
                )
            )
    lift_rows = architecture_lift_rows(summary["rows"])
    if lift_rows:
        lines.append("")
        lines.append("## Architecture Lift")
        lines.append("")
        lines.append("| Model | Dataset | Direct | ReMCTS | Delta |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for item in lift_rows:
            lines.append(
                "| {model_alias} | {dataset} | {direct:.4f} | {remcts:.4f} | {delta:+.4f} |".format(**item)
            )
    lines.append("")
    return "\n".join(lines)


def architecture_lift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        key = (row["model_alias"], row["dataset"])
        by_key.setdefault(key, {})[row["method"]] = float(row["pass_rate"])
    lift: list[dict[str, Any]] = []
    for (model_alias, dataset), values in sorted(by_key.items()):
        if "direct" not in values or "remcts" not in values:
            continue
        lift.append(
            {
                "model_alias": model_alias,
                "dataset": dataset,
                "direct": values["direct"],
                "remcts": values["remcts"],
                "delta": values["remcts"] - values["direct"],
            }
        )
    return lift


def main() -> None:
    parser = argparse.ArgumentParser(description="Run qwen3 8B/14B/32B full ReMCTS and direct baseline experiments.")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-14b", "qwen3-32b"])
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite completed run json files in the report dir.")
    parser.add_argument("--workers", type=int, default=10, help="Number of worker processes for unfinished tasks.")
    parser.add_argument("--budget-cny", type=float, default=100.0)
    parser.add_argument("--initial-spent-cny", type=float, default=0.0)
    parser.add_argument("--stop-margin-cny", type=float, default=5.0)
    parser.add_argument("--skip-remcts", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset names to run, e.g. humaneval_full_no_tests mbpp_sanitized_full_no_tests.",
    )
    parser.add_argument(
        "--label-suffix",
        default="",
        help="Optional suffix inserted into output labels to avoid overwriting prior runs.",
    )
    parser.add_argument(
        "--include-direct-no-tests",
        action="store_true",
        help="Actually call the API for direct no_tests splits. By default they reuse known direct generations.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    started_at = utc_stamp()
    report_dir = Path(args.report_dir) if args.report_dir else ROOT / "outputs/reports" / f"qwen3-8_14_32-full-k3-{started_at}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_experiment_specs(
        models=args.models,
        include_remcts=not args.skip_remcts,
        include_direct=not args.skip_direct,
        include_direct_no_tests=args.include_direct_no_tests,
        dataset_names=set(args.datasets) if args.datasets else None,
        label_suffix=args.label_suffix,
    )

    if args.dry_run:
        print(f"report_dir={report_dir}")
        projected = args.initial_spent_cny
        for spec in specs:
            estimate = estimate_spec_cost_cny(spec)
            projected += estimate
            print(f"{spec.label}: est_cny={estimate:.4f}, projected_total_with_initial={projected:.4f}")
        return

    workers = max(1, args.workers)
    try:
        for spec in specs:
            row = run_one_spec(
                spec,
                report_dir=report_dir,
                force=args.force,
                workers=workers,
                budget_cny=args.budget_cny,
                initial_spent_cny=args.initial_spent_cny,
                stop_margin_cny=args.stop_margin_cny,
            )
            rows = load_completed_outputs(report_dir)
            if row.get("status") == "complete" and all(existing.get("label") != row["label"] for existing in rows):
                rows.append(row)
            summary = summarize_all(
                report_dir,
                rows,
                started_at,
                budget_cny=args.budget_cny,
                initial_spent_cny=args.initial_spent_cny,
                stop_margin_cny=args.stop_margin_cny,
            )
            print(
                json.dumps(
                    {
                        "summary": summary["report_dir"],
                        "completed_rows": len(summary["rows"]),
                        "budget": summary["budget"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
    except BudgetExceeded as exc:
        rows = load_completed_outputs(report_dir)
        summary = summarize_all(
            report_dir,
            rows,
            started_at,
            budget_cny=args.budget_cny,
            initial_spent_cny=args.initial_spent_cny,
            stop_margin_cny=args.stop_margin_cny,
        )
        print(f"[budget-stop] {exc}", flush=True)
        print(json.dumps({"summary": summary["report_dir"], "budget": summary["budget"]}, ensure_ascii=False, indent=2))
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
