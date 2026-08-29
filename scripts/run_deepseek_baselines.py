#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from remcts.core.state import ProblemSpec
from remcts.llm.client import LLMClientError, create_llm_client
from remcts.llm.response_parser import extract_code, parse_json_block
from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from run_qwen_multi_model_experiment import (
    aggregate_usage,
    build_direct_prompt,
    compact_usage,
    estimate_cost_cny_for_model,
    hidden_summary,
    load_config,
    load_dataset,
)


DEFAULT_CONFIG = ROOT / "src/remcts/config/qwen_production_full_k3.yaml"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset: str
    data_path: Path
    hidden_eval_path: Path
    config_path: Path = DEFAULT_CONFIG


DATASETS: dict[str, DatasetSpec] = {
    "humaneval_full_no_tests": DatasetSpec(
        name="humaneval_full_no_tests",
        dataset="humaneval",
        data_path=ROOT / "data/humaneval/HumanEval_no_tests.jsonl",
        hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
    ),
    "mbpp_sanitized_full_no_tests": DatasetSpec(
        name="mbpp_sanitized_full_no_tests",
        dataset="mbpp",
        data_path=ROOT / "data/mbpp/mbpp_sanitized_no_tests.json",
        hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
    ),
    "humaneval_no_tests_5": DatasetSpec(
        name="humaneval_no_tests_5",
        dataset="humaneval",
        data_path=ROOT / "data/subsets/s0/humaneval_no_tests_5.jsonl",
        hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
    ),
    "mbpp_sanitized_no_tests_5": DatasetSpec(
        name="mbpp_sanitized_no_tests_5",
        dataset="mbpp",
        data_path=ROOT / "data/subsets/s0/mbpp_sanitized_no_tests_5.json",
        hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
    ),
}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug_model(model_alias: str) -> str:
    return model_alias.replace(".", "_").replace("-", "_")


def problem_to_dict(problem: ProblemSpec) -> dict[str, Any]:
    return problem.to_dict()


def problem_from_dict(data: dict[str, Any] | None) -> ProblemSpec | None:
    if data is None:
        return None
    return ProblemSpec(
        task_id=str(data.get("task_id", "unknown")),
        prompt=str(data.get("prompt") or ""),
        entry_point=data.get("entry_point"),
        tests=str(data.get("tests") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n...[truncated]..."


def build_generated_tests_prompt(problem: ProblemSpec, count: int) -> str:
    entry_point = problem.entry_point or problem.metadata.get("entry_point") or "solution"
    return f"""
You are the Evaluator test writer in a Reflexion-style programming loop.
Generate {count} diverse Python unit tests for the target function.

Requirements:
- Return JSON only, with exactly this shape: {{"tests": ["assert ...", "..."]}}.
- Every test must be a single Python assert statement.
- Use the entry function name `{entry_point}` exactly.
- Cover normal cases, edge cases, and boundary conditions when possible.
- Do not include imports, helper functions, Markdown, explanations, or the solution.
- Do not call stdin, files, network, random, or time-dependent APIs.

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def parse_generated_tests(raw: str, max_tests: int) -> str:
    candidates: list[str] = []
    try:
        data = parse_json_block(raw)
        raw_tests = data.get("tests", [])
        if isinstance(raw_tests, list):
            candidates.extend(str(item).strip() for item in raw_tests)
    except Exception:
        code = extract_code(raw)
        candidates.extend(line.strip() for line in code.splitlines())

    tests: list[str] = []
    for line in candidates:
        stripped = line.strip().rstrip(";")
        if not stripped:
            continue
        if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')):
            stripped = stripped[1:-1].strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
        if len(tests) >= max_tests:
            break
    return "\n".join(tests)


def normalize_candidate_code(code: str) -> str:
    stripped = str(code or "").lstrip()
    if stripped.startswith("from typing import"):
        return str(code or "").strip()
    return "from typing import *\n" + str(code or "").strip()


def build_best_of_prompt(problem: ProblemSpec, sample_index: int, sample_count: int) -> str:
    base = build_direct_prompt(problem)
    return f"""
{base}

Your output must start with `from typing import *` before any function definitions.
Include all other required imports explicitly.
This is independent sample {sample_index} of {sample_count}. Use your own reasoning and produce a robust implementation rather than copying a likely common template.
""".strip()


def build_actor_prompt(problem: ProblemSpec, generated_tests: str, reflections: list[str], trial: int) -> str:
    entry_point = problem.entry_point or problem.metadata.get("entry_point") or "solution"
    memory = "\n".join(f"- {item}" for item in reflections[-6:]) if reflections else "(empty)"
    tests = generated_tests.strip() or "(the Evaluator produced no usable assert tests; rely on the task specification)"
    return f"""
You are the Actor in a Reflexion-style programming loop.
Write a complete Python solution for the task below.

Return only complete executable Python code. Do not include Markdown fences or explanations.
The solution must define the entry function `{entry_point}` and must not rely on stdin, command-line input, or interactive prompts unless the task explicitly requires it.
Your output must start with `from typing import *` before any function definitions.
Include all other required imports explicitly.

Reflection memory from previous failed trials:
{memory}

Evaluator-generated tests used for feedback:
{tests}

Trial:
{trial}

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def build_reflection_prompt(
    problem: ProblemSpec,
    code: str,
    generated_tests: str,
    feedback: str,
    reflections: list[str],
) -> str:
    memory = "\n".join(f"- {item}" for item in reflections[-6:]) if reflections else "(empty)"
    return f"""
You are the Self-Reflection module in the Reflexion architecture.
Convert the failed programming trajectory into concise natural-language memory for the next Actor attempt.

Return JSON only, with this shape:
{{"summary": "...", "root_cause": "...", "fix_hint": "..."}}

Do not output code. Focus on what was wrong and how the next attempt should change.

Problem:
{problem.prompt}

Previous reflection memory:
{memory}

Evaluator-generated tests:
{generated_tests or "(none)"}

Candidate code:
{truncate_text(code, 2600)}

Evaluator feedback:
{truncate_text(feedback, 2600)}
""".strip()


def execution_feedback(result: Any) -> str:
    return "\n".join(
        [
            f"compile_ok: {bool(result.compile_ok)}",
            f"exec_ok: {bool(result.exec_ok)}",
            f"timed_out: {bool(result.timed_out)}",
            f"passed_tests: {int(result.passed_tests)}/{int(result.total_tests)}",
            f"failed_cases: {result.failed_cases[:8]}",
            f"traceback: {truncate_text(result.traceback, 1200)}",
            f"stderr: {truncate_text(result.stderr, 1200)}",
            f"stdout: {truncate_text(result.stdout, 600)}",
        ]
    )


def proxy_score(result: Any) -> float:
    compile_score = 1.0 if result.compile_ok else 0.0
    exec_score = 1.0 if result.exec_ok else 0.0
    if int(result.total_tests) > 0:
        test_score = float(result.passed_tests) / max(float(result.total_tests), 1.0)
        return round(0.70 * test_score + 0.20 * exec_score + 0.10 * compile_score, 6)
    return round(0.60 * exec_score + 0.40 * compile_score, 6)


def evaluator_success(result: Any) -> bool:
    return bool(result.exec_ok and result.total_tests > 0 and result.passed_tests == result.total_tests)


def generated_test_problem(problem: ProblemSpec, generated_tests: str) -> ProblemSpec:
    return ProblemSpec(
        task_id=problem.task_id,
        prompt=problem.prompt,
        entry_point=problem.entry_point,
        tests=generated_tests,
        metadata=dict(problem.metadata),
    )


def eval_generated_tests(problem: ProblemSpec, code: str, generated_tests: str, config: Any) -> Any:
    return run_problem_tests(
        generated_test_problem(problem, generated_tests),
        code,
        PythonExecutor(),
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )


def hidden_eval_result(hidden_problem: ProblemSpec | None, code: str, config: Any) -> Any | None:
    if hidden_problem is None or not code:
        return None
    return run_problem_tests(
        hidden_problem,
        code,
        PythonExecutor(),
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )


def apply_hidden_fields(row: dict[str, Any], hidden_problem: ProblemSpec | None, code: str, config: Any) -> None:
    result = hidden_eval_result(hidden_problem, code, config)
    if result is None:
        return
    ok = bool(result.exec_ok and result.total_tests > 0 and result.passed_tests == result.total_tests)
    row.update(
        {
            "hidden_success": ok,
            "hidden_passed_tests": int(result.passed_tests),
            "hidden_total_tests": int(result.total_tests),
            "hidden_failed_cases": list(result.failed_cases[:8]),
        }
    )


def generated_tests_summary(generated_tests: str) -> dict[str, Any]:
    tests = [line for line in generated_tests.splitlines() if line.strip().startswith("assert ")]
    return {"count": len(tests), "tests": tests}


def run_best_of_n(
    client: Any,
    problem: ProblemSpec,
    hidden_problem: ProblemSpec | None,
    config: Any,
    n: int,
    generated_test_count: int,
    best_of_temperature: float,
    test_temperature: float,
    compute_oracle_pool: bool,
) -> dict[str, Any]:
    raw_tests = client.generate(
        build_generated_tests_prompt(problem, generated_test_count),
        temperature=test_temperature,
        max_tokens=min(int(config.llm_max_tokens), 900),
    )
    generated_tests = parse_generated_tests(raw_tests, generated_test_count)
    candidates: list[dict[str, Any]] = []

    for sample_index in range(1, n + 1):
        raw = client.generate(
            build_best_of_prompt(problem, sample_index, n),
            temperature=best_of_temperature,
            max_tokens=config.llm_max_tokens,
        )
        code = normalize_candidate_code(extract_code(raw))
        result = eval_generated_tests(problem, code, generated_tests, config)
        item = {
            "sample_index": sample_index,
            "proxy_score": proxy_score(result),
            "evaluator_success": evaluator_success(result),
            "passed_tests": int(result.passed_tests),
            "total_tests": int(result.total_tests),
            "compile_ok": bool(result.compile_ok),
            "exec_ok": bool(result.exec_ok),
            "timed_out": bool(result.timed_out),
            "failed_cases": list(result.failed_cases[:8]),
            "code": code,
        }
        if compute_oracle_pool:
            hidden = hidden_eval_result(hidden_problem, code, config)
            if hidden is not None:
                item["hidden_success"] = bool(
                    hidden.exec_ok and hidden.total_tests > 0 and hidden.passed_tests == hidden.total_tests
                )
                item["hidden_passed_tests"] = int(hidden.passed_tests)
                item["hidden_total_tests"] = int(hidden.total_tests)
        candidates.append(item)

    selected = max(
        candidates,
        key=lambda item: (
            float(item["proxy_score"]),
            bool(item["evaluator_success"]),
            bool(item["exec_ok"]),
            -int(item["sample_index"]),
        ),
    )
    accepted = bool(selected["evaluator_success"])
    row = {
        "method": f"best_of_{n}",
        "success": accepted,
        "accepted": accepted,
        "needs_review": not accepted,
        "iterations": n,
        "tree_size": n,
        "stop_reason": "best_of_n_generated_tests_passed" if accepted else "best_of_n_selected_by_proxy_score",
        "generated_tests": generated_tests_summary(generated_tests),
        "selected_sample_index": int(selected["sample_index"]),
        "selected_proxy_score": float(selected["proxy_score"]),
        "selected_evaluator_passed_tests": int(selected["passed_tests"]),
        "selected_evaluator_total_tests": int(selected["total_tests"]),
        "oracle_pool_hidden_success": any(bool(item.get("hidden_success")) for item in candidates),
        "oracle_pool_hidden_success_count": sum(1 for item in candidates if item.get("hidden_success")),
        "candidate_summaries": [
            {key: value for key, value in item.items() if key != "code"} for item in candidates
        ],
        "selected_code": str(selected["code"]),
    }
    apply_hidden_fields(row, hidden_problem, str(selected["code"]), config)
    return row


def parse_reflection(raw: str) -> str:
    try:
        data = parse_json_block(raw)
        parts = [
            str(data.get("summary") or "").strip(),
            str(data.get("root_cause") or "").strip(),
            str(data.get("fix_hint") or "").strip(),
        ]
        text = " ".join(part for part in parts if part)
        return text.strip() or truncate_text(raw.strip(), 1200)
    except Exception:
        return truncate_text(raw.strip(), 1200)


def run_reflexion(
    client: Any,
    problem: ProblemSpec,
    hidden_problem: ProblemSpec | None,
    config: Any,
    trials: int,
    generated_test_count: int,
    actor_temperature: float,
    test_temperature: float,
    reflection_temperature: float,
    compute_oracle_pool: bool,
) -> dict[str, Any]:
    raw_tests = client.generate(
        build_generated_tests_prompt(problem, generated_test_count),
        temperature=test_temperature,
        max_tokens=min(int(config.llm_max_tokens), 900),
    )
    generated_tests = parse_generated_tests(raw_tests, generated_test_count)
    reflections: list[str] = []
    attempts: list[dict[str, Any]] = []

    selected: dict[str, Any] | None = None
    for trial in range(1, trials + 1):
        raw = client.generate(
            build_actor_prompt(problem, generated_tests, reflections, trial),
            temperature=actor_temperature,
            max_tokens=config.llm_max_tokens,
        )
        code = normalize_candidate_code(extract_code(raw))
        result = eval_generated_tests(problem, code, generated_tests, config)
        attempt = {
            "trial": trial,
            "proxy_score": proxy_score(result),
            "evaluator_success": evaluator_success(result),
            "passed_tests": int(result.passed_tests),
            "total_tests": int(result.total_tests),
            "compile_ok": bool(result.compile_ok),
            "exec_ok": bool(result.exec_ok),
            "timed_out": bool(result.timed_out),
            "failed_cases": list(result.failed_cases[:8]),
            "feedback": truncate_text(execution_feedback(result), 2200),
            "code": code,
        }
        if compute_oracle_pool:
            hidden = hidden_eval_result(hidden_problem, code, config)
            if hidden is not None:
                attempt["hidden_success"] = bool(
                    hidden.exec_ok and hidden.total_tests > 0 and hidden.passed_tests == hidden.total_tests
                )
                attempt["hidden_passed_tests"] = int(hidden.passed_tests)
                attempt["hidden_total_tests"] = int(hidden.total_tests)
        attempts.append(attempt)

        if evaluator_success(result):
            selected = attempt
            break

        reflection_raw = client.generate(
            build_reflection_prompt(problem, code, generated_tests, execution_feedback(result), reflections),
            temperature=reflection_temperature,
            max_tokens=min(int(config.llm_max_tokens), 800),
        )
        reflection = parse_reflection(reflection_raw)
        if reflection:
            reflections.append(reflection)

    if selected is None:
        selected = max(
            attempts,
            key=lambda item: (
                float(item["proxy_score"]),
                bool(item["evaluator_success"]),
                bool(item["exec_ok"]),
                -int(item["trial"]),
            ),
        )

    accepted = bool(selected["evaluator_success"])
    row = {
        "method": "reflexion",
        "success": accepted,
        "accepted": accepted,
        "needs_review": not accepted,
        "iterations": int(selected["trial"]) if accepted else len(attempts),
        "tree_size": len(attempts),
        "stop_reason": "reflexion_generated_tests_passed" if accepted else "reflexion_max_trials",
        "generated_tests": generated_tests_summary(generated_tests),
        "reflection_count": len(reflections),
        "reflections": reflections,
        "selected_trial": int(selected["trial"]),
        "selected_proxy_score": float(selected["proxy_score"]),
        "selected_evaluator_passed_tests": int(selected["passed_tests"]),
        "selected_evaluator_total_tests": int(selected["total_tests"]),
        "oracle_pool_hidden_success": any(bool(item.get("hidden_success")) for item in attempts),
        "oracle_pool_hidden_success_count": sum(1 for item in attempts if item.get("hidden_success")),
        "attempt_summaries": [
            {key: value for key, value in item.items() if key != "code"} for item in attempts
        ],
        "selected_code": str(selected["code"]),
    }
    apply_hidden_fields(row, hidden_problem, str(selected["code"]), config)
    return row


def run_one_task(
    *,
    task_index: int,
    problem_data: dict[str, Any],
    hidden_problem_data: dict[str, Any] | None,
    dataset_name: str,
    method: str,
    model_alias: str,
    config_path: str,
    best_of_n: int,
    reflexion_trials: int,
    generated_test_count: int,
    best_of_temperature: float,
    reflexion_temperature: float,
    test_temperature: float,
    reflection_temperature: float,
    compute_oracle_pool: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    client = None
    problem = problem_from_dict(problem_data)
    hidden_problem = problem_from_dict(hidden_problem_data)
    assert problem is not None
    try:
        config = load_config(Path(config_path), model_alias)
        client = create_llm_client(config)
        if hasattr(client, "reset_usage"):
            client.reset_usage()

        if method == "best_of_8":
            row = run_best_of_n(
                client=client,
                problem=problem,
                hidden_problem=hidden_problem,
                config=config,
                n=best_of_n,
                generated_test_count=generated_test_count,
                best_of_temperature=best_of_temperature,
                test_temperature=test_temperature,
                compute_oracle_pool=compute_oracle_pool,
            )
        elif method == "reflexion":
            row = run_reflexion(
                client=client,
                problem=problem,
                hidden_problem=hidden_problem,
                config=config,
                trials=reflexion_trials,
                generated_test_count=generated_test_count,
                actor_temperature=reflexion_temperature,
                test_temperature=test_temperature,
                reflection_temperature=reflection_temperature,
                compute_oracle_pool=compute_oracle_pool,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")
    except (Exception, LLMClientError) as exc:
        row = {
            "method": method,
            "success": False,
            "accepted": False,
            "needs_review": True,
            "iterations": 0,
            "tree_size": 0,
            "stop_reason": "request_failed",
            "request_failed": True,
            "error": f"{type(exc).__name__}: {exc}",
            "selected_code": "",
        }

    usage = compact_usage(client.usage_snapshot()) if client is not None else compact_usage({})
    row.update(
        {
            "task_index": task_index,
            "task_id": problem.task_id,
            "dataset": dataset_name,
            "model_alias": model_alias,
            "wall_elapsed_sec": round(time.perf_counter() - started, 6),
            "llm_usage": usage,
        }
    )
    return row


def load_existing_rows(progress_path: Path, force: bool) -> dict[str, dict[str, Any]]:
    if force or not progress_path.exists():
        return {}
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    rows = progress.get("tasks", [])
    return {str(row.get("task_id")): row for row in rows if row.get("task_id")}


def write_progress(progress_path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda item: int(item.get("task_index") or 0))
    progress = dict(metadata)
    progress.update(
        {
            "status": "running",
            "completed": len(ordered),
            "tasks": ordered,
        }
    )
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_output(
    *,
    output_path: Path,
    label: str,
    model_alias: str,
    method: str,
    dataset_spec: DatasetSpec,
    tasks: list[ProblemSpec],
    rows: list[dict[str, Any]],
    started: float,
    resumed_count: int,
    workers: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda item: int(item.get("task_index") or 0))
    usage_summary = aggregate_usage(rows)
    totals = usage_summary["totals"]
    hidden = hidden_summary(rows)
    output = {
        "label": label,
        "model_alias": model_alias,
        "method": method,
        "dataset": dataset_spec.name,
        "dataset_type": dataset_spec.dataset,
        "data_path": str(dataset_spec.data_path),
        "hidden_eval_path": str(dataset_spec.hidden_eval_path),
        "config_path": str(dataset_spec.config_path),
        "status": "complete" if len(rows) == len(tasks) else "partial",
        "size": len(tasks),
        "completed": len(rows),
        "pass_rate": hidden["hidden_solve_rate"] if hidden else 0.0,
        "pass_rate_metric": "hidden_solve_rate",
        "raw_search_success_rate": round(sum(1 for row in rows if row.get("success")) / len(rows), 6) if rows else 0.0,
        "avg_iterations": round(sum(int(row.get("iterations") or 0) for row in rows) / len(rows), 6) if rows else 0.0,
        "avg_tree_size": round(sum(int(row.get("tree_size") or 0) for row in rows) / len(rows), 6) if rows else 0.0,
        "oracle_pool_hidden_success_rate": round(
            sum(1 for row in rows if row.get("oracle_pool_hidden_success")) / len(rows), 6
        )
        if rows
        else 0.0,
        "request_failed_tasks": sum(1 for row in rows if row.get("request_failed")),
        "wall_elapsed_sec": round(time.perf_counter() - started, 6),
        "summed_task_wall_elapsed_sec": round(sum(float(row.get("wall_elapsed_sec") or 0.0) for row in rows), 6),
        "resumed_task_count": resumed_count,
        "workers": workers,
        "parameters": parameters,
        "llm_usage": usage_summary,
        "estimated_cost_cny": estimate_cost_cny_for_model(
            model_alias,
            int(totals["prompt_tokens"]),
            int(totals["completion_tokens"]),
        ),
        "hidden_summary": hidden,
        "tasks": rows,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_spec(
    *,
    dataset_spec: DatasetSpec,
    method: str,
    model_alias: str,
    report_dir: Path,
    workers: int,
    force: bool,
    limit: int | None,
    best_of_n: int,
    reflexion_trials: int,
    generated_test_count: int,
    best_of_temperature: float,
    reflexion_temperature: float,
    test_temperature: float,
    reflection_temperature: float,
    compute_oracle_pool: bool,
) -> dict[str, Any]:
    label = f"{slug_model(model_alias)}_{method}_{dataset_spec.name}"
    output_path = report_dir / f"{label}.json"
    progress_path = report_dir / f"{label}.progress.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[skip] {label}: {output_path}", flush=True)
            return existing

    tasks = load_dataset(dataset_spec.dataset, dataset_spec.data_path)
    if limit is not None:
        tasks = tasks[:limit]
    hidden_map = {
        problem.task_id: problem
        for problem in load_dataset(dataset_spec.dataset, dataset_spec.hidden_eval_path)
    }
    existing_by_task = load_existing_rows(progress_path, force=force)
    rows = [existing_by_task[problem.task_id] for problem in tasks if problem.task_id in existing_by_task]
    pending = [problem for problem in tasks if problem.task_id not in existing_by_task]
    resumed_count = len(rows)
    started = time.perf_counter()
    parameters = {
        "best_of_n": best_of_n,
        "reflexion_trials": reflexion_trials,
        "generated_test_count": generated_test_count,
        "best_of_temperature": best_of_temperature,
        "reflexion_temperature": reflexion_temperature,
        "test_temperature": test_temperature,
        "reflection_temperature": reflection_temperature,
        "compute_oracle_pool": compute_oracle_pool,
        "typing_prelude": "from typing import *",
    }
    metadata = {
        "label": label,
        "model_alias": model_alias,
        "method": method,
        "dataset": dataset_spec.name,
        "size": len(tasks),
        "parameters": parameters,
    }
    write_progress(progress_path, metadata, rows)
    print(f"[run] {label}: total={len(tasks)} resumed={resumed_count} pending={len(pending)}", flush=True)

    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(
                    run_one_task,
                    task_index=tasks.index(problem),
                    problem_data=problem_to_dict(problem),
                    hidden_problem_data=problem_to_dict(hidden_map[problem.task_id])
                    if problem.task_id in hidden_map
                    else None,
                    dataset_name=dataset_spec.name,
                    method=method,
                    model_alias=model_alias,
                    config_path=str(dataset_spec.config_path),
                    best_of_n=best_of_n,
                    reflexion_trials=reflexion_trials,
                    generated_test_count=generated_test_count,
                    best_of_temperature=best_of_temperature,
                    reflexion_temperature=reflexion_temperature,
                    test_temperature=test_temperature,
                    reflection_temperature=reflection_temperature,
                    compute_oracle_pool=compute_oracle_pool,
                ): problem
                for problem in pending
            }
            for completed, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
                problem = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "task_index": tasks.index(problem),
                        "task_id": problem.task_id,
                        "dataset": dataset_spec.name,
                        "model_alias": model_alias,
                        "method": method,
                        "success": False,
                        "accepted": False,
                        "needs_review": True,
                        "iterations": 0,
                        "tree_size": 0,
                        "stop_reason": "worker_failed",
                        "request_failed": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "selected_code": "",
                        "wall_elapsed_sec": 0.0,
                        "llm_usage": compact_usage({}),
                    }
                rows.append(row)
                if completed % max(1, min(10, workers)) == 0 or len(rows) == len(tasks):
                    write_progress(progress_path, metadata, rows)
                    hidden = hidden_summary(rows)
                    hidden_rate = hidden["hidden_solve_rate"] if hidden else 0.0
                    print(
                        f"[progress] {label}: {len(rows)}/{len(tasks)} hidden={hidden_rate:.4f}",
                        flush=True,
                    )

    output = finalize_output(
        output_path=output_path,
        label=label,
        model_alias=model_alias,
        method=method,
        dataset_spec=dataset_spec,
        tasks=tasks,
        rows=rows,
        started=started,
        resumed_count=resumed_count,
        workers=workers,
        parameters=parameters,
    )
    progress_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] {label}: pass_rate={output['pass_rate']:.4f} cost_cny={output['estimated_cost_cny']}",
        flush=True,
    )
    return output


def write_summary(report_dir: Path, outputs: list[dict[str, Any]]) -> None:
    summary_json = report_dir / "summary.json"
    summary_md = report_dir / "summary.md"
    ordered = sorted(outputs, key=lambda item: (item.get("dataset", ""), item.get("method", "")))
    summary_json.write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# DeepSeek-V4-Flash Best-of-N 与 Reflexion-style 补充实验",
        "",
        f"- 生成时间：{utc_stamp()}",
        "- 口径：搜索/选择阶段不使用 hidden tests，最终统一使用 hidden evaluation 计分。",
        "- Best-of-8：每题生成 8 个独立候选，用自生成断言测试与执行结果选择单个候选。",
        "- Reflexion-style：按 Actor、Evaluator、自反思、记忆闭环迭代；Evaluator 使用模型自生成单元测试。",
        "- `oracle_pool_hidden_success_rate` 仅为离线候选池上界分析，不参与候选选择。",
        "- 候选代码执行前统一加入 `from typing import *` prelude，用于贴近 HumanEval completion 评测语境；该处理不使用 hidden 反馈。",
        "",
        "| 数据集 | 方法 | 完成 | Hidden Solve Rate | Accepted Rate | Acceptance Precision | Avg Iterations | Oracle Pool Hidden | 费用(CNY) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for output in ordered:
        hidden = output.get("hidden_summary") or {}
        completed = f"{output.get('completed', 0)}/{output.get('size', 0)}"
        lines.append(
            "| {dataset} | {method} | {completed} | {hidden_rate:.2%} | {accepted_rate:.2%} | "
            "{precision:.2%} | {avg_iter:.3f} | {oracle_rate:.2%} | {cost:.4f} |".format(
                dataset=output.get("dataset"),
                method=output.get("method"),
                completed=completed,
                hidden_rate=float(output.get("pass_rate") or 0.0),
                accepted_rate=float(hidden.get("accepted_rate") or 0.0),
                precision=float(hidden.get("acceptance_precision") or 0.0),
                avg_iter=float(output.get("avg_iterations") or 0.0),
                oracle_rate=float(output.get("oracle_pool_hidden_success_rate") or 0.0),
                cost=float(output.get("estimated_cost_cny") or 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "Best-of-8 的 selected candidate 由代理测试通过率、执行成功和编译成功排序得到；hidden tests 只在最终离线评测与候选池上界统计时运行。",
            "",
            "Reflexion-style 的一次迭代包含 Actor 生成代码、Evaluator 运行自生成测试、Self-Reflection 将失败反馈写入自然语言记忆。若某轮通过 Evaluator 测试则提前停止，否则达到最大轮数后选择代理分最高的候选。",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-alias", default="deepseek-v4-flash")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval_full_no_tests", "mbpp_sanitized_full_no_tests"],
        choices=sorted(DATASETS),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["best_of_8", "reflexion"],
        choices=["best_of_8", "reflexion"],
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--best-of-n", type=int, default=8)
    parser.add_argument("--reflexion-trials", type=int, default=8)
    parser.add_argument("--generated-test-count", type=int, default=5)
    parser.add_argument("--best-of-temperature", type=float, default=0.7)
    parser.add_argument("--reflexion-temperature", type=float, default=0.2)
    parser.add_argument("--test-temperature", type=float, default=0.2)
    parser.add_argument("--reflection-temperature", type=float, default=0.2)
    parser.add_argument("--no-oracle-pool", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir or (ROOT / "outputs/reports" / f"deepseek-baselines-{utc_stamp()}")
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    for dataset_name in args.datasets:
        dataset_spec = DATASETS[dataset_name]
        for method in args.methods:
            outputs.append(
                run_spec(
                    dataset_spec=dataset_spec,
                    method=method,
                    model_alias=args.model_alias,
                    report_dir=report_dir,
                    workers=max(1, int(args.workers)),
                    force=bool(args.force),
                    limit=args.limit,
                    best_of_n=int(args.best_of_n),
                    reflexion_trials=int(args.reflexion_trials),
                    generated_test_count=int(args.generated_test_count),
                    best_of_temperature=float(args.best_of_temperature),
                    reflexion_temperature=float(args.reflexion_temperature),
                    test_temperature=float(args.test_temperature),
                    reflection_temperature=float(args.reflection_temperature),
                    compute_oracle_pool=not bool(args.no_oracle_pool),
                )
            )
            write_summary(report_dir, outputs)
    write_summary(report_dir, outputs)
    print(f"[summary] {report_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
