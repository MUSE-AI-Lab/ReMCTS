#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from collections import deque
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from api_key_pool import load_key_pool
from run_deepseek_baselines import (
    apply_hidden_fields,
    build_actor_prompt,
    build_reflection_prompt,
    evaluator_success,
    execution_feedback,
    normalize_candidate_code,
    parse_reflection,
    problem_from_dict,
    problem_to_dict,
    proxy_score,
)
from run_qwen_multi_model_experiment import (
    DatasetSpec,
    ExperimentSpec,
    ROOT,
    _attach_rotation_stats,
    _config_and_client_for_task,
    _finalize_spec_output,
    _is_retryable_key_error,
    _key_rotation_is_enabled,
    _load_progress,
    _max_key_switches,
    _merge_compact_usages,
    _run_task_worker,
    _write_progress,
    aggregate_usage,
    compact_usage,
    estimate_cost_cny_for_model,
    hidden_summary,
    load_completed_outputs,
    load_dataset,
    run_one_spec,
    slug_model,
    stable_key_slot,
    summarize_all,
    utc_stamp,
)
from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from remcts.llm.response_parser import extract_code


DATASETS = {
    "humaneval_full_known": {
        "dataset": "humaneval",
        "data_path": ROOT / "data/humaneval/HumanEval.jsonl",
        "hidden_eval_path": ROOT / "data/humaneval/HumanEval_eval.jsonl",
    },
    "mbpp_sanitized_full_known": {
        "dataset": "mbpp",
        "data_path": ROOT / "data/mbpp/mbpp_sanitized.json",
        "hidden_eval_path": ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run known-setting Reflection-style and MCTS-style baseline completions.")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b", "deepseek-v4-flash"])
    parser.add_argument("--methods", nargs="+", default=["reflection_style", "mcts_style"], choices=["reflection_style", "mcts_style"])
    parser.add_argument("--datasets", nargs="+", default=["humaneval_full_known", "mbpp_sanitized_full_known"], choices=sorted(DATASETS))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-cny", type=float, default=120.0)
    parser.add_argument("--initial-spent-cny", type=float, default=0.0)
    parser.add_argument("--stop-margin-cny", type=float, default=5.0)
    parser.add_argument("--credentials-file", type=Path, default=ROOT / "apikeys.txt")
    parser.add_argument("--key-rotation", default="round_robin", choices=["round_robin", "disabled"])
    parser.add_argument("--max-key-switches", type=int, default=4)
    parser.add_argument("--global-queue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reflexion-trials", type=int, default=8)
    parser.add_argument("--actor-temperature", type=float, default=0.2)
    parser.add_argument("--reflection-temperature", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None, help="Limit tasks per dataset for smoke runs.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def configure_key_rotation(args: argparse.Namespace) -> None:
    if args.key_rotation == "disabled":
        os.environ.pop("REMCTS_KEY_POOL_FILE", None)
        os.environ["REMCTS_KEY_ROTATION"] = "disabled"
        return
    os.environ["REMCTS_KEY_POOL_FILE"] = str(args.credentials_file)
    os.environ["REMCTS_KEY_ROTATION"] = args.key_rotation
    os.environ["REMCTS_MAX_KEY_SWITCHES"] = str(max(1, int(args.max_key_switches)))


def print_key_pool_summary(args: argparse.Namespace) -> None:
    if args.key_rotation == "disabled":
        print("[keys] rotation disabled", flush=True)
        return
    pool = load_key_pool(args.credentials_file, model_aliases=list(args.models))
    for model in args.models:
        profiles = pool.get(model, [])
        fingerprints = ", ".join(profile.fingerprint for profile in profiles[:4])
        more = "..." if len(profiles) > 4 else ""
        print(f"[keys] {model}: {len(profiles)} profiles [{fingerprints}{more}]", flush=True)
        if not profiles:
            raise RuntimeError(f"No API keys available for model {model} in {args.credentials_file}")


def make_report_dir(args: argparse.Namespace) -> Path:
    report_dir = args.report_dir or ROOT / "outputs/reports" / f"known-baseline-completions-{utc_stamp()}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "configs").mkdir(exist_ok=True)
    if args.limit is not None:
        (report_dir / "data").mkdir(exist_ok=True)
    return report_dir


def write_mcts_style_known_config(report_dir: Path, credentials_file: Path) -> Path:
    config = yaml.safe_load((ROOT / "src/remcts/config/qwen_full_k3.yaml").read_text(encoding="utf-8"))
    config = dict(config or {})
    search = dict(config.get("search") or {})
    search.update({"path_memory_k": 0, "global_memory_top_t": 0})
    config["search"] = search
    semantic = dict(config.get("semantic") or {})
    semantic.update({"alpha": 0.0, "beta": 0.0})
    config["semantic"] = semantic
    reward = dict(config.get("reward") or {})
    reward.update({"semantic_mode": "llm", "semantic_on_failure": True})
    config["reward"] = reward
    llm = dict(config.get("llm") or {})
    llm["credentials_file"] = str(credentials_file)
    llm["enable_thinking"] = False
    llm["max_retries"] = max(int(llm.get("max_retries", 2)), 5)
    config["llm"] = llm
    path = report_dir / "configs" / "qwen_full_k3_mcts_style_known.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def subset_data_path(report_dir: Path, dataset_name: str, source: Path, limit: int | None) -> Path:
    if limit is None:
        return source
    target = report_dir / "data" / f"{dataset_name}_limit{limit}{source.suffix}"
    if source.suffix == ".jsonl":
        lines = source.read_text(encoding="utf-8").splitlines()
        target.write_text("\n".join(lines[:limit]) + "\n", encoding="utf-8")
        return target
    data = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(data, list):
        subset = data[:limit]
    elif isinstance(data, dict) and isinstance(data.get("tasks"), list):
        subset = dict(data)
        subset["tasks"] = subset["tasks"][:limit]
    else:
        raise ValueError(f"Unsupported JSON dataset format for subset: {source}")
    target.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def build_dataset_specs(report_dir: Path, args: argparse.Namespace, mcts_config: Path) -> dict[str, DatasetSpec]:
    specs = {}
    for name in args.datasets:
        info = DATASETS[name]
        specs[name] = DatasetSpec(
            name=name,
            dataset=str(info["dataset"]),
            data_path=subset_data_path(report_dir, name, Path(info["data_path"]), args.limit),
            remcts_config_path=mcts_config,
            hidden_eval_path=Path(info["hidden_eval_path"]),
        )
    return specs


def visible_eval_result(problem: Any, code: str, config: Any) -> Any:
    return run_problem_tests(
        problem,
        code,
        PythonExecutor(),
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )


def run_reflection_known_attempt(
    *,
    client: Any,
    problem: Any,
    hidden_problem: Any,
    config: Any,
    trials: int,
    actor_temperature: float,
    reflection_temperature: float,
) -> dict[str, Any]:
    visible_tests = problem.tests.strip() or "(no visible tests available)"
    reflections: list[str] = []
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for trial in range(1, trials + 1):
        raw = client.generate(
            build_actor_prompt(problem, visible_tests, reflections, trial),
            temperature=actor_temperature,
            max_tokens=config.llm_max_tokens,
        )
        code = normalize_candidate_code(extract_code(raw))
        result = visible_eval_result(problem, code, config)
        attempt = {
            "trial": trial,
            "proxy_score": proxy_score(result),
            "evaluator_success": evaluator_success(result),
            "visible_passed_tests": int(result.passed_tests),
            "visible_total_tests": int(result.total_tests),
            "compile_ok": bool(result.compile_ok),
            "exec_ok": bool(result.exec_ok),
            "timed_out": bool(result.timed_out),
            "failed_cases": list(result.failed_cases[:8]),
            "feedback": execution_feedback(result),
            "code": code,
        }
        attempts.append(attempt)
        if evaluator_success(result):
            selected = attempt
            break
        reflection_raw = client.generate(
            build_reflection_prompt(problem, code, visible_tests, execution_feedback(result), reflections),
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
        "method": "reflection_style",
        "success": accepted,
        "accepted": accepted,
        "needs_review": not accepted,
        "iterations": int(selected["trial"]) if accepted else len(attempts),
        "tree_size": len(attempts),
        "stop_reason": "reflection_visible_tests_passed" if accepted else "reflection_max_trials",
        "visible_test_count": len([line for line in visible_tests.splitlines() if line.strip()]),
        "reflection_count": len(reflections),
        "reflections": reflections,
        "selected_trial": int(selected["trial"]),
        "selected_proxy_score": float(selected["proxy_score"]),
        "selected_visible_passed_tests": int(selected["visible_passed_tests"]),
        "selected_visible_total_tests": int(selected["visible_total_tests"]),
        "attempt_summaries": [{key: value for key, value in item.items() if key != "code"} for item in attempts],
        "selected_code": str(selected["code"]),
    }
    apply_hidden_fields(row, hidden_problem, str(selected["code"]), config)
    return row


def run_reflection_known_task(payload: dict[str, Any]) -> dict[str, Any]:
    problem = problem_from_dict(payload["problem"])
    hidden_problem = problem_from_dict(payload["hidden_problem"])
    assert problem is not None
    task_started = time.perf_counter()
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
            row = run_reflection_known_attempt(
                client=client,
                problem=problem,
                hidden_problem=hidden_problem,
                config=config,
                trials=int(payload["reflexion_trials"]),
                actor_temperature=float(payload["actor_temperature"]),
                reflection_temperature=float(payload["reflection_temperature"]),
            )
            attempted_usages.append(compact_usage(client.usage_snapshot()))
            row.update(
                {
                    "task_index": int(payload["task_index"]),
                    "task_id": problem.task_id,
                    "dataset": str(payload["dataset"]),
                    "model_alias": str(payload["model_alias"]),
                    "wall_elapsed_sec": round(time.perf_counter() - task_started, 6),
                    "llm_usage": _merge_compact_usages(attempted_usages),
                }
            )
            _attach_rotation_stats(row, rotation_stats, retryable_errors)
            return row
        except Exception as exc:
            last_error = exc
            attempted_usages.append(compact_usage(client.usage_snapshot() if client is not None else None))
            if _key_rotation_is_enabled() and _is_retryable_key_error(exc) and key_attempt < max_attempts - 1:
                retryable_errors += 1
                continue
            row = {
                "task_index": int(payload["task_index"]),
                "task_id": problem.task_id,
                "dataset": str(payload["dataset"]),
                "model_alias": str(payload["model_alias"]),
                "method": "reflection_style",
                "success": False,
                "accepted": False,
                "needs_review": True,
                "iterations": 0,
                "tree_size": 0,
                "stop_reason": "request_failed",
                "request_failed": True,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "selected_code": "",
                "wall_elapsed_sec": round(time.perf_counter() - task_started, 6),
                "llm_usage": _merge_compact_usages(attempted_usages),
            }
            _attach_rotation_stats(row, last_stats, retryable_errors)
            return row
    row = {
        "task_index": int(payload["task_index"]),
        "task_id": problem.task_id,
        "dataset": str(payload["dataset"]),
        "model_alias": str(payload["model_alias"]),
        "method": "reflection_style",
        "success": False,
        "accepted": False,
        "needs_review": True,
        "iterations": 0,
        "tree_size": 0,
        "stop_reason": "key_rotation_exhausted",
        "request_failed": True,
        "error": f"{type(last_error).__name__}: {last_error}"[:1000] if last_error else "key rotation exhausted",
        "selected_code": "",
        "wall_elapsed_sec": round(time.perf_counter() - task_started, 6),
        "llm_usage": _merge_compact_usages(attempted_usages),
    }
    _attach_rotation_stats(row, last_stats, retryable_errors)
    return row


def load_existing_rows(progress_path: Path, force: bool) -> dict[str, dict[str, Any]]:
    if force or not progress_path.exists():
        return {}
    try:
        data = json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(row.get("task_id")): row for row in data.get("tasks", []) if row.get("task_id") and not row.get("request_failed")}


def write_progress(progress_path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda item: int(item.get("task_index") or 0))
    progress = dict(metadata)
    progress.update({"completed": len(ordered), "tasks": ordered})
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_reflection_output(
    *,
    output_path: Path,
    label: str,
    model_alias: str,
    dataset_spec: DatasetSpec,
    tasks: list[Any],
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
        "method": "reflection_style",
        "dataset": dataset_spec.name,
        "dataset_type": dataset_spec.dataset,
        "data_path": str(dataset_spec.data_path),
        "hidden_eval_path": str(dataset_spec.hidden_eval_path),
        "config_path": str(dataset_spec.remcts_config_path),
        "status": "complete" if len(rows) == len(tasks) and not any(row.get("request_failed") for row in rows) else "partial",
        "size": len(tasks),
        "completed": len(rows),
        "pass_rate": hidden["hidden_solve_rate"] if hidden else 0.0,
        "pass_rate_metric": "hidden_solve_rate",
        "raw_search_success_rate": round(sum(1 for row in rows if row.get("success")) / len(rows), 6) if rows else 0.0,
        "avg_iterations": round(sum(int(row.get("iterations") or 0) for row in rows) / len(rows), 6) if rows else 0.0,
        "avg_tree_size": round(sum(int(row.get("tree_size") or 0) for row in rows) / len(rows), 6) if rows else 0.0,
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


def run_reflection_spec(
    *,
    model_alias: str,
    dataset_spec: DatasetSpec,
    report_dir: Path,
    workers: int,
    force: bool,
    reflexion_trials: int,
    actor_temperature: float,
    reflection_temperature: float,
) -> dict[str, Any]:
    label = f"{slug_model(model_alias)}_reflection_style_{dataset_spec.name}"
    output_path = report_dir / f"{label}.json"
    progress_path = report_dir / f"{label}.progress.json"
    if output_path.exists() and not force:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            print(f"[skip] {label}: {output_path}", flush=True)
            return existing
    tasks = load_dataset(dataset_spec.dataset, dataset_spec.data_path)
    hidden_map = {problem.task_id: problem for problem in load_dataset(dataset_spec.dataset, dataset_spec.hidden_eval_path)}
    existing = load_existing_rows(progress_path, force=force)
    rows = [existing[problem.task_id] for problem in tasks if problem.task_id in existing]
    pending = [(index, problem) for index, problem in enumerate(tasks, start=1) if problem.task_id not in existing]
    parameters = {
        "reflexion_trials": reflexion_trials,
        "actor_temperature": actor_temperature,
        "reflection_temperature": reflection_temperature,
        "evaluator": "visible_tests",
        "hidden_eval": True,
    }
    metadata = {
        "label": label,
        "model_alias": model_alias,
        "method": "reflection_style",
        "dataset": dataset_spec.name,
        "size": len(tasks),
        "parameters": parameters,
    }
    started = time.perf_counter()
    write_progress(progress_path, metadata, rows)
    print(f"[run] {label}: total={len(tasks)} resumed={len(rows)} pending={len(pending)} workers={workers}", flush=True)
    if pending:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_reflection_known_task,
                    {
                        "task_index": index,
                        "key_slot": stable_key_slot(label, index),
                        "problem": problem_to_dict(problem),
                        "hidden_problem": problem_to_dict(hidden_map[problem.task_id]) if problem.task_id in hidden_map else None,
                        "dataset": dataset_spec.name,
                        "model_alias": model_alias,
                        "config_path": str(dataset_spec.remcts_config_path),
                        "reflexion_trials": reflexion_trials,
                        "actor_temperature": actor_temperature,
                        "reflection_temperature": reflection_temperature,
                    },
                )
                for index, problem in pending
            ]
            for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                if completed % max(1, min(workers, 8)) == 0 or len(rows) == len(tasks):
                    write_progress(progress_path, metadata, rows)
                    hidden = hidden_summary(rows)
                    print(f"[progress] {label}: {len(rows)}/{len(tasks)} hidden={(hidden or {}).get('hidden_solve_rate', 0.0):.4f}", flush=True)
    output = finalize_reflection_output(
        output_path=output_path,
        label=label,
        model_alias=model_alias,
        dataset_spec=dataset_spec,
        tasks=tasks,
        rows=rows,
        started=started,
        resumed_count=len(existing),
        workers=workers,
        parameters=parameters,
    )
    progress_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {label}: pass_rate={output['pass_rate']:.4f} cost_cny={output['estimated_cost_cny']:.4f}", flush=True)
    return output


def write_compact_summary(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row.get("model_alias", ""), row.get("dataset", ""), row.get("method", "")))
    lines = [
        "# Known Baseline Completion Summary",
        "",
        "- Reflection-style known uses visible tests as evaluator feedback.",
        "- MCTS-style known uses tree search with path/global memory and semantic UCB disabled.",
        "- Hidden tests are used only for final offline evaluation.",
        "",
        "| Model | Dataset | Method | Completed | Hidden Solve | Accepted | Acc. Prec. | False Acc. | Avg Iter. | Avg Tree | Calls | Total Tok. | Cost(CNY) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        hidden = row.get("hidden_summary") or {}
        usage = row.get("llm_usage", {}).get("totals", {})
        lines.append(
            "| {model} | {dataset} | {method} | {completed}/{size} | {solve:.2%} | {accepted:.2%} | "
            "{precision:.2%} | {false:.2%} | {avg_iter:.3f} | {avg_tree:.3f} | {calls} | {tokens} | {cost:.4f} |".format(
                model=row.get("model_alias"),
                dataset=row.get("dataset"),
                method=row.get("method"),
                completed=row.get("completed", 0),
                size=row.get("size", 0),
                solve=float(row.get("pass_rate") or 0.0),
                accepted=float(hidden.get("accepted_rate") or 0.0),
                precision=float(hidden.get("acceptance_precision") or 0.0),
                false=float(hidden.get("false_acceptance_rate") or 0.0),
                avg_iter=float(row.get("avg_iterations") or 0.0),
                avg_tree=float(row.get("avg_tree_size") or 0.0),
                calls=int(usage.get("calls", 0)),
                tokens=int(usage.get("total_tokens", 0)),
                cost=float(row.get("estimated_cost_cny") or 0.0),
            )
        )
    (report_dir / "known_baseline_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "known_baseline_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_summaries(
    report_dir: Path,
    started_at: str,
    *,
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> None:
    rows = load_completed_outputs(report_dir)
    summarize_all(
        report_dir,
        rows,
        started_at,
        budget_cny=budget_cny,
        initial_spent_cny=initial_spent_cny,
        stop_margin_cny=stop_margin_cny,
    )
    write_compact_summary(report_dir, rows)


def reflection_progress_line(label: str, task_count: int, row: dict[str, Any]) -> str:
    usage = row.get("llm_usage") or {}
    hidden_text = ""
    if "hidden_success" in row:
        hidden_text = (
            f", hidden={row['hidden_success']} "
            f"({row.get('hidden_passed_tests', 0)}/{row.get('hidden_total_tests', 0)})"
        )
    error_text = ", request_failed=True" if row.get("request_failed") else ""
    return (
        f"[task] {label} {row.get('task_index')}/{task_count} {row.get('task_id')}: "
        f"success={row.get('success')}, accepted={row.get('accepted')}, iter={row.get('iterations')}, "
        f"calls={usage.get('calls', 0)}, tokens={usage.get('total_tokens', 0)}"
        f"{hidden_text}{error_text}, wall={float(row.get('wall_elapsed_sec') or 0.0):.2f}s"
    )


def run_plan_global_queue(
    plan: list[tuple[str, str, str]],
    dataset_specs: dict[str, DatasetSpec],
    report_dir: Path,
    started_at: str,
    *,
    workers: int,
    force: bool,
    reflexion_trials: int,
    actor_temperature: float,
    reflection_temperature: float,
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> None:
    states: list[dict[str, Any]] = []

    for model_alias, dataset_name, method in plan:
        dataset_spec = dataset_specs[dataset_name]
        label = f"{slug_model(model_alias)}_{method}_{dataset_name}"
        output_path = report_dir / f"{label}.json"
        if output_path.exists() and not force:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                print(f"[skip] {label}: {output_path}", flush=True)
                continue

        tasks = load_dataset(dataset_spec.dataset, dataset_spec.data_path)
        hidden_map = {
            problem.task_id: problem
            for problem in load_dataset(dataset_spec.dataset, dataset_spec.hidden_eval_path)
        }
        progress_path = report_dir / f"{label}.progress.json"

        if method == "mcts_style":
            spec = ExperimentSpec(
                label=label,
                model_alias=model_alias,
                method="mcts_style",
                dataset_spec=dataset_spec,
            )
            rows = [] if force else _load_progress(progress_path, tasks)
            completed_ids = {str(row.get("task_id")) for row in rows}
            pending = deque(
                (index, problem)
                for index, problem in enumerate(tasks, start=1)
                if problem.task_id not in completed_ids
            )
            _write_progress(progress_path, spec, len(tasks), rows)
            metadata = None
        else:
            spec = None
            existing = load_existing_rows(progress_path, force=force)
            rows = [existing[problem.task_id] for problem in tasks if problem.task_id in existing]
            pending = deque(
                (index, problem)
                for index, problem in enumerate(tasks, start=1)
                if problem.task_id not in existing
            )
            metadata = {
                "label": label,
                "model_alias": model_alias,
                "method": "reflection_style",
                "dataset": dataset_spec.name,
                "size": len(tasks),
                "parameters": {
                    "reflexion_trials": reflexion_trials,
                    "actor_temperature": actor_temperature,
                    "reflection_temperature": reflection_temperature,
                    "evaluator": "visible_tests",
                    "hidden_eval": True,
                },
            }
            write_progress(progress_path, metadata, rows)

        print(
            f"[start] {label}: model={model_alias}, method={method}, size={len(tasks)}, "
            f"pending={len(pending)}, resumed={len(rows)}, workers={workers}, scheduler=global_queue",
            flush=True,
        )
        states.append(
            {
                "label": label,
                "method": method,
                "model_alias": model_alias,
                "dataset_spec": dataset_spec,
                "spec": spec,
                "metadata": metadata,
                "tasks": tasks,
                "hidden_map": hidden_map,
                "rows": rows,
                "pending": pending,
                "progress_path": progress_path,
                "output_path": output_path,
                "started": time.perf_counter(),
                "resumed_count": len(rows),
                "finalized": False,
            }
        )

    def make_payload(state: dict[str, Any], index: int, problem: Any) -> dict[str, Any]:
        label = state["label"]
        hidden_problem = state["hidden_map"].get(problem.task_id)
        if state["method"] == "mcts_style":
            spec = state["spec"]
            return {
                "label": label,
                "model_alias": state["model_alias"],
                "method": "mcts_style",
                "task_index": index,
                "key_slot": stable_key_slot(label, index),
                "task_count": len(state["tasks"]),
                "config_path": str(spec.dataset_spec.remcts_config_path),
                "problem": problem.to_dict(),
                "hidden_problem": hidden_problem.to_dict() if hidden_problem is not None else None,
            }
        return {
            "task_index": index,
            "key_slot": stable_key_slot(label, index),
            "problem": problem_to_dict(problem),
            "hidden_problem": problem_to_dict(hidden_problem) if hidden_problem is not None else None,
            "dataset": state["dataset_spec"].name,
            "model_alias": state["model_alias"],
            "config_path": str(state["dataset_spec"].remcts_config_path),
            "reflexion_trials": reflexion_trials,
            "actor_temperature": actor_temperature,
            "reflection_temperature": reflection_temperature,
        }

    def finalize_state(state: dict[str, Any]) -> dict[str, Any]:
        if state["finalized"]:
            return json.loads(state["output_path"].read_text(encoding="utf-8"))
        if state["method"] == "mcts_style":
            for row in state["rows"]:
                row.pop("progress_line", None)
            output = _finalize_spec_output(
                state["spec"],
                state["output_path"],
                state["tasks"],
                state["rows"],
                state["started"],
                resumed_count=state["resumed_count"],
                workers=workers,
            )
            print(f"[done] {state['label']}: {state['output_path']}", flush=True)
        else:
            output = finalize_reflection_output(
                output_path=state["output_path"],
                label=state["label"],
                model_alias=state["model_alias"],
                dataset_spec=state["dataset_spec"],
                tasks=state["tasks"],
                rows=state["rows"],
                started=state["started"],
                resumed_count=state["resumed_count"],
                workers=workers,
                parameters=state["metadata"]["parameters"],
            )
            state["progress_path"].write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[done] {state['label']}: pass_rate={output['pass_rate']:.4f} cost_cny={output['estimated_cost_cny']:.4f}", flush=True)
        state["finalized"] = True
        refresh_summaries(
            report_dir,
            started_at,
            budget_cny=budget_cny,
            initial_spent_cny=initial_spent_cny,
            stop_margin_cny=stop_margin_cny,
        )
        return output

    def next_state_with_pending(cursor: int) -> tuple[int, dict[str, Any] | None]:
        if not states:
            return cursor, None
        for offset in range(len(states)):
            idx = (cursor + offset) % len(states)
            state = states[idx]
            if state["pending"]:
                return idx + 1, state
        return cursor, None

    for state in states:
        if not state["pending"] and len(state["rows"]) == len(state["tasks"]):
            finalize_state(state)

    active: dict[concurrent.futures.Future, tuple[dict[str, Any], int, Any]] = {}
    cursor = 0
    workers = max(1, int(workers))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        def fill_slots() -> None:
            nonlocal cursor
            while len(active) < workers:
                cursor, state = next_state_with_pending(cursor)
                if state is None:
                    return
                index, problem = state["pending"].popleft()
                worker = _run_task_worker if state["method"] == "mcts_style" else run_reflection_known_task
                active[executor.submit(worker, make_payload(state, index, problem))] = (state, index, problem)

        fill_slots()
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                state, index, problem = active.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        "task_index": index,
                        "task_id": problem.task_id,
                        "dataset": state["dataset_spec"].name,
                        "model_alias": state["model_alias"],
                        "method": state["method"],
                        "success": False,
                        "accepted": False,
                        "needs_review": True,
                        "iterations": 0,
                        "tree_size": 0,
                        "stop_reason": "worker_failed",
                        "request_failed": True,
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                        "selected_code": "",
                        "wall_elapsed_sec": 0.0,
                        "llm_usage": {},
                    }
                state["rows"].append(row)
                if state["method"] == "mcts_style":
                    _write_progress(state["progress_path"], state["spec"], len(state["tasks"]), state["rows"])
                    line = row.pop("progress_line", None)
                    if line:
                        print(line, flush=True)
                else:
                    write_progress(state["progress_path"], state["metadata"], state["rows"])
                    print(reflection_progress_line(state["label"], len(state["tasks"]), row), flush=True)
                if len(state["rows"]) == len(state["tasks"]):
                    finalize_state(state)
            fill_slots()

    for state in states:
        if not state["finalized"] and len(state["rows"]) == len(state["tasks"]):
            finalize_state(state)
    refresh_summaries(
        report_dir,
        started_at,
        budget_cny=budget_cny,
        initial_spent_cny=initial_spent_cny,
        stop_margin_cny=stop_margin_cny,
    )


def main() -> None:
    args = parse_args()
    started_at = utc_stamp()
    report_dir = make_report_dir(args)
    configure_key_rotation(args)
    print_key_pool_summary(args)
    mcts_config = write_mcts_style_known_config(report_dir, args.credentials_file)
    dataset_specs = build_dataset_specs(report_dir, args, mcts_config)

    plan: list[tuple[str, str, str]] = []
    for model in args.models:
        for dataset_name in args.datasets:
            for method in args.methods:
                plan.append((model, dataset_name, method))
    if args.dry_run:
        print(f"report_dir={report_dir}", flush=True)
        for model, dataset_name, method in plan:
            print(f"{slug_model(model)}_{method}_{dataset_name}", flush=True)
        return

    if args.global_queue:
        run_plan_global_queue(
            plan,
            dataset_specs,
            report_dir,
            started_at,
            workers=max(1, int(args.workers)),
            force=bool(args.force),
            reflexion_trials=int(args.reflexion_trials),
            actor_temperature=float(args.actor_temperature),
            reflection_temperature=float(args.reflection_temperature),
            budget_cny=float(args.budget_cny),
            initial_spent_cny=float(args.initial_spent_cny),
            stop_margin_cny=float(args.stop_margin_cny),
        )
    else:
        for model, dataset_name, method in plan:
            dataset_spec = dataset_specs[dataset_name]
            if method == "mcts_style":
                row = run_one_spec(
                    ExperimentSpec(
                        label=f"{slug_model(model)}_mcts_style_{dataset_name}",
                        model_alias=model,
                        method="mcts_style",
                        dataset_spec=dataset_spec,
                    ),
                    report_dir=report_dir,
                    force=bool(args.force),
                    workers=max(1, int(args.workers)),
                    budget_cny=float(args.budget_cny),
                    initial_spent_cny=float(args.initial_spent_cny),
                    stop_margin_cny=float(args.stop_margin_cny),
                )
            else:
                row = run_reflection_spec(
                    model_alias=model,
                    dataset_spec=dataset_spec,
                    report_dir=report_dir,
                    workers=max(1, int(args.workers)),
                    force=bool(args.force),
                    reflexion_trials=int(args.reflexion_trials),
                    actor_temperature=float(args.actor_temperature),
                    reflection_temperature=float(args.reflection_temperature),
                )
            rows = load_completed_outputs(report_dir)
            if row.get("status") == "complete" and all(existing.get("label") != row["label"] for existing in rows):
                rows.append(row)
            summarize_all(
                report_dir,
                rows,
                started_at,
                budget_cny=float(args.budget_cny),
                initial_spent_cny=float(args.initial_spent_cny),
                stop_margin_cny=float(args.stop_margin_cny),
            )
            write_compact_summary(report_dir, rows)
    print(f"[summary] {report_dir / 'known_baseline_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
