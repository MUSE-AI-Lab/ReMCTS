#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from remcts.tasks.humaneval import HumanEvalDataset
from remcts.tasks.mbpp import MBPPDataset


ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICE_CNY_PER_MILLION = 0.3
OUTPUT_PRICE_CNY_PER_MILLION = 1.2


@dataclass(frozen=True)
class RunSpec:
    label: str
    dataset: str
    data_path: Path
    config_path: Path
    hidden_eval_path: Path | None = None


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_dataset(dataset: str, path: Path) -> list[ProblemSpec]:
    if dataset == "humaneval":
        return list(HumanEvalDataset(str(path)).iter_tasks())
    if dataset == "mbpp":
        return list(MBPPDataset(str(path)).iter_tasks())
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_config(path: Path, api_alias: str) -> SearchConfig:
    config = SearchConfig.from_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))
    config.llm_api_alias = api_alias
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
        model = str(usage.get("model") or "unknown")
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


def estimate_cost_cny(prompt_tokens: int, completion_tokens: int) -> float:
    return round(
        prompt_tokens * INPUT_PRICE_CNY_PER_MILLION / 1_000_000
        + completion_tokens * OUTPUT_PRICE_CNY_PER_MILLION / 1_000_000,
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


def _problem_from_dict(data: dict[str, Any]) -> ProblemSpec:
    return ProblemSpec(
        task_id=str(data.get("task_id", "unknown")),
        prompt=str(data.get("prompt") or ""),
        entry_point=data.get("entry_point"),
        tests=str(data.get("tests") or ""),
        metadata=dict(data.get("metadata") or {}),
    )


def _run_task_worker(payload: dict[str, Any]) -> dict[str, Any]:
    spec_label = str(payload["spec_label"])
    task_index = int(payload["task_index"])
    task_count = int(payload["task_count"])
    config = load_config(Path(payload["config_path"]), str(payload["api_alias"]))
    problem = _problem_from_dict(payload["problem"])
    hidden_problem = (
        _problem_from_dict(payload["hidden_problem"])
        if payload.get("hidden_problem") is not None
        else None
    )
    engine = SearchEngine(config)
    hidden_executor = PythonExecutor()

    task_started = time.perf_counter()
    result = engine.run(problem)
    task_elapsed = round(time.perf_counter() - task_started, 6)
    usage = compact_usage(result.summary.get("llm_usage"))
    task_row: dict[str, Any] = {
        "task_index": task_index,
        "task_id": problem.task_id,
        "success": bool(result.success),
        "accepted": bool(result.accepted),
        "needs_review": bool(result.needs_review),
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "tree_size": result.tree_size,
        "run_dir": result.run_dir,
        "wall_elapsed_sec": task_elapsed,
        "llm_usage": usage,
    }
    if hidden_problem is not None:
        hidden_result = run_problem_tests(
            hidden_problem,
            result.best_code,
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
    hidden_text = ""
    if "hidden_success" in task_row:
        hidden_text = (
            f", hidden={task_row['hidden_success']} "
            f"({task_row['hidden_passed_tests']}/{task_row['hidden_total_tests']})"
        )
    task_row["progress_line"] = (
        f"[task] {spec_label} {task_index}/{task_count} {problem.task_id}: "
        f"success={result.success}, accepted={result.accepted}, iter={result.iterations}, "
        f"calls={usage['calls']}, tokens={usage['total_tokens']}{hidden_text}, "
        f"wall={task_elapsed:.2f}s"
    )
    return task_row


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
        task_id = str(row.get("task_id") or "")
        if task_id not in order:
            continue
        row["task_index"] = int(row.get("task_index") or order[task_id])
        dedup[task_id] = row
    return sorted(dedup.values(), key=lambda item: int(item.get("task_index") or 0))


def _write_progress(progress_path: Path, label: str, task_count: int, task_rows: list[dict[str, Any]]) -> None:
    ordered_rows = sorted(task_rows, key=lambda item: int(item.get("task_index") or 0))
    progress = {
        "label": label,
        "completed": len(ordered_rows),
        "size": task_count,
        "tasks": ordered_rows,
    }
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def _finalize_spec_output(
    spec: RunSpec,
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
        round(sum(1 for row in task_rows if row["success"]) / len(task_rows), 6)
        if task_rows
        else 0.0
    )
    output = {
        "label": spec.label,
        "dataset": spec.dataset,
        "data_path": str(spec.data_path),
        "config_path": str(spec.config_path),
        "hidden_eval_path": str(spec.hidden_eval_path) if spec.hidden_eval_path else None,
        "size": len(tasks),
        "pass_rate": hidden["hidden_solve_rate"] if hidden else raw_search_success_rate,
        "pass_rate_metric": "hidden_solve_rate" if hidden else "explicit_test_pass_rate",
        "raw_search_success_rate": raw_search_success_rate,
        "avg_iterations": round(sum(row["iterations"] for row in task_rows) / len(task_rows), 6) if task_rows else 0.0,
        "avg_tree_size": round(sum(row["tree_size"] for row in task_rows) / len(task_rows), 6) if task_rows else 0.0,
        "wall_elapsed_sec": wall_elapsed,
        "summed_task_wall_elapsed_sec": summed_task_wall,
        "resumed_task_count": resumed_count,
        "workers": workers,
        "llm_usage": usage_summary,
        "estimated_cost_cny": estimate_cost_cny(totals["prompt_tokens"], totals["completion_tokens"]),
        "hidden_summary": hidden,
        "tasks": task_rows,
    }
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def run_one_spec(spec: RunSpec, report_dir: Path, api_alias: str, force: bool, workers: int) -> dict[str, Any]:
    output_path = report_dir / f"{spec.label}.json"
    if output_path.exists() and not force:
        print(f"[skip] {spec.label}: {output_path}", flush=True)
        return json.loads(output_path.read_text(encoding="utf-8"))

    tasks = load_dataset(spec.dataset, spec.data_path)
    hidden_tasks = (
        {problem.task_id: problem for problem in load_dataset(spec.dataset, spec.hidden_eval_path)}
        if spec.hidden_eval_path
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

    print(
        f"[start] {spec.label}: size={len(tasks)}, pending={len(pending_tasks)}, "
        f"resumed={len(task_rows)}, workers={workers}, config={spec.config_path}, "
        f"hidden_eval={bool(hidden_tasks)}",
        flush=True,
    )
    started = time.perf_counter()

    def make_payload(index: int, problem: ProblemSpec) -> dict[str, Any]:
        hidden_problem = hidden_tasks.get(problem.task_id)
        return {
            "spec_label": spec.label,
            "task_index": index,
            "task_count": len(tasks),
            "config_path": str(spec.config_path),
            "api_alias": api_alias,
            "problem": problem.to_dict(),
            "hidden_problem": hidden_problem.to_dict() if hidden_problem is not None else None,
        }

    try:
        if workers <= 1:
            for index, problem in pending_tasks:
                task_row = _run_task_worker(make_payload(index, problem))
                task_rows.append(task_row)
                _write_progress(progress_path, spec.label, len(tasks), task_rows)
                print(task_row.pop("progress_line"), flush=True)
        elif pending_tasks:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(_run_task_worker, make_payload(index, problem)) for index, problem in pending_tasks]
                for future in as_completed(futures):
                    task_row = future.result()
                    task_rows.append(task_row)
                    _write_progress(progress_path, spec.label, len(tasks), task_rows)
                    print(task_row.pop("progress_line"), flush=True)
        else:
            _write_progress(progress_path, spec.label, len(tasks), task_rows)
    finally:
        for row in task_rows:
            row.pop("progress_line", None)

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


def build_specs() -> list[RunSpec]:
    return [
        RunSpec(
            label="humaneval_full_known",
            dataset="humaneval",
            data_path=ROOT / "data/humaneval/HumanEval.jsonl",
            config_path=ROOT / "src/remcts/config/qwen_full_k3.yaml",
        ),
        RunSpec(
            label="humaneval_full_no_tests",
            dataset="humaneval",
            data_path=ROOT / "data/humaneval/HumanEval_no_tests.jsonl",
            config_path=ROOT / "src/remcts/config/qwen_production_full_k3.yaml",
            hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
        ),
        RunSpec(
            label="mbpp_sanitized_full_known",
            dataset="mbpp",
            data_path=ROOT / "data/mbpp/mbpp_sanitized.json",
            config_path=ROOT / "src/remcts/config/qwen_full_k3.yaml",
        ),
        RunSpec(
            label="mbpp_sanitized_full_no_tests",
            dataset="mbpp",
            data_path=ROOT / "data/mbpp/mbpp_sanitized_no_tests.json",
            config_path=ROOT / "src/remcts/config/qwen_production_full_k3.yaml",
            hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
        ),
    ]


def summarize_all(report_dir: Path, api_alias: str, rows: list[dict[str, Any]], started_at: str) -> dict[str, Any]:
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
        cost = row["estimated_cost_cny"]
        hidden = row.get("hidden_summary")
        raw_search_success_rate = row.get("raw_search_success_rate", row["pass_rate"])
        displayed_pass_rate = hidden["hidden_solve_rate"] if hidden else row["pass_rate"]
        pass_rate_metric = "hidden_solve_rate" if hidden else row.get("pass_rate_metric", "explicit_test_pass_rate")
        table_row = {
            "label": row["label"],
            "file": str(report_dir / f"{row['label']}.json"),
            "size": row["size"],
            "pass_rate": displayed_pass_rate,
            "pass_rate_metric": pass_rate_metric,
            "raw_search_success_rate": raw_search_success_rate,
            "avg_iterations": row["avg_iterations"],
            "avg_tree_size": row["avg_tree_size"],
            "wall_elapsed_sec": row["wall_elapsed_sec"],
            "summed_task_wall_elapsed_sec": row.get("summed_task_wall_elapsed_sec", row["wall_elapsed_sec"]),
            "resumed_task_count": row.get("resumed_task_count", 0),
            "workers": row.get("workers", 1),
            "calls": usage_totals["calls"],
            "successful_calls": usage_totals["successful_calls"],
            "failed_calls": usage_totals["failed_calls"],
            "prompt_tokens": usage_totals["prompt_tokens"],
            "completion_tokens": usage_totals["completion_tokens"],
            "total_tokens": usage_totals["total_tokens"],
            "llm_elapsed_sec": usage_totals["elapsed_sec"],
            "estimated_cost_cny": cost,
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
        totals["estimated_cost_cny"] += cost
    totals["llm_elapsed_sec"] = round(totals["llm_elapsed_sec"], 6)
    totals["wall_elapsed_sec"] = round(totals["wall_elapsed_sec"], 6)
    totals["estimated_cost_cny"] = round(totals["estimated_cost_cny"], 6)

    summary = {
        "report_dir": str(report_dir),
        "model": api_alias,
        "stage": "qwen3-1.7b full LATS-like ReMCTS k=3",
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
        },
        "price_assumption_cny_per_million": {
            "input": INPUT_PRICE_CNY_PER_MILLION,
            "output": OUTPUT_PRICE_CNY_PER_MILLION,
        },
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
    lines.append("# qwen3-1.7b Full LATS-like ReMCTS k=3 Summary")
    lines.append("")
    lines.append(f"- Report dir: `{summary['report_dir']}`")
    lines.append(f"- Model alias: `{summary['model']}`")
    lines.append("- Budget: `max_iterations=8`, `branch_factor=5`, candidate budget 40")
    lines.append("- ReMCTS memory: `path_memory_k=3`, `global_memory_top_t=2`")
    lines.append("- LATS alignment: 8 iterations and 5 expansion samples per iteration")
    lines.append("- Price assumption: input CNY 0.3 / 1M tokens, output CNY 1.2 / 1M tokens")
    lines.append("- Note: HumanEval now explicitly calls `check(entry_point)` during evaluation.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append(
        "| Run | Size | Workers | Resumed | Solve Rate | Rate Metric | Avg Iter | Avg Tree | Wall Time | Task Time Sum | Calls | Input Tok | Output Tok | Total Tok | Est. CNY |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["rows"]:
        rate_metric_label = (
            "hidden solve"
            if row.get("pass_rate_metric") == "hidden_solve_rate"
            else "explicit tests"
        )
        display_row = dict(row, rate_metric_label=rate_metric_label)
        lines.append(
            "| {label} | {size} | {workers} | {resumed_task_count} | {pass_rate:.4f} | {rate_metric_label} | "
            "{avg_iterations:.4f} | {avg_tree_size:.2f} | {wall_elapsed_sec:.2f}s | "
            "{summed_task_wall_elapsed_sec:.2f}s | {calls} | {prompt_tokens} | {completion_tokens} | "
            "{total_tokens} | {estimated_cost_cny:.6f} |".format(**display_row)
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
    lines.append("")
    lines.append("## Production Hidden Eval")
    lines.append("")
    lines.append(
        "| Run | Tasks | Hidden Solve | Accepted | Needs Review | Acceptance Precision | False Acceptance |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary["rows"]:
        hidden = row.get("hidden_summary")
        if not hidden:
            continue
        lines.append(
            "| {label} | {tasks} | {hidden_success}/{tasks} ({hidden_rate}) | "
            "{accepted}/{tasks} ({accepted_rate}) | {needs_review}/{tasks} ({review_rate}) | "
            "{precision} | {false_rate} |".format(
                label=row["label"],
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
    lines.append("")
    lines.append("## Analysis")
    lines.append("")
    known_rows = [row for row in summary["rows"] if row["label"].endswith("_known")]
    no_test_rows = [row for row in summary["rows"] if row["label"].endswith("_no_tests")]
    if known_rows:
        best_known = max(known_rows, key=lambda row: row["pass_rate"])
        lines.append(
            f"- Known/oracle best solve rate: `{best_known['label']}` at {percent(best_known['pass_rate'])}. "
            "These runs use explicit tests during search, so they are mechanism checks rather than standard public pass@1."
        )
    if no_test_rows:
        hidden_parts = []
        for row in no_test_rows:
            hidden = row.get("hidden_summary")
            if hidden:
                hidden_parts.append(f"`{row['label']}` hidden solve {percent(hidden['hidden_solve_rate'])}")
        if hidden_parts:
            lines.append("- No-tests hidden evaluation: " + "; ".join(hidden_parts) + ".")
        for row in no_test_rows:
            hidden = row.get("hidden_summary")
            if not hidden:
                continue
            lines.append(
                f"- `{row['label']}` is a no-explicit-tests setting: internal pass rate is not meaningful during search; "
                f"hidden solve is {hidden['hidden_success']}/{hidden['tasks']} ({percent(hidden['hidden_solve_rate'])}), "
                f"accepted is {hidden['accepted']}/{hidden['tasks']} ({percent(hidden['accepted_rate'])}), "
                f"acceptance precision is {percent(hidden['acceptance_precision'])}, "
                f"and false acceptance is {percent(hidden['false_acceptance_rate'])}."
            )
    if summary["rows"]:
        max_token_row = max(summary["rows"], key=lambda row: row["total_tokens"])
        lines.append(
            f"- Token use is dominated by `{max_token_row['label']}` "
            f"({max_token_row['total_tokens']} tokens, CNY {max_token_row['estimated_cost_cny']:.6f})."
        )
    lines.append(
        "- For thesis reporting, qwen3-1.7b should be treated as a low-cost full-data pilot. "
        "Use qwen3-14b/qwen3-32b and later DeepSeek/GPT-4 runs for main conclusions."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run qwen3-1.7b full LATS-like ReMCTS k=3 experiments.")
    parser.add_argument("--api-alias", default="qwen3-1.7b")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--force", action="store_true", help="Overwrite existing run json files in the report dir.")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes for unfinished tasks.")
    parser.add_argument(
        "--only",
        action="append",
        choices=[
            "humaneval_full_known",
            "humaneval_full_no_tests",
            "mbpp_sanitized_full_known",
            "mbpp_sanitized_full_no_tests",
        ],
        help="Run only selected labels. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    started_at = utc_stamp()
    report_dir = Path(args.report_dir) if args.report_dir else ROOT / "outputs/reports" / f"qwen3-1_7b-full-lats-k3-{started_at}"
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = build_specs()
    if args.only:
        requested = set(args.only)
        specs = [spec for spec in specs if spec.label in requested]

    workers = max(1, args.workers)
    rows = [run_one_spec(spec, report_dir, args.api_alias, args.force, workers) for spec in specs]
    summary = summarize_all(report_dir, args.api_alias, rows, started_at)
    print(json.dumps({"summary": summary["report_dir"], "totals": summary["totals"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
