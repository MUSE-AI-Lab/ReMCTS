#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from collections import deque
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from api_key_pool import load_key_pool
from run_qwen_multi_model_experiment import (
    DatasetSpec,
    ExperimentSpec,
    ROOT,
    _finalize_spec_output,
    _load_progress,
    _run_task_worker,
    _write_progress,
    budget_exceeded,
    load_completed_outputs,
    load_dataset,
    run_one_spec,
    slug_model,
    stable_key_slot,
    summarize_all,
    utc_stamp,
)


BASE_CONFIGS = {
    "known": ROOT / "src/remcts/config/qwen_full_k3.yaml",
    "no_tests": ROOT / "src/remcts/config/qwen_production_full_k3.yaml",
}


DATASET_DEFS = {
    "humaneval_full_known": {
        "setting": "known",
        "dataset": "humaneval",
        "data_path": ROOT / "data/humaneval/HumanEval.jsonl",
        "hidden_eval_path": ROOT / "data/humaneval/HumanEval_eval.jsonl",
    },
    "mbpp_sanitized_full_known": {
        "setting": "known",
        "dataset": "mbpp",
        "data_path": ROOT / "data/mbpp/mbpp_sanitized.json",
        "hidden_eval_path": ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
    },
    "humaneval_full_no_tests": {
        "setting": "no_tests",
        "dataset": "humaneval",
        "data_path": ROOT / "data/humaneval/HumanEval_no_tests.jsonl",
        "hidden_eval_path": ROOT / "data/humaneval/HumanEval_eval.jsonl",
    },
    "mbpp_sanitized_full_no_tests": {
        "setting": "no_tests",
        "dataset": "mbpp",
        "data_path": ROOT / "data/mbpp/mbpp_sanitized_no_tests.json",
        "hidden_eval_path": ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
    },
}


ABLATIONS = {
    "no_global_memory": {"path_memory_k": 3, "global_memory_top_t": 0},
    "no_path_memory": {"path_memory_k": 0, "global_memory_top_t": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReMCTS path/global memory ablation experiments.")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b", "deepseek-v4-flash"])
    parser.add_argument("--settings", nargs="+", default=["no_tests"], choices=["known", "no_tests"])
    parser.add_argument("--ablations", nargs="+", default=["no_global_memory", "no_path_memory"], choices=sorted(ABLATIONS))
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget-cny", type=float, default=100.0)
    parser.add_argument("--initial-spent-cny", type=float, default=0.0)
    parser.add_argument("--stop-margin-cny", type=float, default=5.0)
    parser.add_argument("--credentials-file", type=Path, default=ROOT / "apikeys.txt")
    parser.add_argument("--key-rotation", default="round_robin", choices=["round_robin", "disabled"])
    parser.add_argument("--max-key-switches", type=int, default=4)
    parser.add_argument("--global-queue", action=argparse.BooleanOptionalAction, default=True)
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
    report_dir = args.report_dir or ROOT / "outputs/reports" / f"memory-ablation-{utc_stamp()}"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "configs").mkdir(exist_ok=True)
    if args.limit is not None:
        (report_dir / "data").mkdir(exist_ok=True)
    return report_dir


def write_ablation_config(report_dir: Path, setting: str, ablation: str, credentials_file: Path) -> Path:
    base = yaml.safe_load(BASE_CONFIGS[setting].read_text(encoding="utf-8"))
    base = dict(base or {})
    search = dict(base.get("search") or {})
    search.update(ABLATIONS[ablation])
    base["search"] = search
    llm = dict(base.get("llm") or {})
    llm["credentials_file"] = str(credentials_file)
    llm["enable_thinking"] = False
    llm["max_retries"] = max(int(llm.get("max_retries", 2)), 5)
    base["llm"] = llm
    path = report_dir / "configs" / f"qwen_{setting}_{ablation}.yaml"
    path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
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


def build_dataset_specs(report_dir: Path, args: argparse.Namespace, configs: dict[tuple[str, str], Path]) -> dict[tuple[str, str], DatasetSpec]:
    specs: dict[tuple[str, str], DatasetSpec] = {}
    wanted_settings = set(args.settings)
    for dataset_name, info in DATASET_DEFS.items():
        setting = str(info["setting"])
        if setting not in wanted_settings:
            continue
        for ablation in args.ablations:
            data_path = subset_data_path(report_dir, dataset_name, Path(info["data_path"]), args.limit)
            specs[(dataset_name, ablation)] = DatasetSpec(
                name=dataset_name,
                dataset=str(info["dataset"]),
                data_path=data_path,
                remcts_config_path=configs[(setting, ablation)],
                hidden_eval_path=Path(info["hidden_eval_path"]),
            )
    return specs


def build_experiment_specs(args: argparse.Namespace, dataset_specs: dict[tuple[str, str], DatasetSpec]) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    for model in args.models:
        for (dataset_name, ablation), dataset_spec in sorted(dataset_specs.items()):
            specs.append(
                ExperimentSpec(
                    label=f"{slug_model(model)}_{ablation}_remcts_{dataset_name}",
                    model_alias=model,
                    method=f"remcts_{ablation}",
                    dataset_spec=dataset_spec,
                )
            )
    return specs


def write_compact_summary(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row.get("model_alias", ""), row.get("dataset", ""), row.get("method", "")))
    lines = [
        "# Memory Ablation Summary",
        "",
        "- Methods: `remcts_no_global_memory` sets `global_memory_top_t=0`; `remcts_no_path_memory` sets `path_memory_k=0`.",
        "- Hidden tests are used only for final offline evaluation.",
        "",
        "| Model | Dataset | Method | Completed | Hidden Solve | Accepted | Acc. Prec. | False Acc. | Avg Iter. | Avg Tree | Calls | Total Tok. | Cost(CNY) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        hidden = row.get("hidden_summary") or {}
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
                calls=int(row.get("calls") or (row.get("llm_usage", {}).get("totals", {}).get("calls", 0))),
                tokens=int(row.get("total_tokens") or (row.get("llm_usage", {}).get("totals", {}).get("total_tokens", 0))),
                cost=float(row.get("estimated_cost_cny") or 0.0),
            )
        )
    (report_dir / "memory_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "memory_ablation_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


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


def run_specs_global_queue(
    specs: list[ExperimentSpec],
    report_dir: Path,
    started_at: str,
    *,
    force: bool,
    workers: int,
    budget_cny: float,
    initial_spent_cny: float,
    stop_margin_cny: float,
) -> list[dict[str, Any]]:
    states: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []

    for spec in specs:
        output_path = report_dir / f"{spec.label}.json"
        if output_path.exists() and not force:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if existing.get("status") == "complete":
                print(f"[skip] {spec.label}: {output_path}", flush=True)
                completed_rows.append(existing)
                continue

        tasks = load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.data_path)
        hidden_tasks = (
            {problem.task_id: problem for problem in load_dataset(spec.dataset_spec.dataset, spec.dataset_spec.hidden_eval_path)}
            if spec.dataset_spec.hidden_eval_path
            else {}
        )
        progress_path = report_dir / f"{spec.label}.progress.json"
        task_rows = [] if force else _load_progress(progress_path, tasks)
        completed_task_ids = {str(row.get("task_id")) for row in task_rows}
        pending = deque(
            (index, problem)
            for index, problem in enumerate(tasks, start=1)
            if problem.task_id not in completed_task_ids
        )

        if pending:
            existing_cost = sum(float(row.get("estimated_cost_cny") or 0.0) for row in completed_rows)
            next_estimate = 0.0
            projected = initial_spent_cny + existing_cost + next_estimate
            if budget_cny > 0 and projected > budget_cny - stop_margin_cny:
                raise RuntimeError(
                    f"Budget guard stopped before {spec.label}: initial={initial_spent_cny:.2f}, "
                    f"completed_report={existing_cost:.2f}, budget={budget_cny:.2f}, "
                    f"margin={stop_margin_cny:.2f}."
                )

        print(
            f"[start] {spec.label}: model={spec.model_alias}, method={spec.method}, size={len(tasks)}, "
            f"pending={len(pending)}, resumed={len(task_rows)}, workers={workers}, "
            f"hidden_eval={bool(hidden_tasks)}, scheduler=global_queue",
            flush=True,
        )
        _write_progress(progress_path, spec, len(tasks), task_rows)

        state = {
            "spec": spec,
            "output_path": output_path,
            "progress_path": progress_path,
            "tasks": tasks,
            "hidden_tasks": hidden_tasks,
            "rows": task_rows,
            "pending": pending,
            "started": time.perf_counter(),
            "resumed_count": len(completed_task_ids),
            "finalized": False,
        }
        states.append(state)

    def make_payload(state: dict[str, Any], index: int, problem: Any) -> dict[str, Any]:
        spec = state["spec"]
        hidden_problem = state["hidden_tasks"].get(problem.task_id)
        return {
            "label": spec.label,
            "model_alias": spec.model_alias,
            "method": spec.method,
            "task_index": index,
            "key_slot": stable_key_slot(spec.label, index),
            "task_count": len(state["tasks"]),
            "config_path": str(spec.dataset_spec.remcts_config_path),
            "problem": problem.to_dict(),
            "hidden_problem": hidden_problem.to_dict() if hidden_problem is not None else None,
        }

    def finalize_state(state: dict[str, Any]) -> dict[str, Any]:
        if state["finalized"]:
            return json.loads(state["output_path"].read_text(encoding="utf-8"))
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
        state["finalized"] = True
        print(f"[done] {state['spec'].label}: {state['output_path']}", flush=True)
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
            completed_rows.append(finalize_state(state))

    workers = max(1, int(workers))
    active: dict[concurrent.futures.Future, tuple[dict[str, Any], int, Any]] = {}
    cursor = 0
    budget_hit = False

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        def fill_slots() -> None:
            nonlocal cursor
            while len(active) < workers and not budget_hit:
                cursor, state = next_state_with_pending(cursor)
                if state is None:
                    return
                index, problem = state["pending"].popleft()
                future = executor.submit(_run_task_worker, make_payload(state, index, problem))
                active[future] = (state, index, problem)

        fill_slots()
        while active:
            done, _ = concurrent.futures.wait(active, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                state, index, problem = active.pop(future)
                try:
                    task_row = future.result()
                except Exception as exc:
                    task_row = {
                        "task_index": index,
                        "task_id": problem.task_id,
                        "model_alias": state["spec"].model_alias,
                        "method": state["spec"].method,
                        "success": False,
                        "accepted": False,
                        "needs_review": True,
                        "iterations": 0,
                        "tree_size": 0,
                        "stop_reason": "worker_failed",
                        "request_failed": True,
                        "error": f"{type(exc).__name__}: {exc}"[:1000],
                        "wall_elapsed_sec": 0.0,
                        "llm_usage": {},
                    }
                state["rows"].append(task_row)
                _write_progress(state["progress_path"], state["spec"], len(state["tasks"]), state["rows"])
                line = task_row.pop("progress_line", None)
                if line:
                    print(line, flush=True)
                if budget_exceeded(
                    report_dir,
                    state["spec"],
                    state["rows"],
                    budget_cny,
                    initial_spent_cny,
                    stop_margin_cny,
                ):
                    budget_hit = True
                if len(state["rows"]) == len(state["tasks"]):
                    completed_rows.append(finalize_state(state))
            fill_slots()

        if budget_hit:
            raise RuntimeError(f"Budget guard stopped during global queue. Progress saved to {report_dir}.")

    for state in states:
        if not state["finalized"] and len(state["rows"]) == len(state["tasks"]):
            completed_rows.append(finalize_state(state))

    refresh_summaries(
        report_dir,
        started_at,
        budget_cny=budget_cny,
        initial_spent_cny=initial_spent_cny,
        stop_margin_cny=stop_margin_cny,
    )
    return completed_rows


def main() -> None:
    args = parse_args()
    started_at = utc_stamp()
    report_dir = make_report_dir(args)
    configure_key_rotation(args)
    print_key_pool_summary(args)

    configs = {
        (setting, ablation): write_ablation_config(report_dir, setting, ablation, args.credentials_file)
        for setting in args.settings
        for ablation in args.ablations
    }
    dataset_specs = build_dataset_specs(report_dir, args, configs)
    specs = build_experiment_specs(args, dataset_specs)

    if args.dry_run:
        print(f"report_dir={report_dir}", flush=True)
        for spec in specs:
            print(f"{spec.label}: model={spec.model_alias} method={spec.method} dataset={spec.dataset_spec.name} config={spec.dataset_spec.remcts_config_path}", flush=True)
        return

    if args.global_queue:
        run_specs_global_queue(
            specs,
            report_dir,
            started_at,
            force=bool(args.force),
            workers=max(1, int(args.workers)),
            budget_cny=float(args.budget_cny),
            initial_spent_cny=float(args.initial_spent_cny),
            stop_margin_cny=float(args.stop_margin_cny),
        )
    else:
        for spec in specs:
            row = run_one_spec(
                spec,
                report_dir=report_dir,
                force=bool(args.force),
                workers=max(1, int(args.workers)),
                budget_cny=float(args.budget_cny),
                initial_spent_cny=float(args.initial_spent_cny),
                stop_margin_cny=float(args.stop_margin_cny),
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
    print(f"[summary] {report_dir / 'memory_ablation_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
