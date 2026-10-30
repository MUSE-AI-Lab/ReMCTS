"""Run experiment specs over a shared worker pool with resumable progress and a budget guard.

Tasks from all specs are scheduled round-robin, so a slow model or dataset does not idle the
pool. Every finished task is appended to ``<label>.progress.json``; a rerun with the same
``--report-dir`` skips complete reports and retries only missing or failed tasks.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterator

from remcts.config import load_config
from remcts.core.state import ProblemSpec
from remcts.sandbox.test_runner import executor_for, run_problem_tests

from .datasets import load_problems
from .keys import stable_key_slot
from .methods import hidden_fields
from .reporting import build_output, load_outputs, read_json, utc_stamp, write_json, write_summary
from .specs import ExperimentSpec
from .usage import compact_usage, rows_cost_cny
from .worker import TaskPayload, failed_row, run_task


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class RunOptions:
    report_dir: Path
    data_dir: Path = Path("data")
    workers: int = 4
    limit: int | None = None
    force: bool = False
    budget_cny: float = 100.0  # 0 disables the guard
    initial_spent_cny: float = 0.0
    stop_margin_cny: float = 5.0
    credentials_file: str | None = "apikeys.txt"
    key_rotation: bool = True
    max_key_switches: int = 4

    def budget(self) -> dict[str, float]:
        return {
            "budget_cny": self.budget_cny,
            "initial_spent_cny": self.initial_spent_cny,
            "stop_margin_cny": self.stop_margin_cny,
        }


@dataclass
class _Experiment:
    spec: ExperimentSpec
    tasks: list[ProblemSpec]
    hidden: dict[str, ProblemSpec]
    rows: list[dict[str, Any]]
    pending: deque[tuple[int, ProblemSpec]]
    resumed: int
    started: float = field(default_factory=time.perf_counter)
    finalized: bool = False
    cost_cny: float = 0.0

    def update_cost(self) -> None:
        self.cost_cny = rows_cost_cny(self.spec.model_alias, self.rows)


def load_tasks(spec: ExperimentSpec, options: RunOptions) -> list[ProblemSpec]:
    tasks = load_problems(spec.dataset, options.data_dir)
    return tasks[: options.limit] if options.limit is not None else tasks


def run_experiments(specs: list[ExperimentSpec], options: RunOptions) -> list[dict[str, Any]]:
    """Run ``specs`` and write per-experiment reports plus ``summary.json``/``summary.md``."""
    report_dir = options.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    started_at = utc_stamp()

    experiments = []
    for spec in specs:
        if spec.reuse_from_label:
            continue
        if not options.force and _complete_output(report_dir, spec.label):
            print(f"[skip] {spec.label}: complete", flush=True)
            continue
        experiments.append(_prepare(spec, options))

    budget_error: BudgetExceeded | None = None
    try:
        _execute(experiments, options)
    except BudgetExceeded as exc:
        budget_error = exc
    for spec in specs:
        if spec.reuse_from_label:
            _rescore_direct_generations(spec, options)
    write_summary(report_dir, load_outputs(report_dir), started_at=started_at, budget=options.budget())
    print(f"[summary] {report_dir / 'summary.md'}", flush=True)
    if budget_error is not None:
        raise budget_error
    return load_outputs(report_dir)


def _complete_output(report_dir: Path, label: str) -> dict[str, Any] | None:
    output = read_json(report_dir / f"{label}.json")
    return output if isinstance(output, dict) and output.get("status") == "complete" else None


def _progress_path(report_dir: Path, label: str) -> Path:
    return report_dir / f"{label}.progress.json"


def _prepare(spec: ExperimentSpec, options: RunOptions) -> _Experiment:
    tasks = load_tasks(spec, options)
    hidden = {problem.task_id: problem for problem in load_problems(spec.dataset, options.data_dir, hidden=True)}
    rows = [] if options.force else _load_progress(_progress_path(options.report_dir, spec.label), tasks)
    done = {row["task_id"] for row in rows}
    pending = deque((index, problem) for index, problem in enumerate(tasks, start=1) if problem.task_id not in done)
    experiment = _Experiment(spec, tasks, hidden, rows, pending, resumed=len(rows))
    experiment.update_cost()
    _write_progress(options.report_dir, experiment)
    print(
        f"[start] {spec.label}: model={spec.model_alias}, method={spec.method}, dataset={spec.dataset.name}, "
        f"size={len(tasks)}, pending={len(pending)}, resumed={len(rows)}",
        flush=True,
    )
    return experiment


def _load_progress(path: Path, tasks: list[ProblemSpec]) -> list[dict[str, Any]]:
    """Finished rows of an earlier run; failed requests are dropped so that they are retried."""
    data = read_json(path)
    order = {problem.task_id: index for index, problem in enumerate(tasks, start=1)}
    rows: dict[str, dict[str, Any]] = {}
    for row in (data or {}).get("tasks") or []:
        if isinstance(row, dict) and not row.get("request_failed") and str(row.get("task_id")) in order:
            row["task_index"] = int(row.get("task_index") or order[str(row["task_id"])])
            rows[str(row["task_id"])] = row
    return sorted(rows.values(), key=lambda row: row["task_index"])


def _write_progress(report_dir: Path, experiment: _Experiment) -> None:
    spec = experiment.spec
    write_json(
        _progress_path(report_dir, spec.label),
        {
            "label": spec.label,
            "model_alias": spec.model_alias,
            "method": spec.method,
            "dataset": spec.dataset.name,
            "completed": len(experiment.rows),
            "size": len(experiment.tasks),
            "tasks": sorted(experiment.rows, key=lambda row: int(row.get("task_index") or 0)),
        },
    )


def _payload(experiment: _Experiment, index: int, problem: ProblemSpec, options: RunOptions) -> TaskPayload:
    spec = experiment.spec
    hidden = experiment.hidden.get(problem.task_id)
    return TaskPayload(
        label=spec.label,
        method=spec.method,
        runner=spec.runner,
        model_alias=spec.model_alias,
        dataset=spec.dataset.name,
        task_index=index,
        key_slot=stable_key_slot(spec.label, index),
        config=spec.config,
        problem=problem.to_dict(),
        hidden_problem=hidden.to_dict() if hidden is not None else None,
        config_overrides=spec.config_overrides,
        params=spec.params,
        run_output_dir=str(options.report_dir / "runs"),
        credentials_file=options.credentials_file,
        key_rotation=options.key_rotation,
        max_key_switches=options.max_key_switches,
    )


def _round_robin(experiments: list[_Experiment]) -> Iterator[tuple[_Experiment, int, ProblemSpec]]:
    while True:
        progressed = False
        for experiment in experiments:
            if experiment.pending:
                index, problem = experiment.pending.popleft()
                progressed = True
                yield experiment, index, problem
        if not progressed:
            return


def _execute(experiments: list[_Experiment], options: RunOptions) -> None:
    report_dir = options.report_dir
    active_labels = {experiment.spec.label for experiment in experiments}
    earlier_cost = sum(
        float(output.get("estimated_cost_cny") or 0.0)
        for output in load_outputs(report_dir)
        if output["label"] not in active_labels
    )

    def spent() -> float:
        return options.initial_spent_cny + earlier_cost + sum(item.cost_cny for item in experiments)

    def over_budget() -> bool:
        return options.budget_cny > 0 and spent() >= options.budget_cny - options.stop_margin_cny

    def record(experiment: _Experiment, row: dict[str, Any]) -> None:
        experiment.rows.append(row)
        experiment.update_cost()
        _write_progress(report_dir, experiment)
        print(_progress_line(experiment, row), flush=True)
        if len(experiment.rows) == len(experiment.tasks):
            _finalize(experiment, options)

    for experiment in experiments:
        if not experiment.pending and len(experiment.rows) == len(experiment.tasks):
            _finalize(experiment, options)

    queue = _round_robin(experiments)
    workers = max(1, int(options.workers))
    if workers == 1:
        for experiment, index, problem in queue:
            if over_budget():
                break
            record(experiment, run_task(_payload(experiment, index, problem, options)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            active: dict[Future, tuple[_Experiment, int, ProblemSpec]] = {}

            def fill() -> None:
                while len(active) < workers and not over_budget():
                    item = next(queue, None)
                    if item is None:
                        return
                    experiment, index, problem = item
                    active[pool.submit(run_task, _payload(experiment, index, problem, options))] = item

            fill()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    experiment, index, problem = active.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:  # the worker process itself failed
                        row = _worker_failed_row(experiment, index, problem, exc)
                    record(experiment, row)
                fill()

    unfinished = [item.spec.label for item in experiments if not item.finalized]
    if unfinished and over_budget():
        raise BudgetExceeded(
            f"Budget guard stopped the run at CNY {spent():.2f} (budget {options.budget_cny:.2f}, "
            f"margin {options.stop_margin_cny:.2f}); progress is saved for: {', '.join(unfinished)}"
        )


def _worker_failed_row(experiment: _Experiment, index: int, problem: ProblemSpec, exc: Exception) -> dict[str, Any]:
    spec = experiment.spec
    return {
        "task_index": index,
        "task_id": problem.task_id,
        "dataset": spec.dataset.name,
        "model_alias": spec.model_alias,
        "method": spec.method,
        **failed_row(spec.runner, exc, stop_reason="worker_failed"),
        "wall_elapsed_sec": 0.0,
        "llm_usage": compact_usage(None),
    }


def _progress_line(experiment: _Experiment, row: dict[str, Any]) -> str:
    usage = row.get("llm_usage") or {}
    hidden = ""
    if "hidden_success" in row:
        hidden = f", hidden={row['hidden_success']} ({row['hidden_passed_tests']}/{row['hidden_total_tests']})"
    failed = ", request_failed=True" if row.get("request_failed") else ""
    return (
        f"[task] {experiment.spec.label} {row.get('task_index')}/{len(experiment.tasks)} {row.get('task_id')}: "
        f"success={row.get('success')}, accepted={row.get('accepted')}, iter={row.get('iterations')}, "
        f"calls={usage.get('calls', 0)}, tokens={usage.get('total_tokens', 0)}{hidden}{failed}, "
        f"wall={float(row.get('wall_elapsed_sec') or 0.0):.2f}s"
    )


def _finalize(experiment: _Experiment, options: RunOptions) -> None:
    output = build_output(
        experiment.spec,
        experiment.tasks,
        experiment.rows,
        wall_elapsed_sec=time.perf_counter() - experiment.started,
        resumed_count=experiment.resumed,
        workers=options.workers,
    )
    write_json(options.report_dir / f"{experiment.spec.label}.json", output)
    experiment.finalized = True
    print(
        f"[done] {experiment.spec.label}: {output['pass_rate_metric']}={output['pass_rate']:.4f}, "
        f"status={output['status']}, cost_cny={output['estimated_cost_cny']:.4f}",
        flush=True,
    )


def _rescore_direct_generations(spec: ExperimentSpec, options: RunOptions) -> None:
    """Direct no_tests: score the saved Direct known generations on the hidden split (no API calls)."""
    report_dir = options.report_dir
    if not options.force and _complete_output(report_dir, spec.label):
        print(f"[skip] {spec.label}: complete", flush=True)
        return
    source = _complete_output(report_dir, spec.reuse_from_label or "")
    if source is None:
        print(f"[skip] {spec.label}: needs a complete {spec.reuse_from_label} report first", flush=True)
        return

    started = time.perf_counter()
    config = load_config(spec.config)
    tasks = load_tasks(spec, options)
    hidden = {problem.task_id: problem for problem in load_problems(spec.dataset, options.data_dir, hidden=True)}
    source_rows = {str(row.get("task_id")): row for row in source.get("tasks") or []}
    rows = []
    for index, problem in enumerate(tasks, start=1):
        source_row = source_rows.get(problem.task_id)
        if source_row is None or not source_row.get("code"):
            raise RuntimeError(f"{spec.label}: {spec.reuse_from_label} has no saved code for {problem.task_id}")
        code = str(source_row["code"])
        row: dict[str, Any] = {
            "task_index": index,
            "task_id": problem.task_id,
            "dataset": spec.dataset.name,
            "model_alias": spec.model_alias,
            "method": spec.method,
            "success": bool(source_row.get("success")),
            "accepted": bool(source_row.get("accepted")),
            "needs_review": bool(source_row.get("needs_review")),
            "stop_reason": "direct_reused_for_hidden_eval",
            "source_stop_reason": source_row.get("stop_reason"),
            "reused_llm_from_label": spec.reuse_from_label,
            "iterations": 1,
            "tree_size": 1,
            "run_dir": None,
            "code": code,
            "llm_usage": compact_usage(None),
        }
        hidden_problem = hidden.get(problem.task_id)
        if "hidden_success" in source_row:
            row.update({key: source_row[key] for key in source_row if key.startswith("hidden_")})
        elif hidden_problem is not None:
            result = run_problem_tests(
                hidden_problem, code, executor_for(hidden_problem), config.timeout_sec, config.max_output_chars
            )
            row.update(hidden_fields(result))
        if "hidden_success" in row:
            row.update(success=bool(row["hidden_success"]), accepted=bool(row["hidden_success"]), needs_review=False)
        row["wall_elapsed_sec"] = 0.0
        rows.append(row)

    output = build_output(
        spec,
        tasks,
        rows,
        wall_elapsed_sec=time.perf_counter() - started,
        resumed_count=0,
        workers=1,
    )
    write_json(report_dir / f"{spec.label}.json", output)
    print(
        f"[done] {spec.label}: reused {spec.reuse_from_label}, hidden_solve_rate={output['pass_rate']:.4f}",
        flush=True,
    )
