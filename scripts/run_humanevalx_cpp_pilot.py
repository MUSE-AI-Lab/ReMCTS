#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.request

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from api_key_pool import ApiProfile, load_key_pool, select_profile
from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import LLMClientError, OpenAICompatibleLLMClient
from remcts.llm.response_parser import extract_code
from remcts.sandbox.cpp_executor import CppExecutor, strip_cpp_main_or_tests


DATA_URL = (
    "https://raw.githubusercontent.com/THUDM/CodeGeeX/main/"
    "codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
)
DATA_DIR = ROOT / "data" / "humaneval_x"
DATA_PATH = DATA_DIR / "humaneval_cpp.jsonl"
CONFIG_PATH = ROOT / "src/remcts/config/cpp_known.yaml"

MODEL_PRICES_CNY_PER_MILLION: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.952, 1.904),
    "qwen3-8b": (0.5, 2.0),
    "qwen3-32b": (2.0, 8.0),
}


@dataclass(frozen=True)
class Task:
    index: int
    data: dict[str, Any]
    entry_point: str


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slug_model(model_alias: str) -> str:
    return model_alias.replace(".", "_").replace("-", "_")


def ensure_dataset(force: bool = False) -> Path:
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / "humaneval_cpp.jsonl.gz"
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "remcts/0.1"})
    with urllib.request.urlopen(req, timeout=60) as response:
        gz_path.write_bytes(response.read())
    DATA_PATH.write_text(gzip.decompress(gz_path.read_bytes()).decode("utf-8"), encoding="utf-8")
    return DATA_PATH


def load_tasks(limit: int | None = None) -> list[Task]:
    path = ensure_dataset()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks: list[Task] = []
    for index, row in enumerate(rows):
        tasks.append(Task(index=index, data=row, entry_point=parse_entry_point(row)))
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    return tasks


def parse_entry_point(item: dict[str, Any]) -> str:
    declaration = str(item.get("declaration") or item.get("prompt") or "")
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", declaration)
    if matches:
        return matches[-1]
    prompt = str(item.get("prompt") or "")
    matches = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{", prompt)
    return matches[-1] if matches else "solution"


def normalize_cpp_code(raw: str, task: Task) -> str:
    code = strip_cpp_main_or_tests(extract_code(raw).strip())
    if not code:
        code = strip_cpp_main_or_tests(raw.strip())
    declaration = str(task.data.get("declaration") or task.data.get("prompt") or "")
    entry = task.entry_point
    has_entry = bool(re.search(rf"\b{re.escape(entry)}\s*\(", code))
    if not has_entry:
        code = f"{declaration.rstrip()}\n{code.strip()}"
    elif "#include" not in code:
        code = "#include <bits/stdc++.h>\nusing namespace std;\n" + code
    return code.strip() + "\n"


def build_problem(task: Task) -> ProblemSpec:
    return ProblemSpec(
        task_id=str(task.data.get("task_id") or f"CPP/{task.index}"),
        prompt=str(task.data.get("prompt") or ""),
        entry_point=task.entry_point,
        tests=str(task.data.get("example_test") or ""),
        metadata={
            "language": "cpp",
            "declaration": str(task.data.get("declaration") or ""),
            "hidden_test": str(task.data.get("test") or ""),
            "source": "HumanEval-X/CodeGeeX",
        },
    )


def build_direct_prompt(task: Task) -> str:
    return f"""
You are a careful C++17 programming assistant. Complete the following HumanEval-X C++ task.

Return only complete C++17 source code. Do not include Markdown fences or explanations.
The code must define the required entry function `{task.entry_point}`. Do not define `main` and do not include tests.

Problem and partial declaration:
{task.data.get("prompt", "")}

Required declaration / prefix:
{task.data.get("declaration", "")}
""".strip()


def make_client(profile: ApiProfile, *, timeout_sec: int, max_retries: int) -> OpenAICompatibleLLMClient:
    return OpenAICompatibleLLMClient(
        model_name=profile.model_name,
        base_url=profile.base_url,
        api_key=profile.api_key,
        api_alias=f"{profile.provider}-{profile.index}",
        enable_thinking=False,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
    )


def load_config(model_alias: str, report_dir: Path) -> SearchConfig:
    config = SearchConfig.from_mapping(yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")))
    config.llm_api_alias = model_alias
    config.llm_enable_thinking = False
    config.llm_max_retries = max(int(config.llm_max_retries), 5)
    config.run_output_dir = str(report_dir / "runs")
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


def estimate_cost_cny(model_alias: str, usage: dict[str, Any]) -> float:
    input_price, output_price = MODEL_PRICES_CNY_PER_MILLION.get(model_alias, MODEL_PRICES_CNY_PER_MILLION["qwen3-32b"])
    return round(
        int(usage.get("prompt_tokens") or 0) * input_price / 1_000_000
        + int(usage.get("completion_tokens") or 0) * output_price / 1_000_000,
        6,
    )


def run_hidden_eval(task: Task, code: str, executor: CppExecutor, timeout_sec: int, max_output_chars: int) -> dict[str, Any]:
    hidden = executor.run(
        code,
        tests=str(task.data.get("test") or ""),
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
    )
    return {
        "hidden_success": hidden.exec_ok and hidden.total_tests > 0 and hidden.passed_tests == hidden.total_tests,
        "hidden_passed_tests": hidden.passed_tests,
        "hidden_total_tests": hidden.total_tests,
        "hidden_failed_cases": hidden.failed_cases,
        "hidden_execution": hidden.to_dict(),
    }


def run_direct_task(
    task: Task,
    model_alias: str,
    pool: dict[str, list[ApiProfile]],
    report_dir: Path,
    max_key_switches: int,
) -> dict[str, Any]:
    executor = CppExecutor()
    last_error: Exception | None = None
    for attempt in range(max_key_switches):
        profile = select_profile(pool, model_alias, task.index, attempt)
        if profile is None:
            raise RuntimeError(f"No API profile available for {model_alias}")
        client = make_client(profile, timeout_sec=90, max_retries=5)
        started = time.perf_counter()
        try:
            raw = client.generate(build_direct_prompt(task), temperature=0.2, max_tokens=2048)
            code = normalize_cpp_code(raw, task)
            visible = executor.run(code, tests=str(task.data.get("example_test") or ""), timeout_sec=8, max_output_chars=4000)
            hidden = run_hidden_eval(task, code, executor, timeout_sec=8, max_output_chars=4000)
            usage = compact_usage(client.usage_snapshot())
            return {
                "task_index": task.index,
                "task_id": task.data.get("task_id"),
                "method": "direct",
                "model_alias": model_alias,
                "entry_point": task.entry_point,
                "success": bool(visible.exec_ok and visible.total_tests > 0 and visible.passed_tests == visible.total_tests),
                "visible_passed_tests": visible.passed_tests,
                "visible_total_tests": visible.total_tests,
                "visible_execution": visible.to_dict(),
                **hidden,
                "code": code,
                "raw_response": raw,
                "wall_elapsed_sec": round(time.perf_counter() - started, 6),
                "llm_usage": usage,
                "estimated_cost_cny": estimate_cost_cny(model_alias, usage),
                "key_profile": profile.public_dict(),
                "key_attempt": attempt,
            }
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{model_alias} direct {task.data.get('task_id')} failed after key rotation: {last_error}")


def run_remcts_task(
    task: Task,
    model_alias: str,
    pool: dict[str, list[ApiProfile]],
    report_dir: Path,
    max_key_switches: int,
) -> dict[str, Any]:
    executor = CppExecutor()
    problem = build_problem(task)
    last_error: Exception | None = None
    for attempt in range(max_key_switches):
        profile = select_profile(pool, model_alias, task.index, attempt)
        if profile is None:
            raise RuntimeError(f"No API profile available for {model_alias}")
        client = make_client(profile, timeout_sec=90, max_retries=5)
        config = load_config(model_alias, report_dir)
        started = time.perf_counter()
        try:
            result = SearchEngine(config=config, llm_client=client, executor=executor).run(problem)
            hidden = run_hidden_eval(task, result.best_code, executor, timeout_sec=config.timeout_sec, max_output_chars=config.max_output_chars)
            usage = compact_usage(client.usage_snapshot())
            return {
                "task_index": task.index,
                "task_id": task.data.get("task_id"),
                "method": "remcts",
                "model_alias": model_alias,
                "entry_point": task.entry_point,
                "success": result.success,
                "accepted": result.accepted,
                "needs_review": result.needs_review,
                "stop_reason": result.stop_reason,
                "iterations": result.iterations,
                "tree_size": result.tree_size,
                "run_dir": result.run_dir,
                **hidden,
                "code": result.best_code,
                "wall_elapsed_sec": round(time.perf_counter() - started, 6),
                "llm_usage": usage,
                "estimated_cost_cny": estimate_cost_cny(model_alias, usage),
                "key_profile": profile.public_dict(),
                "key_attempt": attempt,
                "summary": result.summary,
            }
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"{model_alias} remcts {task.data.get('task_id')} failed after key rotation: {last_error}")


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_sec": 0.0,
    }
    for row in rows:
        usage = row.get("llm_usage") or {}
        for key in totals:
            value = usage.get(key, 0)
            totals[key] += float(value or 0) if key == "elapsed_sec" else int(value or 0)
    totals["elapsed_sec"] = round(float(totals["elapsed_sec"]), 6)
    return totals


def write_report(report_dir: Path, model_alias: str, method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)
    label = f"{slug_model(model_alias)}_{method}_humanevalx_cpp"
    completed = len(rows)
    hidden_success = sum(1 for row in rows if row.get("hidden_success"))
    visible_success = sum(1 for row in rows if row.get("success"))
    usage = aggregate_usage(rows)
    report = {
        "label": label,
        "model_alias": model_alias,
        "method": method,
        "dataset": "humanevalx_cpp",
        "dataset_source": DATA_URL,
        "status": "complete",
        "size": completed,
        "completed": completed,
        "visible_success_rate": round(visible_success / completed, 4) if completed else 0.0,
        "hidden_success_rate": round(hidden_success / completed, 4) if completed else 0.0,
        "llm_usage": {"totals": usage},
        "estimated_cost_cny": round(sum(float(row.get("estimated_cost_cny") or 0.0) for row in rows), 6),
        "tasks": sorted(rows, key=lambda row: int(row.get("task_index", 0))),
    }
    (report_dir / f"{label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def write_summary(report_dir: Path, reports: list[dict[str, Any]]) -> None:
    lines = [
        "# HumanEval-X C++ Pilot",
        "",
        f"- Generated: {utc_stamp()}",
        f"- Dataset source: {DATA_URL}",
        "- Visible tests are HumanEval-X `example_test`; hidden tests are the official `test` field.",
        "- ReMCTS uses the C++ known-test config with 3 iterations and branch factor 3.",
        "",
        "| Model | Method | Tasks | Visible pass | Hidden pass | Tok/task | Calls/task | Cost CNY |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        completed = int(report.get("completed") or 0)
        usage = report.get("llm_usage", {}).get("totals", {})
        tok_task = int(usage.get("total_tokens") or 0) / completed if completed else 0.0
        calls_task = int(usage.get("calls") or 0) / completed if completed else 0.0
        lines.append(
            "| {model} | {method} | {tasks} | {visible:.2f}% | {hidden:.2f}% | {tok:.0f} | {calls:.1f} | {cost:.4f} |".format(
                model=report.get("model_alias"),
                method=report.get("method"),
                tasks=completed,
                visible=100 * float(report.get("visible_success_rate") or 0.0),
                hidden=100 * float(report.get("hidden_success_rate") or 0.0),
                tok=tok_task,
                calls=calls_task,
                cost=float(report.get("estimated_cost_cny") or 0.0),
            )
        )
    lines.append("")
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a HumanEval-X C++ Direct/ReMCTS pilot.")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b"])
    parser.add_argument("--methods", nargs="+", default=["direct", "remcts"], choices=["direct", "remcts"])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--key-pool-file", default="apikeys.txt")
    parser.add_argument("--max-key-switches", type=int, default=6)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    if args.force_download:
        ensure_dataset(force=True)
    tasks = load_tasks(limit=args.limit)
    report_dir = Path(args.report_dir or ROOT / "outputs/reports" / f"rebuttal-humanevalx-cpp-{utc_stamp()}")
    report_dir.mkdir(parents=True, exist_ok=True)
    pool = load_key_pool(args.key_pool_file, args.models)
    print(f"[info] report_dir={report_dir}", flush=True)
    print(f"[info] key_pool={{" + ", ".join(f"{k}:{len(v)}" for k, v in pool.items()) + "}", flush=True)

    specs: list[tuple[str, str, Task]] = []
    for model in args.models:
        for method in args.methods:
            for task in tasks:
                specs.append((model, method, task))

    rows_by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {
        (model, method): [] for model in args.models for method in args.methods
    }
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        futures = {}
        for model, method, task in specs:
            fn = run_direct_task if method == "direct" else run_remcts_task
            future = executor.submit(fn, task, model, pool, report_dir, max(1, int(args.max_key_switches)))
            futures[future] = (model, method, task)
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            model, method, task = futures[future]
            completed += 1
            try:
                row = future.result()
                rows_by_pair[(model, method)].append(row)
                print(
                    f"[progress] {completed}/{len(specs)} {model} {method} {task.data.get('task_id')} "
                    f"hidden={bool(row.get('hidden_success'))}",
                    flush=True,
                )
            except Exception as exc:
                error = {"model": model, "method": method, "task_id": task.data.get("task_id"), "error": repr(exc)}
                errors.append(error)
                print(f"[error] {error}", flush=True)

    reports = []
    for (model, method), rows in rows_by_pair.items():
        if rows:
            reports.append(write_report(report_dir, model, method, rows))
    write_summary(report_dir, reports)
    if errors:
        (report_dir / "errors.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"{len(errors)} tasks failed; see {report_dir / 'errors.json'}")
    print(f"[summary] {report_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
