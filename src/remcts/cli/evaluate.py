from __future__ import annotations

import argparse
import json
from pathlib import Path

from remcts.config import load_config
from remcts.core.search_engine import SearchEngine
from remcts.experiments.usage import aggregate_usage, compact_usage
from remcts.sandbox.test_runner import executor_for, run_problem_tests
from remcts.tasks.humaneval import HumanEvalDataset
from remcts.tasks.mbpp import MBPPDataset

LOADERS = {"humaneval": HumanEvalDataset, "mbpp": MBPPDataset}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReMCTS over a dataset file with one config.")
    parser.add_argument("--config", default="default", help="Packaged config name or path to a YAML file.")
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True, help="File format of --data-path.")
    parser.add_argument("--data-path", required=True, help="Tasks given to the search (with or without tests).")
    parser.add_argument("--eval-path", help="Hidden-evaluation split used only to score the returned programs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-alias", help="Alias of the API profile to load from llm.credentials_file.")
    parser.add_argument("--output", type=Path, help="Write per-task rows and the summary as JSON.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.api_alias:
        config.llm_api_alias = args.api_alias
    loader = LOADERS[args.dataset]
    problems = list(loader(args.data_path).iter_tasks())[: args.limit]
    hidden = {problem.task_id: problem for problem in loader(args.eval_path).iter_tasks()} if args.eval_path else {}

    rows = []
    for problem in problems:
        # A fresh engine per task keeps the global reflection memory task-local, as in the paper.
        result = SearchEngine(config).run(problem)
        row = {
            "task_id": problem.task_id,
            "success": result.success,
            "accepted": result.accepted,
            "stop_reason": result.stop_reason,
            "iterations": result.iterations,
            "tree_size": result.tree_size,
            "run_dir": result.run_dir,
            "llm_usage": compact_usage(result.summary.get("llm_usage")),
        }
        if problem.task_id in hidden:
            hidden_problem = hidden[problem.task_id]
            check = run_problem_tests(
                hidden_problem,
                result.best_code,
                executor_for(hidden_problem),
                config.timeout_sec,
                config.max_output_chars,
            )
            row["hidden_success"] = check.all_passed()
        rows.append(row)
        print(json.dumps({key: row[key] for key in row if key != "llm_usage"}, ensure_ascii=False), flush=True)

    n = len(rows)
    summary = {
        "dataset": args.data_path,
        "config": args.config,
        "size": n,
        "search_success_rate": sum(row["success"] for row in rows) / n if n else 0.0,
        "avg_iterations": sum(row["iterations"] for row in rows) / n if n else 0.0,
        "llm_usage": aggregate_usage(rows),
    }
    scored = [row for row in rows if "hidden_success" in row]
    if scored:
        summary["hidden_solve_rate"] = sum(row["hidden_success"] for row in scored) / len(scored)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = json.dumps({"summary": summary, "tasks": rows}, ensure_ascii=False, indent=2)
        args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
