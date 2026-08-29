from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from remcts.core.search_engine import SearchEngine
from remcts.core.state import SearchConfig
from remcts.evaluation.benchmark_runner import BenchmarkRunner
from remcts.evaluation.metrics import average_iterations, pass_rate
from remcts.tasks.humaneval import HumanEvalDataset
from remcts.tasks.mbpp import MBPPDataset


def _aggregate_llm_usage(results) -> dict:
    totals = {
        "calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "elapsed_sec": 0.0,
    }
    by_model: dict[str, dict] = {}
    for result in results:
        usage = result.summary.get("llm_usage") or {}
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
    totals["elapsed_sec"] = round(totals["elapsed_sec"], 6)
    for model_totals in by_model.values():
        model_totals["elapsed_sec"] = round(model_totals["elapsed_sec"], 6)
    return {"totals": totals, "by_model": by_model}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReMCTS on a dataset.")
    parser.add_argument("--config", default="src/remcts/config/default.yaml")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--api-alias", help="Alias of the API profile to load from llm.credentials_file.")
    args = parser.parse_args()

    config = SearchConfig.from_mapping(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    if args.api_alias:
        config.llm_api_alias = args.api_alias
    dataset = HumanEvalDataset(args.data_path) if args.dataset == "humaneval" else MBPPDataset(args.data_path)
    runner = BenchmarkRunner(SearchEngine(config))
    results = runner.run_dataset(dataset)
    summary = {
        "dataset": args.dataset,
        "size": len(results),
        "pass_rate": pass_rate(results),
        "avg_iterations": average_iterations(results),
        "llm_usage": _aggregate_llm_usage(results),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
