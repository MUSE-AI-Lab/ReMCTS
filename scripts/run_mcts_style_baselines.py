#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_qwen_multi_model_experiment import (
    DatasetSpec,
    ExperimentSpec,
    ROOT,
    load_completed_outputs,
    run_one_spec,
    slug_model,
    summarize_all,
    utc_stamp,
)


MCTS_STYLE_CONFIG = ROOT / "src/remcts/config/qwen_mcts_style_production_full_k3.yaml"


def build_dataset_specs() -> dict[str, DatasetSpec]:
    return {
        "humaneval_full_no_tests": DatasetSpec(
            name="humaneval_full_no_tests",
            dataset="humaneval",
            data_path=ROOT / "data/humaneval/HumanEval_no_tests.jsonl",
            remcts_config_path=MCTS_STYLE_CONFIG,
            hidden_eval_path=ROOT / "data/humaneval/HumanEval_eval.jsonl",
        ),
        "mbpp_sanitized_full_no_tests": DatasetSpec(
            name="mbpp_sanitized_full_no_tests",
            dataset="mbpp",
            data_path=ROOT / "data/mbpp/mbpp_sanitized_no_tests.json",
            remcts_config_path=MCTS_STYLE_CONFIG,
            hidden_eval_path=ROOT / "data/mbpp/mbpp_sanitized_eval_split.json",
        ),
    }


def write_compact_summary(report_dir: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: (row.get("model_alias", ""), row.get("dataset", "")))
    lines = [
        "# MCTS-style Baseline Summary",
        "",
        "- 口径：no_tests 输入，搜索阶段不使用 hidden tests，最终统一 hidden evaluation。",
        "- MCTS-style：保留树搜索、UCB、相同搜索预算和 LLM 语义评审；关闭路径反思记忆、全局记忆检索、语义 UCB 和 LLM probe。",
        "",
        "| 模型 | 数据集 | 完成 | Hidden Solve | Accepted | Acc. Precision | False Accept. | Avg Iter. | Avg Tree | 费用(CNY) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        hidden = row.get("hidden_summary") or {}
        lines.append(
            "| {model} | {dataset} | {completed}/{size} | {rate:.2%} ({succ}/{size}) | "
            "{accepted:.2%} | {precision:.2%} | {false:.2%} | {avg_iter:.3f} | "
            "{avg_tree:.3f} | {cost:.4f} |".format(
                model=row.get("model_alias"),
                dataset=row.get("dataset"),
                completed=row.get("completed", 0),
                size=row.get("size", 0),
                rate=float(row.get("pass_rate") or 0.0),
                succ=hidden.get("hidden_success", 0),
                accepted=float(hidden.get("accepted_rate") or 0.0),
                precision=float(hidden.get("acceptance_precision") or 0.0),
                false=float(hidden.get("false_acceptance_rate") or 0.0),
                avg_iter=float(row.get("avg_iterations") or 0.0),
                avg_tree=float(row.get("avg_tree_size") or 0.0),
                cost=float(row.get("estimated_cost_cny") or 0.0),
            )
        )
    (report_dir / "mcts_style_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (report_dir / "mcts_style_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCTS-style no_tests baselines.")
    parser.add_argument("--models", nargs="+", default=["qwen3-8b", "qwen3-32b", "deepseek-v4-flash"])
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["humaneval_full_no_tests", "mbpp_sanitized_full_no_tests"],
        choices=sorted(build_dataset_specs()),
    )
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--budget-cny", type=float, default=160.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = utc_stamp()
    report_dir = args.report_dir or ROOT / "outputs/reports" / f"mcts-style-baselines-{started_at}"
    report_dir.mkdir(parents=True, exist_ok=True)
    dataset_specs = build_dataset_specs()
    specs: list[ExperimentSpec] = []
    for model in args.models:
        for dataset_name in args.datasets:
            dataset_spec = dataset_specs[dataset_name]
            specs.append(
                ExperimentSpec(
                    label=f"{slug_model(model)}_mcts_style_{dataset_name}",
                    model_alias=model,
                    method="mcts_style",
                    dataset_spec=dataset_spec,
                )
            )

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_one_spec(
            spec,
            report_dir=report_dir,
            force=bool(args.force),
            workers=max(1, int(args.workers)),
            budget_cny=float(args.budget_cny),
            initial_spent_cny=0.0,
            stop_margin_cny=0.0,
        )
        rows = load_completed_outputs(report_dir)
        if row.get("status") == "complete" and all(existing.get("label") != row["label"] for existing in rows):
            rows.append(row)
        summarize_all(
            report_dir,
            rows,
            started_at,
            budget_cny=float(args.budget_cny),
            initial_spent_cny=0.0,
            stop_margin_cny=0.0,
        )
        write_compact_summary(report_dir, rows)
    print(f"[summary] {report_dir / 'mcts_style_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
