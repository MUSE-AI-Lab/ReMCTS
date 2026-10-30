"""remcts-experiment: reproduce the paper experiments.

    remcts-experiment main       Direct and ReMCTS (Table 1; ``--datasets humanevalx_cpp`` for the C++ pilot)
    remcts-experiment baselines  Best-of-N, Reflexion-style, Reflection-style, pure-MCTS and MCTS-style
    remcts-experiment ablations  ReMCTS without global memory or without path memory
    remcts-experiment status     progress of a report directory

Start with ``--dry-run``: it lists the planned runs with a rough cost estimate and makes no API call.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .datasets import DATASETS, PYTHON_DATASETS
from .keys import load_key_pool
from .reporting import status_lines, utc_stamp
from .runner import BudgetExceeded, RunOptions, load_tasks, run_experiments
from .specs import (
    ABLATIONS,
    BASELINE_METHODS,
    ExperimentSpec,
    ablation_specs,
    baseline_specs,
    estimate_spec_cost_cny,
    main_specs,
)

PAPER_MODELS = ["qwen3-1.7b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "deepseek-v4-flash"]
SUPPLEMENTARY_MODELS = ["qwen3-8b", "qwen3-32b", "deepseek-v4-flash"]


def _add_common(parser: argparse.ArgumentParser, models: list[str]) -> None:
    parser.add_argument("--models", nargs="+", default=models, help="API model aliases ('mock' runs offline).")
    parser.add_argument("--datasets", nargs="+", default=PYTHON_DATASETS, choices=sorted(DATASETS))
    parser.add_argument("--report-dir", type=Path, default=None, help="Default: outputs/reports/<command>-<time>.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=4, help="Worker processes shared by all runs.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N tasks of each dataset.")
    parser.add_argument("--force", action="store_true", help="Ignore existing reports and progress files.")
    parser.add_argument("--dry-run", action="store_true", help="List the planned runs and estimated cost only.")
    parser.add_argument("--budget-cny", type=float, default=100.0, help="Stop scheduling at this spend (0 = no limit).")
    parser.add_argument("--initial-spent-cny", type=float, default=0.0)
    parser.add_argument("--stop-margin-cny", type=float, default=5.0)
    parser.add_argument("--credentials-file", default="apikeys.txt", help="Key pool or llm.credentials_file profiles.")
    parser.add_argument("--no-key-rotation", action="store_true", help="Do not spread tasks over pooled keys.")
    parser.add_argument("--max-key-switches", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remcts-experiment", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    main = commands.add_parser("main", help="Direct and ReMCTS on the main benchmarks.")
    _add_common(main, PAPER_MODELS)
    main.add_argument("--skip-direct", action="store_true")
    main.add_argument("--skip-remcts", action="store_true")
    main.add_argument(
        "--include-direct-no-tests",
        action="store_true",
        help="Call the API for Direct no_tests instead of re-scoring the Direct known generations.",
    )
    main.add_argument("--label-suffix", default="", help="Inserted into labels to keep separate runs apart.")

    baselines = commands.add_parser("baselines", help="Supplementary sampling, reflection and tree-search baselines.")
    _add_common(baselines, SUPPLEMENTARY_MODELS)
    baselines.add_argument("--methods", nargs="+", default=list(BASELINE_METHODS), choices=BASELINE_METHODS)
    baselines.add_argument("--best-of-n", type=int, default=8)
    baselines.add_argument("--reflexion-trials", type=int, default=8)
    baselines.add_argument("--generated-test-count", type=int, default=5)
    baselines.add_argument("--best-of-temperature", type=float, default=0.7)
    baselines.add_argument("--actor-temperature", type=float, default=0.2)
    baselines.add_argument("--test-temperature", type=float, default=0.2)
    baselines.add_argument("--reflection-temperature", type=float, default=0.2)
    baselines.add_argument("--no-oracle-pool", action="store_true", help="Skip the offline Oracle-Pool statistic.")

    ablations = commands.add_parser("ablations", help="Memory ablations of ReMCTS.")
    _add_common(ablations, SUPPLEMENTARY_MODELS)
    ablations.add_argument("--ablations", nargs="+", default=list(ABLATIONS), choices=sorted(ABLATIONS))

    status = commands.add_parser("status", help="Show the progress of a report directory.")
    status.add_argument("report_dir", type=Path)
    return parser


def _specs(args: argparse.Namespace) -> list[ExperimentSpec]:
    if args.command == "main":
        return main_specs(
            args.models,
            args.datasets,
            include_direct=not args.skip_direct,
            include_remcts=not args.skip_remcts,
            reuse_direct_for_no_tests=not args.include_direct_no_tests,
            label_suffix=args.label_suffix,
        )
    if args.command == "baselines":
        return baseline_specs(
            args.models,
            args.methods,
            args.datasets,
            best_of_n=args.best_of_n,
            reflexion_trials=args.reflexion_trials,
            generated_test_count=args.generated_test_count,
            best_of_temperature=args.best_of_temperature,
            actor_temperature=args.actor_temperature,
            test_temperature=args.test_temperature,
            reflection_temperature=args.reflection_temperature,
            compute_oracle_pool=not args.no_oracle_pool,
        )
    return ablation_specs(args.models, args.ablations, args.datasets)


def _print_key_pool(args: argparse.Namespace) -> None:
    for model in args.models:
        if model == "mock" or args.no_key_rotation:
            continue
        profiles = load_key_pool(args.credentials_file, model)
        if profiles:
            fingerprints = ", ".join(profile.fingerprint for profile in profiles[:4])
            more = ", ..." if len(profiles) > 4 else ""
            print(f"[keys] {model}: {len(profiles)} pooled keys [{fingerprints}{more}]")
        else:
            print(f"[keys] {model}: no pooled key; using llm.credentials_file profiles or $REMCTS_API_KEY")


def _dry_run(specs: list[ExperimentSpec], options: RunOptions) -> None:
    total = options.initial_spent_cny
    for spec in specs:
        tasks = load_tasks(spec, options)
        estimate = estimate_spec_cost_cny(spec, tasks)
        total += estimate
        reuse = f", reuses {spec.reuse_from_label}" if spec.reuse_from_label else ""
        print(f"{spec.label}: tasks={len(tasks)}, config={spec.config}, est_cny={estimate:.2f}{reuse}")
    print(f"projected total (incl. initial spend): CNY {total:.2f}; budget: CNY {options.budget_cny:.2f}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print("\n".join(status_lines(args.report_dir)) or f"no experiments in {args.report_dir}")
        return

    options = RunOptions(
        report_dir=args.report_dir or Path("outputs/reports") / f"{args.command}-{utc_stamp()}",
        data_dir=args.data_dir,
        workers=args.workers,
        limit=args.limit,
        force=args.force,
        budget_cny=args.budget_cny,
        initial_spent_cny=args.initial_spent_cny,
        stop_margin_cny=args.stop_margin_cny,
        credentials_file=args.credentials_file,
        key_rotation=not args.no_key_rotation,
        max_key_switches=args.max_key_switches,
    )
    specs = _specs(args)
    if not specs:
        sys.exit("No runs match the requested methods and datasets.")
    print(f"report_dir={options.report_dir}")
    _print_key_pool(args)
    if args.dry_run:
        _dry_run(specs, options)
        return
    try:
        run_experiments(specs, options)
    except BudgetExceeded as exc:
        print(f"[budget-stop] {exc}", flush=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
