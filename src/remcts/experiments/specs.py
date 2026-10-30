"""Experiment specs and the suites of the paper: main table, baselines and memory ablations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from remcts.core.state import ProblemSpec

from .datasets import DATASETS, DatasetSpec
from .prompts import build_direct_prompt
from .usage import estimate_cost_cny

# Search configs (packaged YAML names) per (method, setting).
SEARCH_CONFIGS = {
    ("remcts", "known"): "remcts_known",
    ("remcts", "no_tests"): "remcts_no_tests",
    ("pure_mcts", "no_tests"): "pure_mcts_no_tests",
    ("mcts_style", "known"): "mcts_style_known",
    ("mcts_style", "no_tests"): "mcts_style_no_tests",
}
CPP_CONFIG = "remcts_cpp_known"

# Memory ablations: config overrides on top of the full ReMCTS config.
ABLATIONS = {
    "no_global_memory": {"search": {"global_memory_top_t": 0}},
    "no_path_memory": {"search": {"path_memory_k": 0}},
}
BASELINE_METHODS = ("best_of_n", "reflexion", "reflection_style", "pure_mcts", "mcts_style")

# Logged (prompt, completion) tokens of full Qwen3-8B ReMCTS runs; only used for budget estimates.
REFERENCE_SEARCH_USAGE = {
    "humaneval_full_known": (1_633_843, 372_884),
    "humaneval_full_no_tests": (821_943, 210_036),
    "mbpp_sanitized_full_known": (5_798_278, 1_286_010),
    "mbpp_sanitized_full_no_tests": (1_493_723, 415_300),
}


@dataclass
class ExperimentSpec:
    label: str
    model_alias: str
    method: str  # name used in labels and reports, e.g. "remcts", "best_of_8"
    runner: str  # key of remcts.experiments.methods.METHODS
    dataset: DatasetSpec
    config: str  # packaged config name or YAML path
    config_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    reuse_from_label: str | None = None  # Direct no_tests reuses the generations of Direct known


def slug(text: str) -> str:
    return text.replace(".", "_").replace("-", "_")


def make_label(model_alias: str, method: str, dataset: str, suffix: str = "") -> str:
    prefix = f"{slug(model_alias)}_{slug(suffix)}" if suffix else slug(model_alias)
    return f"{prefix}_{method}_{dataset}"


def _config_for(method: str, dataset: DatasetSpec) -> str:
    if dataset.language == "cpp":
        return CPP_CONFIG
    return SEARCH_CONFIGS[(method, dataset.setting)]


def main_specs(
    models: list[str],
    datasets: list[str],
    *,
    include_direct: bool = True,
    include_remcts: bool = True,
    reuse_direct_for_no_tests: bool = True,
    label_suffix: str = "",
) -> list[ExperimentSpec]:
    """Direct and ReMCTS on each dataset (Table 1; with ``humanevalx_cpp``, the C++ pilot of Table 7).

    The direct baseline does not use tests, so by default Direct no_tests re-scores the Direct known
    generations instead of calling the API again.
    """
    specs: list[ExperimentSpec] = []
    for model in models:
        for method, runner, enabled in (("direct", "direct", include_direct), ("remcts", "search", include_remcts)):
            if not enabled:
                continue
            for name in datasets:
                dataset = DATASETS[name]
                reuse = None
                if runner == "direct" and dataset.setting == "no_tests" and reuse_direct_for_no_tests:
                    reuse = make_label(model, "direct", name.replace("_no_tests", "_known"), label_suffix)
                specs.append(
                    ExperimentSpec(
                        label=make_label(model, method, name, label_suffix),
                        model_alias=model,
                        method=method,
                        runner=runner,
                        dataset=dataset,
                        config=_config_for("remcts", dataset),
                        reuse_from_label=reuse,
                    )
                )
    return specs


def baseline_specs(
    models: list[str],
    methods: list[str],
    datasets: list[str],
    *,
    best_of_n: int = 8,
    reflexion_trials: int = 8,
    generated_test_count: int = 5,
    best_of_temperature: float = 0.7,
    actor_temperature: float = 0.2,
    test_temperature: float = 0.2,
    reflection_temperature: float = 0.2,
    compute_oracle_pool: bool = True,
) -> list[ExperimentSpec]:
    """Supplementary baselines (Tables 2, 11, 12).

    Best-of-N and Reflexion-style generate their own tests and run on ``no_tests`` datasets;
    Reflection-style uses the visible tests (``known``); pure-MCTS runs on ``no_tests`` and
    MCTS-style on both settings. Incompatible (method, dataset) pairs are skipped.
    """
    generated_tests = {
        "generated_test_count": generated_test_count,
        "test_temperature": test_temperature,
        "compute_oracle_pool": compute_oracle_pool,
    }
    loop = {
        "trials": reflexion_trials,
        "actor_temperature": actor_temperature,
        "reflection_temperature": reflection_temperature,
    }
    # method -> (setting it runs in, name in labels, runner, config, params)
    variants: dict[str, list[tuple[str, str, str, str, dict[str, Any]]]] = {
        "best_of_n": [
            (
                "no_tests",
                f"best_of_{best_of_n}",
                "best_of_n",
                "remcts_no_tests",
                {"n": best_of_n, "best_of_temperature": best_of_temperature, **generated_tests},
            )
        ],
        "reflexion": [
            (
                "no_tests",
                "reflexion" if reflexion_trials == 8 else f"reflexion_{reflexion_trials}",
                "reflexion",
                "remcts_no_tests",
                {**loop, **generated_tests},
            )
        ],
        "reflection_style": [("known", "reflection_style", "reflection_style", "remcts_known", loop)],
        "pure_mcts": [("no_tests", "pure_mcts", "search", SEARCH_CONFIGS[("pure_mcts", "no_tests")], {})],
        "mcts_style": [
            (setting, "mcts_style", "search", SEARCH_CONFIGS[("mcts_style", setting)], {})
            for setting in ("known", "no_tests")
        ],
    }
    specs: list[ExperimentSpec] = []
    for model in models:
        for name in datasets:
            dataset = DATASETS[name]
            for method in methods:
                for setting, label, runner, config, params in variants[method]:
                    if setting != dataset.setting or dataset.language != "python":
                        continue
                    specs.append(
                        ExperimentSpec(
                            label=make_label(model, label, name),
                            model_alias=model,
                            method=label,
                            runner=runner,
                            dataset=dataset,
                            config=config,
                            params=dict(params),
                        )
                    )
    return specs


def ablation_specs(models: list[str], ablations: list[str], datasets: list[str]) -> list[ExperimentSpec]:
    """ReMCTS without global memory retrieval or without path reflection memory (Tables 3, 16)."""
    specs: list[ExperimentSpec] = []
    for model in models:
        for name in datasets:
            dataset = DATASETS[name]
            for ablation in ablations:
                method = f"remcts_{ablation}"
                specs.append(
                    ExperimentSpec(
                        label=make_label(model, method, name),
                        model_alias=model,
                        method=method,
                        runner="search",
                        dataset=dataset,
                        config=SEARCH_CONFIGS[("remcts", dataset.setting)],
                        config_overrides=ABLATIONS[ablation],
                    )
                )
    return specs


def estimate_spec_cost_cny(spec: ExperimentSpec, tasks: list[ProblemSpec]) -> float:
    """Rough budget estimate: reference ReMCTS usage for searches, prompt length for API baselines."""
    if spec.reuse_from_label or not tasks:
        return 0.0
    if spec.runner == "search":
        prompt_tokens, completion_tokens = REFERENCE_SEARCH_USAGE.get(spec.dataset.name, (0, 0))
        scale = len(tasks) / spec.dataset.full_size if spec.dataset.full_size else 1.0
        return estimate_cost_cny(spec.model_alias, int(prompt_tokens * scale), int(completion_tokens * scale))
    calls = {
        "direct": 1,
        "best_of_n": int(spec.params.get("n", 8)) + 1,
        "reflexion": 2 * int(spec.params.get("trials", 8)) + 1,
        "reflection_style": 2 * int(spec.params.get("trials", 8)),
    }[spec.runner]
    prompt_tokens = sum(max(1, len(build_direct_prompt(problem)) // 3) for problem in tasks)
    return estimate_cost_cny(spec.model_alias, calls * prompt_tokens, calls * 700 * len(tasks))
