"""Per-task methods compared in the paper.

``search``            ReMCTS and the tree-search baselines (pure-MCTS, MCTS-style, memory ablations);
                      they differ only in the YAML config.
``direct``            one generation, scored on the hidden tests.
``best_of_n``         n independent samples, selected with model-generated tests.
``reflexion``         Actor / Evaluator / Self-Reflection loop with model-generated tests.
``reflection_style``  the same loop with the visible tests as the evaluator (``known`` setting).

Each method returns the method-specific part of a task row. Hidden tests are only run on
returned candidates (and, for the offline Oracle-Pool upper bound, on every candidate),
never to generate, select or repair them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import re
from typing import Any, Callable

from remcts.core.search_engine import SearchEngine
from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.llm.prompt_builder import is_cpp
from remcts.llm.response_parser import extract_code, parse_json_block
from remcts.sandbox.cpp_executor import strip_cpp_main_or_tests
from remcts.sandbox.test_runner import executor_for, run_problem_tests

from .prompts import (
    build_actor_prompt,
    build_best_of_prompt,
    build_cpp_direct_prompt,
    build_direct_prompt,
    build_generated_tests_prompt,
    build_reflection_prompt,
    execution_feedback,
    truncate_text,
)


@dataclass
class TaskContext:
    client: BaseLLMClient
    config: SearchConfig
    problem: ProblemSpec
    hidden_problem: ProblemSpec | None
    params: dict[str, Any] = field(default_factory=dict)

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return self.client.generate(prompt, temperature=temperature, max_tokens=max_tokens)

    def run_tests(self, problem: ProblemSpec, code: str) -> ExecutionResult:
        return run_problem_tests(
            problem,
            code,
            executor_for(problem),
            timeout_sec=self.config.timeout_sec,
            max_output_chars=self.config.max_output_chars,
        )

    def hidden_eval(self, code: str) -> dict[str, Any]:
        if self.hidden_problem is None or not code:
            return {}
        return hidden_fields(self.run_tests(self.hidden_problem, code))


def hidden_fields(result: ExecutionResult) -> dict[str, Any]:
    return {
        "hidden_success": result.all_passed(),
        "hidden_passed_tests": int(result.passed_tests),
        "hidden_total_tests": int(result.total_tests),
        "hidden_failed_cases": list(result.failed_cases[:8]),
    }


def run_search(ctx: TaskContext) -> dict[str, Any]:
    result = SearchEngine(ctx.config, llm_client=ctx.client, executor=executor_for(ctx.problem)).run(ctx.problem)
    return {
        "success": bool(result.success),
        "accepted": bool(result.accepted),
        "needs_review": bool(result.needs_review),
        "stop_reason": result.stop_reason,
        "iterations": result.iterations,
        "tree_size": result.tree_size,
        "run_dir": result.run_dir,
        "code": result.best_code,
        **ctx.hidden_eval(result.best_code),
    }


def run_direct(ctx: TaskContext) -> dict[str, Any]:
    problem, config = ctx.problem, ctx.config
    prompt = build_cpp_direct_prompt(problem) if is_cpp(problem) else build_direct_prompt(problem)
    raw = ctx.generate(prompt, temperature=config.llm_temperature, max_tokens=config.llm_max_tokens)
    code = normalize_cpp_code(raw, problem) if is_cpp(problem) else extract_code(raw).strip()

    hidden = ctx.hidden_problem
    result = ctx.run_tests(hidden or problem, code)
    success = result.all_passed()
    if hidden is not None:
        stop_reason = "direct_hidden_tests_passed" if success else "direct_hidden_single_call_evaluated"
    else:
        stop_reason = "direct_tests_passed" if success else "direct_single_call_evaluated"
    row: dict[str, Any] = {
        "success": success,
        "accepted": success,
        "needs_review": False,
        "stop_reason": stop_reason,
        "iterations": 1,
        "tree_size": 1,
        "run_dir": None,
        "execution": result.to_dict(),
        "code": code,
    }
    if hidden is not None:
        row.update(hidden_fields(result))
        if problem.tests.strip():
            visible = ctx.run_tests(problem, code)
            row.update(
                visible_success=visible.all_passed(),
                visible_passed_tests=int(visible.passed_tests),
                visible_total_tests=int(visible.total_tests),
            )
    return row


def run_best_of_n(ctx: TaskContext) -> dict[str, Any]:
    n = int(ctx.params["n"])
    tests = _generate_tests(ctx)
    feedback_problem = replace(ctx.problem, tests=tests, metadata=dict(ctx.problem.metadata))
    candidates: list[dict[str, Any]] = []
    for sample_index in range(1, n + 1):
        raw = ctx.generate(
            build_best_of_prompt(ctx.problem, sample_index, n),
            temperature=float(ctx.params["best_of_temperature"]),
            max_tokens=ctx.config.llm_max_tokens,
        )
        code = normalize_candidate_code(extract_code(raw))
        result = ctx.run_tests(feedback_problem, code)
        candidates.append({"sample_index": sample_index, **_attempt_fields(ctx, result, code), "code": code})

    selected = max(candidates, key=lambda item: _selection_key(item, item["sample_index"]))
    accepted = bool(selected["evaluator_success"])
    return {
        "success": accepted,
        "accepted": accepted,
        "needs_review": not accepted,
        "iterations": n,
        "tree_size": n,
        "stop_reason": "best_of_n_generated_tests_passed" if accepted else "best_of_n_selected_by_proxy_score",
        "generated_tests": _tests_summary(tests),
        "selected_sample_index": int(selected["sample_index"]),
        **_selected_fields(selected, candidates),
        "candidate_summaries": [_without_code(item) for item in candidates],
        "code": str(selected["code"]),
        **ctx.hidden_eval(str(selected["code"])),
    }


def run_reflexion(ctx: TaskContext) -> dict[str, Any]:
    tests = _generate_tests(ctx)
    feedback_problem = replace(ctx.problem, tests=tests, metadata=dict(ctx.problem.metadata))
    attempts, reflections, selected = _reflexion_loop(ctx, feedback_problem, tests, truncate_feedback=True)
    accepted = bool(selected["evaluator_success"])
    return {
        **_reflexion_row(attempts, reflections, selected),
        "stop_reason": "reflexion_generated_tests_passed" if accepted else "reflexion_max_trials",
        "generated_tests": _tests_summary(tests),
        **ctx.hidden_eval(str(selected["code"])),
    }


def run_reflection_style(ctx: TaskContext) -> dict[str, Any]:
    visible_tests = ctx.problem.tests.strip() or "(no visible tests available)"
    attempts, reflections, selected = _reflexion_loop(ctx, ctx.problem, visible_tests, truncate_feedback=False)
    accepted = bool(selected["evaluator_success"])
    return {
        **_reflexion_row(attempts, reflections, selected),
        "stop_reason": "reflection_visible_tests_passed" if accepted else "reflection_max_trials",
        "visible_test_count": len([line for line in visible_tests.splitlines() if line.strip()]),
        **ctx.hidden_eval(str(selected["code"])),
    }


METHODS: dict[str, Callable[[TaskContext], dict[str, Any]]] = {
    "search": run_search,
    "direct": run_direct,
    "best_of_n": run_best_of_n,
    "reflexion": run_reflexion,
    "reflection_style": run_reflection_style,
}


def _reflexion_loop(
    ctx: TaskContext,
    feedback_problem: ProblemSpec,
    tests_text: str,
    *,
    truncate_feedback: bool,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    """Actor -> evaluator -> self-reflection until the feedback tests pass or trials run out."""
    reflections: list[str] = []
    attempts: list[dict[str, Any]] = []
    for trial in range(1, int(ctx.params["trials"]) + 1):
        raw = ctx.generate(
            build_actor_prompt(ctx.problem, tests_text, reflections, trial),
            temperature=float(ctx.params["actor_temperature"]),
            max_tokens=ctx.config.llm_max_tokens,
        )
        code = normalize_candidate_code(extract_code(raw))
        result = ctx.run_tests(feedback_problem, code)
        feedback = execution_feedback(result)
        attempt = {
            "trial": trial,
            **_attempt_fields(ctx, result, code),
            "feedback": truncate_text(feedback, 2200) if truncate_feedback else feedback,
            "code": code,
        }
        attempts.append(attempt)
        if attempt["evaluator_success"]:
            return attempts, reflections, attempt
        reflection = parse_reflection(
            ctx.generate(
                build_reflection_prompt(ctx.problem, code, tests_text, feedback, reflections),
                temperature=float(ctx.params["reflection_temperature"]),
                max_tokens=min(int(ctx.config.llm_max_tokens), 800),
            )
        )
        if reflection:
            reflections.append(reflection)
    selected = max(attempts, key=lambda item: _selection_key(item, item["trial"]))
    return attempts, reflections, selected


def _reflexion_row(attempts: list[dict[str, Any]], reflections: list[str], selected: dict[str, Any]) -> dict[str, Any]:
    accepted = bool(selected["evaluator_success"])
    return {
        "success": accepted,
        "accepted": accepted,
        "needs_review": not accepted,
        "iterations": int(selected["trial"]) if accepted else len(attempts),
        "tree_size": len(attempts),
        "reflection_count": len(reflections),
        "reflections": reflections,
        "selected_trial": int(selected["trial"]),
        **_selected_fields(selected, attempts),
        "attempt_summaries": [_without_code(item) for item in attempts],
        "code": str(selected["code"]),
    }


def _generate_tests(ctx: TaskContext) -> str:
    count = int(ctx.params["generated_test_count"])
    raw = ctx.generate(
        build_generated_tests_prompt(ctx.problem, count),
        temperature=float(ctx.params["test_temperature"]),
        max_tokens=min(int(ctx.config.llm_max_tokens), 900),
    )
    return parse_generated_tests(raw, count)


def _attempt_fields(ctx: TaskContext, result: ExecutionResult, code: str) -> dict[str, Any]:
    fields = {
        "proxy_score": baseline_proxy_score(result),
        "evaluator_success": result.all_passed(),
        "passed_tests": int(result.passed_tests),
        "total_tests": int(result.total_tests),
        "compile_ok": bool(result.compile_ok),
        "exec_ok": bool(result.exec_ok),
        "timed_out": bool(result.timed_out),
        "failed_cases": list(result.failed_cases[:8]),
    }
    if ctx.params.get("compute_oracle_pool"):
        # Offline Oracle-Pool upper bound only; never used for selection.
        hidden = ctx.hidden_eval(code)
        hidden.pop("hidden_failed_cases", None)
        fields.update(hidden)
    return fields


def _selection_key(item: dict[str, Any], order: int) -> tuple:
    return float(item["proxy_score"]), bool(item["evaluator_success"]), bool(item["exec_ok"]), -int(order)


def _selected_fields(selected: dict[str, Any], pool: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_proxy_score": float(selected["proxy_score"]),
        "selected_feedback_passed_tests": int(selected["passed_tests"]),
        "selected_feedback_total_tests": int(selected["total_tests"]),
        "oracle_pool_hidden_success": any(bool(item.get("hidden_success")) for item in pool),
        "oracle_pool_hidden_success_count": sum(1 for item in pool if item.get("hidden_success")),
    }


def _without_code(item: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "code"}


def _tests_summary(tests: str) -> dict[str, Any]:
    lines = [line for line in tests.splitlines() if line.strip().startswith("assert ")]
    return {"count": len(lines), "tests": lines}


def baseline_proxy_score(result: ExecutionResult) -> float:
    """Selection score of the sampling/reflection baselines (not the ReMCTS reward)."""
    compile_score = 1.0 if result.compile_ok else 0.0
    exec_score = 1.0 if result.exec_ok else 0.0
    if int(result.total_tests) > 0:
        test_score = float(result.passed_tests) / max(float(result.total_tests), 1.0)
        return round(0.70 * test_score + 0.20 * exec_score + 0.10 * compile_score, 6)
    return round(0.60 * exec_score + 0.40 * compile_score, 6)


def parse_generated_tests(raw: str, max_tests: int) -> str:
    candidates: list[str] = []
    try:
        data = parse_json_block(raw)
        raw_tests = data.get("tests", [])
        if isinstance(raw_tests, list):
            candidates.extend(str(item).strip() for item in raw_tests)
    except Exception:
        candidates.extend(line.strip() for line in extract_code(raw).splitlines())

    tests: list[str] = []
    for line in candidates:
        stripped = line.strip().rstrip(";")
        if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')):
            stripped = stripped[1:-1].strip()
        if stripped.startswith("assert "):
            tests.append(stripped)
        if len(tests) >= max_tests:
            break
    return "\n".join(tests)


def parse_reflection(raw: str) -> str:
    try:
        data = parse_json_block(raw)
        parts = [str(data.get(key) or "").strip() for key in ("summary", "root_cause", "fix_hint")]
        text = " ".join(part for part in parts if part)
        return text.strip() or truncate_text(raw.strip(), 1200)
    except Exception:
        return truncate_text(raw.strip(), 1200)


def normalize_candidate_code(code: str) -> str:
    """Baseline candidates run with a ``from typing import *`` prelude (HumanEval completion context)."""
    stripped = str(code or "").lstrip()
    if stripped.startswith("from typing import"):
        return str(code or "").strip()
    return "from typing import *\n" + str(code or "").strip()


def normalize_cpp_code(raw: str, problem: ProblemSpec) -> str:
    code = strip_cpp_main_or_tests(extract_code(raw).strip()) or strip_cpp_main_or_tests(raw.strip())
    entry = problem.entry_point or "solution"
    if not re.search(rf"\b{re.escape(entry)}\s*\(", code):
        declaration = str(problem.metadata.get("declaration") or problem.prompt or "")
        code = f"{declaration.rstrip()}\n{code.strip()}"
    elif "#include" not in code:
        code = "#include <bits/stdc++.h>\nusing namespace std;\n" + code
    return code.strip() + "\n"
