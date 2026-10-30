"""Prompts of the Direct, Best-of-N and Reflexion-style baselines (kept verbatim from the paper runs)."""

from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec


def entry_point_of(problem: ProblemSpec) -> str:
    return problem.entry_point or problem.metadata.get("entry_point") or "solution"


def truncate_text(text: str, max_chars: int = 3000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 24] + "\n...[truncated]..."


def build_direct_prompt(problem: ProblemSpec) -> str:
    entry_point = entry_point_of(problem)
    return f"""
You are a careful Python coding assistant. Solve the following programming problem in one shot.

Return only complete executable Python code. Do not include Markdown fences or explanations.
The solution must define the entry function `{entry_point}` and should not rely on stdin, command-line input, or interactive prompts unless the problem explicitly requires it.

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def build_cpp_direct_prompt(problem: ProblemSpec) -> str:
    return f"""
You are a careful C++17 programming assistant. Complete the following HumanEval-X C++ task.

Return only complete C++17 source code. Do not include Markdown fences or explanations.
The code must define the required entry function `{problem.entry_point}`. Do not define `main` and do not include tests.

Problem and partial declaration:
{problem.prompt}

Required declaration / prefix:
{problem.metadata.get("declaration", "")}
""".strip()


def build_generated_tests_prompt(problem: ProblemSpec, count: int) -> str:
    entry_point = entry_point_of(problem)
    return f"""
You are the Evaluator test writer in a Reflexion-style programming loop.
Generate {count} diverse Python unit tests for the target function.

Requirements:
- Return JSON only, with exactly this shape: {{"tests": ["assert ...", "..."]}}.
- Every test must be a single Python assert statement.
- Use the entry function name `{entry_point}` exactly.
- Cover normal cases, edge cases, and boundary conditions when possible.
- Do not include imports, helper functions, Markdown, explanations, or the solution.
- Do not call stdin, files, network, random, or time-dependent APIs.

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def build_best_of_prompt(problem: ProblemSpec, sample_index: int, sample_count: int) -> str:
    return f"""
{build_direct_prompt(problem)}

Your output must start with `from typing import *` before any function definitions.
Include all other required imports explicitly.
This is independent sample {sample_index} of {sample_count}. Use your own reasoning and produce a robust implementation rather than copying a likely common template.
""".strip()


def build_actor_prompt(problem: ProblemSpec, tests: str, reflections: list[str], trial: int) -> str:
    entry_point = entry_point_of(problem)
    memory = "\n".join(f"- {item}" for item in reflections[-6:]) if reflections else "(empty)"
    tests = tests.strip() or "(the Evaluator produced no usable assert tests; rely on the task specification)"
    return f"""
You are the Actor in a Reflexion-style programming loop.
Write a complete Python solution for the task below.

Return only complete executable Python code. Do not include Markdown fences or explanations.
The solution must define the entry function `{entry_point}` and must not rely on stdin, command-line input, or interactive prompts unless the task explicitly requires it.
Your output must start with `from typing import *` before any function definitions.
Include all other required imports explicitly.

Reflection memory from previous failed trials:
{memory}

Evaluator-generated tests used for feedback:
{tests}

Trial:
{trial}

Problem:
{problem.prompt}

Entry function:
{entry_point}
""".strip()


def build_reflection_prompt(
    problem: ProblemSpec,
    code: str,
    tests: str,
    feedback: str,
    reflections: list[str],
) -> str:
    memory = "\n".join(f"- {item}" for item in reflections[-6:]) if reflections else "(empty)"
    return f"""
You are the Self-Reflection module in the Reflexion architecture.
Convert the failed programming trajectory into concise natural-language memory for the next Actor attempt.

Return JSON only, with this shape:
{{"summary": "...", "root_cause": "...", "fix_hint": "..."}}

Do not output code. Focus on what was wrong and how the next attempt should change.

Problem:
{problem.prompt}

Previous reflection memory:
{memory}

Evaluator-generated tests:
{tests or "(none)"}

Candidate code:
{truncate_text(code, 2600)}

Evaluator feedback:
{truncate_text(feedback, 2600)}
""".strip()


def execution_feedback(result: ExecutionResult) -> str:
    return "\n".join(
        [
            f"compile_ok: {bool(result.compile_ok)}",
            f"exec_ok: {bool(result.exec_ok)}",
            f"timed_out: {bool(result.timed_out)}",
            f"passed_tests: {int(result.passed_tests)}/{int(result.total_tests)}",
            f"failed_cases: {result.failed_cases[:8]}",
            f"traceback: {truncate_text(result.traceback, 1200)}",
            f"stderr: {truncate_text(result.stderr, 1200)}",
            f"stdout: {truncate_text(result.stdout, 600)}",
        ]
    )
