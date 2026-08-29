from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec

from .executor import PythonExecutor


def run_problem_tests(
    problem: ProblemSpec,
    candidate_code: str,
    executor: PythonExecutor,
    timeout_sec: int,
    max_output_chars: int,
) -> ExecutionResult:
    tests = _with_humaneval_check_call(problem)
    return executor.run(
        code=candidate_code,
        tests=tests,
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
    )


def _with_humaneval_check_call(problem: ProblemSpec) -> str:
    tests = problem.tests
    entry_point = problem.entry_point
    if not tests or not entry_point:
        return tests
    if "def check(candidate)" not in tests:
        return tests
    if f"check({entry_point})" in tests:
        return tests
    return f"{tests.rstrip()}\n\ncheck({entry_point})\n"
