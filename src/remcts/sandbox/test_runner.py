from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec

from .cpp_executor import CppExecutor
from .executor import PythonExecutor


def executor_for(problem: ProblemSpec) -> PythonExecutor | CppExecutor:
    """C++17 executor for HumanEval-X C++ tasks, Python executor otherwise."""
    language = str(problem.metadata.get("language") or "").lower()
    return CppExecutor() if language in {"cpp", "c++", "cxx"} else PythonExecutor()


def run_problem_tests(
    problem: ProblemSpec,
    candidate_code: str,
    executor: PythonExecutor | CppExecutor,
    timeout_sec: int,
    max_output_chars: int,
) -> ExecutionResult:
    return executor.run(
        code=candidate_code,
        tests=_with_humaneval_check_call(problem),
        timeout_sec=timeout_sec,
        max_output_chars=max_output_chars,
    )


def _with_humaneval_check_call(problem: ProblemSpec) -> str:
    """HumanEval tests define ``check(candidate)``; append the call when it is missing."""
    tests = problem.tests
    entry_point = problem.entry_point
    if not tests or not entry_point or "def check(candidate)" not in tests or f"check({entry_point})" in tests:
        return tests
    return f"{tests.rstrip()}\n\ncheck({entry_point})\n"
