from __future__ import annotations

from remcts.core.state import ExecutionResult


def score_unit(execution_result: ExecutionResult) -> float:
    if execution_result.total_tests <= 0:
        return 0.0
    return execution_result.passed_tests / execution_result.total_tests
