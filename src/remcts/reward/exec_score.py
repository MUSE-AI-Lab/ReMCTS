from __future__ import annotations

from remcts.core.state import ExecutionResult


def score_execution(execution_result: ExecutionResult) -> float:
    score = 0.0
    if execution_result.compile_ok:
        score += 0.3
    if execution_result.exec_ok:
        score += 0.3
    if execution_result.total_tests > 0:
        score += 0.2
    if execution_result.stdout or execution_result.exec_ok:
        score += 0.2
    return min(score, 1.0)
