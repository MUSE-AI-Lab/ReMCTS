from __future__ import annotations

from remcts.core.state import ExecutionResult


def score_semantic_heuristic(execution_result: ExecutionResult) -> tuple[float, str]:
    if not execution_result.compile_ok:
        return 0.0, "代码未通过语法检查"
    if execution_result.exec_ok and execution_result.total_tests > 0:
        score = 0.3 + 0.7 * (execution_result.passed_tests / execution_result.total_tests)
        return min(score, 1.0), "基于执行结果的启发式语义评分"
    if execution_result.exec_ok:
        return 0.4, "代码可以运行，但缺少充分测试信息"
    return 0.1, "代码可以编译但执行失败"


def score_semantic(execution_result: ExecutionResult) -> tuple[float, str]:
    return score_semantic_heuristic(execution_result)
