from __future__ import annotations

from uuid import uuid4

from remcts.core.state import ExecutionResult, ReflectionRecord
from remcts.sandbox.trace_parser import parse_traceback


def infer_fix_hint(error_type: str) -> str:
    hints = {
        "SyntaxError": "检查括号、缩进和函数定义格式。",
        "TypeError": "检查变量类型、函数参数和返回值。",
        "IndexError": "检查边界条件和索引访问范围。",
        "AssertionError": "检查输出逻辑与题目要求是否一致。",
        "TimeoutError": "检查是否存在死循环或超高复杂度。",
    }
    return hints.get(error_type, "结合报错信息重新检查核心逻辑和边界条件。")


def build_reflection_from_execution(
    execution_result: ExecutionResult,
    reward_delta: float = 0.0,
    source_task_id: str | None = None,
    source_node_id: str | None = None,
) -> ReflectionRecord:
    parsed = parse_traceback(execution_result.traceback or execution_result.stderr)
    error_type = parsed.get("error_type") or ("TimeoutError" if execution_result.timed_out else "ExecutionError")
    root_cause = parsed.get("message") or execution_result.traceback or execution_result.stderr or "Unknown execution failure"
    fix_hint = infer_fix_hint(error_type)
    severity = 1.0
    if execution_result.compile_ok and not execution_result.exec_ok:
        severity = 0.8
    if execution_result.exec_ok and execution_result.total_tests > 0:
        pass_ratio = execution_result.passed_tests / execution_result.total_tests
        severity = max(0.2, 1.0 - pass_ratio)
    summary = f"{error_type}: {root_cause.strip().splitlines()[-1][:200]}"
    return ReflectionRecord(
        reflection_id=f"reflection-{uuid4().hex[:12]}",
        summary=summary,
        error_type=error_type,
        root_cause=root_cause[:500],
        fix_hint=fix_hint,
        severity=severity,
        reward_delta=reward_delta,
        source_task_id=source_task_id,
        source_node_id=source_node_id,
    )


def render_reflection(record: ReflectionRecord) -> str:
    return (
        f"error_type={record.error_type}; summary={record.summary}; "
        f"root_cause={record.root_cause}; fix_hint={record.fix_hint}"
    )
