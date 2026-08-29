from __future__ import annotations

from remcts.core.state import ProblemSpec


def build_task_context(problem: ProblemSpec) -> str:
    entry = f"\n入口函数：{problem.entry_point}" if problem.entry_point else ""
    return f"任务ID：{problem.task_id}{entry}\n\n题目：\n{problem.prompt}".strip()
