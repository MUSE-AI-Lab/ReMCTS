from __future__ import annotations

from remcts.core.state import ReflectionRecord


def build_path_memory(
    parent_memory: list[ReflectionRecord],
    new_reflection: ReflectionRecord | None,
    k: int,
    depth: int | None = None,
) -> list[ReflectionRecord]:
    effective_k = k if depth is None else min(k, depth + 1)
    if effective_k <= 0:
        return []
    inherited = list(parent_memory)[-(max(effective_k - 1, 0)) :]
    if new_reflection is not None:
        inherited.append(new_reflection)
    return inherited[-effective_k:]


def render_memory_for_prompt(memory: list[ReflectionRecord]) -> str:
    if not memory:
        return "无历史反思记忆。"
    lines = []
    for idx, record in enumerate(memory, start=1):
        lines.append(
            f"[{idx}] error_type={record.error_type}; root_cause={record.root_cause}; fix_hint={record.fix_hint}"
        )
    return "\n".join(lines)
