from __future__ import annotations

import json

from uuid import uuid4

from remcts.core.state import ExecutionResult, ProblemSpec, ReflectionRecord, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.llm.prompt_builder import build_reflect_prompt
from remcts.llm.response_parser import parse_json_block

from .reflection import build_reflection_from_execution


def build_execution_summary(execution_result: ExecutionResult, max_chars: int = 3000) -> str:
    parts = [
        f"compile_ok={execution_result.compile_ok}",
        f"exec_ok={execution_result.exec_ok}",
        f"timed_out={execution_result.timed_out}",
        f"passed_tests={execution_result.passed_tests}/{execution_result.total_tests}",
    ]
    if execution_result.failed_cases:
        parts.append("failed_cases:\n" + "\n".join(execution_result.failed_cases))
    if execution_result.stdout:
        parts.append("stdout:\n" + execution_result.stdout)
    if execution_result.stderr:
        parts.append("stderr:\n" + execution_result.stderr)
    if execution_result.traceback:
        parts.append("traceback:\n" + execution_result.traceback)
    if execution_result.coverage_hint:
        parts.append(
            "coverage_hint:\n"
            + json.dumps(execution_result.coverage_hint, ensure_ascii=False, indent=2)
        )
    summary = "\n\n".join(parts)
    if len(summary) > max_chars:
        return summary[:max_chars] + "\n...[truncated]"
    return summary


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _fallback_reflection(
    execution_result: ExecutionResult,
    reward_delta: float,
    source_task_id: str | None,
    source_node_id: str | None,
    fallback_reason: str | None = None,
) -> ReflectionRecord:
    record = build_reflection_from_execution(
        execution_result,
        reward_delta=reward_delta,
        source_task_id=source_task_id,
        source_node_id=source_node_id,
    )
    record.metadata["reflection_mode"] = "heuristic"
    if fallback_reason:
        record.metadata["fallback_reason"] = fallback_reason
    return record


class ReflectionGenerator:
    def __init__(self, llm_client: BaseLLMClient, config: SearchConfig):
        self.llm_client = llm_client
        self.config = config

    def generate(
        self,
        problem: ProblemSpec,
        code: str,
        execution_result: ExecutionResult,
        reward_delta: float = 0.0,
        source_task_id: str | None = None,
        source_node_id: str | None = None,
    ) -> ReflectionRecord:
        mode = self.config.reflection_mode.lower()
        if mode != "llm":
            return _fallback_reflection(
                execution_result,
                reward_delta=reward_delta,
                source_task_id=source_task_id,
                source_node_id=source_node_id,
            )

        execution_summary = build_execution_summary(
            execution_result,
            max_chars=self.config.reflection_max_feedback_chars,
        )
        prompt = build_reflect_prompt(problem, code, execution_summary)
        try:
            raw_response = self.llm_client.generate(
                prompt,
                temperature=self.config.llm_temperature,
                max_tokens=min(self.config.llm_max_tokens, 800),
            )
            parsed = parse_json_block(raw_response)
            record = self._record_from_json(
                parsed,
                reward_delta=reward_delta,
                source_task_id=source_task_id,
                source_node_id=source_node_id,
            )
            record.metadata.update(
                {
                    "reflection_mode": "llm",
                    "llm_prompt": prompt,
                    "llm_raw_response": raw_response,
                    "parsed_json": parsed,
                }
            )
            return record
        except Exception as exc:
            if not self.config.reflection_fallback_on_error:
                raise
            record = _fallback_reflection(
                execution_result,
                reward_delta=reward_delta,
                source_task_id=source_task_id,
                source_node_id=source_node_id,
                fallback_reason=f"{type(exc).__name__}: {exc}",
            )
            record.metadata["reflection_mode_requested"] = "llm"
            record.metadata["llm_prompt"] = prompt
            return record

    def _record_from_json(
        self,
        parsed: dict,
        reward_delta: float,
        source_task_id: str | None,
        source_node_id: str | None,
    ) -> ReflectionRecord:
        error_type = str(parsed.get("error_type") or "ExecutionError").strip() or "ExecutionError"
        failure_summary = str(parsed.get("failure_summary") or parsed.get("summary") or "").strip()
        root_cause = str(parsed.get("root_cause") or failure_summary or "Unknown root cause").strip()
        fix_hint = str(parsed.get("fix_hint") or "Revise the code according to the failing tests.").strip()
        try:
            severity = _clamp(float(parsed.get("severity", 0.8)))
        except (TypeError, ValueError):
            severity = 0.8
        summary = failure_summary or f"{error_type}: {root_cause[:200]}"
        return ReflectionRecord(
            reflection_id=f"reflection-{uuid4().hex[:12]}",
            summary=summary[:500],
            error_type=error_type[:100],
            root_cause=root_cause[:1000],
            fix_hint=fix_hint[:1000],
            severity=severity,
            reward_delta=reward_delta,
            source_task_id=source_task_id,
            source_node_id=source_node_id,
        )
