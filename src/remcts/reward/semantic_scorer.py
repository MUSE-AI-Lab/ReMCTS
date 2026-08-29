from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.llm.prompt_builder import build_production_semantic_score_prompt, build_semantic_score_prompt
from remcts.llm.response_parser import parse_json_block
from remcts.memory.reflection_generator import build_execution_summary

from .semantic_score import score_semantic_heuristic


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


class SemanticScorer:
    def __init__(self, llm_client: BaseLLMClient | None, config: SearchConfig):
        self.llm_client = llm_client
        self.config = config

    def score(
        self,
        problem: ProblemSpec,
        code: str,
        execution_result: ExecutionResult,
    ) -> tuple[float, str, dict]:
        mode = self.config.reward_semantic_mode.lower()
        if mode == "disabled":
            return 0.0, "语义奖励已禁用", {"semantic_mode": "disabled"}
        if mode != "llm":
            return self._heuristic(execution_result)
        if self.config.evaluation_mode.lower() == "production":
            return self._score_production(problem, code, execution_result)

        if not execution_result.compile_ok:
            return 0.0, "代码未通过语法检查，跳过 LLM 语义评分", {
                "semantic_mode": "shortcut_compile_error"
            }
        if execution_result.exec_ok and execution_result.total_tests > 0 and execution_result.passed_tests == execution_result.total_tests:
            if not self.config.reward_semantic_on_success:
                return 1.0, "全部测试通过，跳过 LLM 语义评分", {
                    "semantic_mode": "shortcut_success"
                }
        if not self.config.reward_semantic_on_failure and execution_result.failed_cases:
            score, reason, details = self._heuristic(execution_result)
            details["semantic_mode_requested"] = "llm"
            details["semantic_skip_reason"] = "semantic_on_failure=false"
            return score, reason, details
        if self.llm_client is None:
            score, reason, details = self._heuristic(execution_result)
            details["semantic_mode_requested"] = "llm"
            details["semantic_fallback_reason"] = "missing llm client"
            return score, reason, details

        execution_summary = build_execution_summary(
            execution_result,
            max_chars=self.config.reward_semantic_max_feedback_chars,
        )
        prompt = build_semantic_score_prompt(problem, code, execution_summary)
        try:
            raw_response = self.llm_client.generate(
                prompt,
                temperature=self.config.llm_temperature,
                max_tokens=min(self.config.llm_max_tokens, 500),
            )
            parsed = parse_json_block(raw_response)
            score = _clamp(float(parsed.get("score", 0.0)))
            reason = str(parsed.get("reason") or "LLM semantic score")
            return score, reason, {
                "semantic_mode": "llm",
                "semantic_prompt": prompt,
                "semantic_raw_response": raw_response,
                "semantic_parsed_json": parsed,
            }
        except Exception as exc:
            score, reason, details = self._heuristic(execution_result)
            details["semantic_mode_requested"] = "llm"
            details["semantic_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            details["semantic_prompt"] = prompt
            return score, reason, details

    def _heuristic(self, execution_result: ExecutionResult) -> tuple[float, str, dict]:
        score, reason = score_semantic_heuristic(execution_result)
        return score, reason, {"semantic_mode": "heuristic"}

    def _score_production(
        self,
        problem: ProblemSpec,
        code: str,
        execution_result: ExecutionResult,
    ) -> tuple[float, str, dict]:
        if not execution_result.compile_ok:
            return 0.0, "代码未通过语法检查，跳过生产语义评分", {
                "semantic_mode": "production_shortcut_compile_error"
            }
        if self.llm_client is None:
            score, reason, details = self._heuristic(execution_result)
            details["semantic_mode_requested"] = "production_llm"
            details["semantic_fallback_reason"] = "missing llm client"
            return score, reason, details

        execution_summary = build_execution_summary(
            execution_result,
            max_chars=self.config.reward_semantic_max_feedback_chars,
        )
        prompt = build_production_semantic_score_prompt(problem, code, execution_summary)
        try:
            raw_response = self.llm_client.generate(
                prompt,
                temperature=self.config.llm_temperature,
                max_tokens=min(self.config.llm_max_tokens, 700),
            )
            parsed = parse_json_block(raw_response)
            requirement = _clamp(float(parsed.get("requirement_coverage", parsed.get("score", 0.0))))
            api = _clamp(float(parsed.get("api_contract_match", requirement)))
            context = _clamp(float(parsed.get("context_compatibility", 0.7)))
            maintainability = _clamp(float(parsed.get("maintainability", 0.7)))
            risk = _clamp(float(parsed.get("risk", 0.0)))
            confidence = _clamp(float(parsed.get("confidence", 0.5)))
            weighted = (
                0.35 * requirement
                + 0.25 * api
                + 0.20 * context
                + 0.20 * maintainability
            )
            score = _clamp(confidence * weighted * (1.0 - risk))
            reason = str(parsed.get("reason") or "production semantic score")
            return score, reason, {
                "semantic_mode": "production_llm",
                "semantic_prompt": prompt,
                "semantic_raw_response": raw_response,
                "semantic_parsed_json": parsed,
                "semantic_subscores": {
                    "requirement_coverage": requirement,
                    "api_contract_match": api,
                    "context_compatibility": context,
                    "maintainability": maintainability,
                    "risk": risk,
                    "confidence": confidence,
                },
            }
        except Exception as exc:
            score, reason, details = self._heuristic(execution_result)
            details["semantic_mode_requested"] = "production_llm"
            details["semantic_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            details["semantic_prompt"] = prompt
            return score, reason, details
