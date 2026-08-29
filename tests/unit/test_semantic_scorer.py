from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.reward.scorer import score_candidate
from remcts.reward.semantic_scorer import SemanticScorer
from remcts.sandbox.executor import PythonExecutor


class StaticLLM(BaseLLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


def make_partial_execution() -> ExecutionResult:
    return ExecutionResult(
        compile_ok=True,
        exec_ok=True,
        timed_out=False,
        stdout="",
        stderr="",
        traceback="line 2: AssertionError",
        passed_tests=1,
        total_tests=2,
        failed_cases=["line 2: AssertionError"],
    )


def test_semantic_scorer_parses_llm_json():
    scorer = SemanticScorer(
        StaticLLM('{"score":0.73,"reason":"mostly right"}'),
        SearchConfig(reward_semantic_mode="llm"),
    )

    score, reason, details = scorer.score(
        ProblemSpec(task_id="t", prompt="return input"),
        "def solution(x): return x",
        make_partial_execution(),
    )

    assert score == 0.73
    assert reason == "mostly right"
    assert details["semantic_mode"] == "llm"


def test_semantic_scorer_clamps_score():
    scorer = SemanticScorer(
        StaticLLM('{"score":1.8,"reason":"too high"}'),
        SearchConfig(reward_semantic_mode="llm"),
    )

    score, _, _ = scorer.score(
        ProblemSpec(task_id="t", prompt="return input"),
        "def solution(x): return x",
        make_partial_execution(),
    )

    assert score == 1.0


def test_score_candidate_uses_semantic_scorer():
    config = SearchConfig(
        reward_semantic_mode="llm",
        w_unit=0.0,
        w_exec=0.0,
        w_semantic=1.0,
    )
    scorer = SemanticScorer(StaticLLM('{"score":0.42,"reason":"partial"}'), config)

    result = score_candidate(
        make_partial_execution(),
        config,
        problem=ProblemSpec(task_id="t", prompt="return input"),
        code="def solution(x): return x",
        semantic_scorer=scorer,
    )

    assert result.r_semantic == 0.42
    assert result.total_reward == 0.42
    assert result.details["semantic_mode"] == "llm"
    assert "benchmark_evidence" in result.details
    assert {item["name"] for item in result.details["benchmark_evidence"]} >= {"unit", "exec"}


def test_score_candidate_production_uses_unified_evidence_bundle():
    config = SearchConfig(
        evaluation_mode="production",
        reward_semantic_mode="llm",
        production_accept_threshold=0.62,
        production_semantic_threshold=0.45,
    )
    scorer = SemanticScorer(
        StaticLLM(
            '{"requirement_coverage":0.95,"api_contract_match":0.95,'
            '"context_compatibility":0.9,"maintainability":0.9,'
            '"risk":0.0,"confidence":0.9,"reason":"good"}'
        ),
        config,
    )
    execution = ExecutionResult(
        compile_ok=True,
        exec_ok=True,
        timed_out=False,
        stdout="",
        stderr="",
        traceback="",
        passed_tests=0,
        total_tests=0,
    )

    result = score_candidate(
        execution,
        config,
        problem=ProblemSpec(
            task_id="p",
            prompt="Write a Python function named solution(x) that returns the input unchanged.",
            entry_point="solution",
        ),
        code="def solution(x):\n    return x\n",
        semantic_scorer=scorer,
        executor=PythonExecutor(),
    )

    assert result.details["evaluation_mode"] == "production"
    assert result.details["accepted"] is True
    assert {item["name"] for item in result.details["all_evidence"]} >= {
        "unit",
        "exec",
        "contract",
        "meta",
        "probe",
    }
    assert {item["name"] for item in result.details["evidence"]} == {
        "contract",
        "static",
        "meta",
        "probe",
    }
