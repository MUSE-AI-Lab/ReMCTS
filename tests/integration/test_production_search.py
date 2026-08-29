from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient


class ProductionLLM(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        if "生产环境代码验收评审器" in prompt:
            return (
                '{"requirement_coverage":0.95,"api_contract_match":0.95,'
                '"context_compatibility":0.9,"maintainability":0.9,'
                '"risk":0.0,"confidence":0.9,"reason":"simple identity implementation"}'
            )
        return "```python\ndef solution(x):\n    return x\n```"


def test_production_search_accepts_without_testcases(tmp_path):
    config = SearchConfig(
        evaluation_mode="production",
        oracle_policy="no_oracle",
        reward_semantic_mode="llm",
        reflection_mode="heuristic",
        max_iterations=3,
        branch_factor=1,
        run_output_dir=str(tmp_path),
        production_accept_threshold=0.62,
        production_semantic_threshold=0.45,
    )

    result = SearchEngine(config, llm_client=ProductionLLM()).run(
        ProblemSpec(
            task_id="prod",
            prompt="Write a Python function named solution(x) that returns the input unchanged.",
            entry_point="solution",
        )
    )

    assert result.success is False
    assert result.accepted is True
    assert result.needs_review is False
    assert result.stop_reason == "accepted_by_confidence"

    run_dir = next(tmp_path.glob("prod-*"))
    summary = (run_dir / "summary.json").read_text(encoding="utf-8")
    nodes = (run_dir / "nodes.jsonl").read_text(encoding="utf-8")
    assert '"accepted": true' in summary
    assert '"evaluation_mode": "production"' in nodes
    assert '"accept_score"' in nodes
