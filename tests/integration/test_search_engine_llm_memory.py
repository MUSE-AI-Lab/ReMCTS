from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient


class RepairLLM(BaseLLMClient):
    def __init__(self):
        self.code_calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        if "语义奖励评估器" in prompt:
            return '{"score":0.35,"reason":"wrong return value but close structure"}'
        if "失败反思模块" in prompt:
            return (
                '{"error_type":"AssertionError","failure_summary":"wrong value",'
                '"root_cause":"solution returns x + 1","fix_hint":"return x unchanged","severity":0.6}'
            )
        self.code_calls += 1
        if self.code_calls == 1:
            return "```python\ndef solution(x):\n    return x + 1\n```"
        return "```python\ndef solution(x):\n    return x\n```"


class AlwaysCorrectLLM(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        return "```python\ndef solution(x):\n    return x\n```"


def test_search_uses_llm_reflection_and_semantic_score(tmp_path):
    config = SearchConfig(
        max_iterations=4,
        branch_factor=1,
        max_depth=4,
        reflection_mode="llm",
        reward_semantic_mode="llm",
        run_output_dir=str(tmp_path),
    )
    result = SearchEngine(config, llm_client=RepairLLM()).run(
        ProblemSpec(
            task_id="repair",
            prompt="Return the input value unchanged.",
            tests="assert solution(1) == 1",
        )
    )

    assert result.success is True
    assert result.iterations == 2
    memory_files = list(tmp_path.glob("repair-*/memory.jsonl"))
    assert memory_files
    memory_text = memory_files[0].read_text(encoding="utf-8")
    assert '"reflection_mode": "llm"' in memory_text
    assert "return x unchanged" in memory_text
    node_text = list(tmp_path.glob("repair-*/nodes.jsonl"))[0].read_text(encoding="utf-8")
    assert '"semantic_mode": "llm"' in node_text


def test_benchmark_search_respects_min_accept_iterations(tmp_path):
    config = SearchConfig(
        max_iterations=4,
        branch_factor=1,
        max_depth=4,
        production_complete_branch_before_accept=True,
        production_min_accept_iterations=3,
        run_output_dir=str(tmp_path),
    )

    result = SearchEngine(config, llm_client=AlwaysCorrectLLM()).run(
        ProblemSpec(
            task_id="min-rounds",
            prompt="Return the input value unchanged.",
            tests="assert solution(1) == 1",
        )
    )

    assert result.success is True
    assert result.accepted is True
    assert result.iterations == 3
    assert result.stop_reason == "tests_passed"
