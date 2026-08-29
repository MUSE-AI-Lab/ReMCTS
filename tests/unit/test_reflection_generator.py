from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.memory.reflection_generator import ReflectionGenerator


class StaticLLM(BaseLLMClient):
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str, **kwargs) -> str:
        return self.response


def make_execution_result() -> ExecutionResult:
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


def test_reflection_generator_parses_llm_json():
    llm = StaticLLM(
        '{"error_type":"AssertionError","failure_summary":"second assert fails",'
        '"root_cause":"solution changes the input","fix_hint":"return x unchanged","severity":0.4}'
    )
    config = SearchConfig(reflection_mode="llm")
    generator = ReflectionGenerator(llm, config)

    record = generator.generate(
        ProblemSpec(task_id="t", prompt="return input"),
        "def solution(x): return x + 1",
        make_execution_result(),
        source_task_id="t",
        source_node_id="root",
    )

    assert record.error_type == "AssertionError"
    assert record.fix_hint == "return x unchanged"
    assert record.severity == 0.4
    assert record.metadata["reflection_mode"] == "llm"
    assert "llm_raw_response" in record.metadata


def test_reflection_generator_falls_back_on_bad_json():
    llm = StaticLLM("not json")
    config = SearchConfig(reflection_mode="llm", reflection_fallback_on_error=True)
    generator = ReflectionGenerator(llm, config)

    record = generator.generate(
        ProblemSpec(task_id="t", prompt="return input"),
        "def solution(x): return x + 1",
        make_execution_result(),
    )

    assert record.metadata["reflection_mode"] == "heuristic"
    assert record.metadata["reflection_mode_requested"] == "llm"
    assert "fallback_reason" in record.metadata
