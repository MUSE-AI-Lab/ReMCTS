from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.evaluation.contract_checker import check_contract
from remcts.evaluation.metamorphic import build_metamorphic_suite, run_metamorphic_tests
from remcts.evaluation.probe_generator import build_probe_tests, run_probe_tests
from remcts.reward.proxy_score import score_proxy
from remcts.reward.risk_score import score_risk
from remcts.sandbox.executor import PythonExecutor


def make_execution_result() -> ExecutionResult:
    return ExecutionResult(
        compile_ok=True,
        exec_ok=True,
        timed_out=False,
        stdout="",
        stderr="",
        traceback="",
        passed_tests=0,
        total_tests=0,
    )


def test_contract_checker_accepts_expected_entry_point():
    result = check_contract(
        ProblemSpec(task_id="p", prompt="Return input", entry_point="solution"),
        "def solution(x):\n    return x\n",
        SearchConfig(evaluation_mode="production"),
    )

    assert result.available is True
    assert result.passed is True
    assert result.score == 1.0


def test_proxy_score_uses_contract_static_and_probe_without_tests():
    score, confidence, evidences = score_proxy(
        ProblemSpec(task_id="p", prompt="Return the input unchanged", entry_point="solution"),
        "def solution(x):\n    return x\n",
        make_execution_result(),
        SearchConfig(evaluation_mode="production"),
        executor=PythonExecutor(),
    )

    assert score > 0.8
    assert confidence > 0.4
    assert {item.name for item in evidences} == {"contract", "static", "meta", "probe"}
    meta = next(item for item in evidences if item.name == "meta")
    assert meta.passed is True
    assert "identity" in meta.details["properties"]


def test_risk_score_detects_dangerous_calls():
    result = score_risk(
        "import os\n\ndef solution(x):\n    os.system('rm -rf /tmp/example')\n    return x\n",
        SearchConfig(evaluation_mode="production"),
    )

    assert result.score > 0.0
    assert result.passed is False
    assert result.details["findings"]


def test_metamorphic_suite_extracts_identity_properties():
    suite = build_metamorphic_suite(
        ProblemSpec(task_id="p", prompt="Return the input unchanged", entry_point="solution"),
        "def solution(x):\n    return x\n",
    )

    assert "identity" in suite.names
    assert "assert solution(0) == 0" in suite.tests


def test_metamorphic_evidence_fails_wrong_identity_code():
    result = run_metamorphic_tests(
        ProblemSpec(task_id="p", prompt="Return the input unchanged", entry_point="solution"),
        "def solution(x):\n    return x + 1\n",
        PythonExecutor(),
        SearchConfig(evaluation_mode="production"),
    )

    assert result.available is True
    assert result.passed is False
    assert result.score < 1.0
    assert "identity" in result.details["properties"]


def test_metamorphic_evidence_supports_sort_property():
    result = run_metamorphic_tests(
        ProblemSpec(task_id="p", prompt="Sort the input list in ascending order", entry_point="solution"),
        "def solution(x):\n    return sorted(x)\n",
        PythonExecutor(),
        SearchConfig(evaluation_mode="production"),
    )

    assert result.passed is True
    assert result.score == 1.0
    assert "sorted" in result.details["properties"]


def test_metamorphic_evidence_supports_mex_property_with_optional_arg():
    result = run_metamorphic_tests(
        ProblemSpec(
            task_id="p",
            prompt="给你一个长度为n的非负整数数组，求这个数组的mex",
            entry_point="solution",
        ),
        "def solution(nums=None):\n"
        "    if nums is None:\n"
        "        return None\n"
        "    n = len(nums)\n"
        "    seen = set(nums)\n"
        "    i = 0\n"
        "    while i in seen:\n"
        "        i += 1\n"
        "    return i\n",
        PythonExecutor(),
        SearchConfig(evaluation_mode="production"),
    )

    assert result.passed is True
    assert result.score == 1.0
    assert "mex" in result.details["properties"]


def test_mex_probe_prefers_array_calls_over_optional_no_arg_call():
    problem = ProblemSpec(
        task_id="p",
        prompt="给你一个长度为n的非负整数数组，求这个数组的mex",
        entry_point="solution",
    )
    code = "def solution(nums=None):\n    return 0 if nums != [0] else 1\n"

    tests = build_probe_tests(problem, code)

    assert "solution()" not in tests
    assert "solution([0, 1, 2])" in tests


def test_mex_probe_runs_against_candidate_with_stdin_fallback():
    problem = ProblemSpec(
        task_id="p",
        prompt="给你一个长度为n的非负整数数组，求这个数组的mex",
        entry_point="solution",
    )
    code = "\n".join(
        [
            "def solution(nums=None):",
            "    if nums is None:",
            "        data = list(map(int, sys.stdin.buffer.read().split()))",
            "        nums = data[1:]",
            "    n = len(nums)",
            "    i = 0",
            "    while i < n:",
            "        v = nums[i]",
            "        if 0 <= v < n and nums[v] != v:",
            "            nums[i], nums[v] = nums[v], nums[i]",
            "        else:",
            "            i += 1",
            "    for i in range(n):",
            "        if nums[i] != i:",
            "            return i",
            "    return n",
        ]
    )

    result = run_probe_tests(
        problem,
        code,
        PythonExecutor(),
        SearchConfig(evaluation_mode="production"),
    )

    assert result.passed is True
    assert result.score == 1.0
