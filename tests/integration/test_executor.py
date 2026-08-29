from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from remcts.core.state import ProblemSpec
from remcts.sandbox.safety import truncate_output


def test_executor_can_run_simple_asserts():
    code = "def solution(x):\n    return x\n"
    tests = "assert solution(1) == 1"
    result = PythonExecutor().run(code, tests=tests)
    assert result.compile_ok is True
    assert result.exec_ok is True
    assert result.passed_tests == 1


def test_executor_counts_partial_asserts():
    code = "def solution(x):\n    return x\n"
    tests = "assert solution(1) == 1\nassert solution(2) == 3"
    result = PythonExecutor().run(code, tests=tests)
    assert result.compile_ok is True
    assert result.exec_ok is True
    assert result.passed_tests == 1
    assert result.total_tests == 2
    assert result.failed_cases


def test_executor_imports_candidate_without_running_main_block():
    code = "\n".join(
        [
            "def solution(x):",
            "    return x",
            "",
            "if __name__ == '__main__':",
            "    raise RuntimeError('main block should not run during validation')",
        ]
    )

    result = PythonExecutor().run(code, tests="assert solution(1) == 1")

    assert result.compile_ok is True
    assert result.exec_ok is True
    assert result.passed_tests == 1


def test_executor_no_tests_only_imports_candidate_module():
    code = "\n".join(
        [
            "def solution(nums=None):",
            "    if nums is None:",
            "        data = list(map(int, sys.stdin.buffer.read().split()))",
            "        return data",
            "    return nums",
            "",
            "if __name__ == '__main__':",
            "    print(solution())",
        ]
    )

    result = PythonExecutor().run(code)

    assert result.compile_ok is True
    assert result.exec_ok is True
    assert result.total_tests == 0


def test_run_problem_tests_calls_humaneval_check_function():
    problem = ProblemSpec(
        task_id="HumanEval/test",
        prompt="def solution(x):\n",
        entry_point="solution",
        tests="\n".join(
            [
                "def check(candidate):",
                "    assert candidate(1) == 2",
                "",
            ]
        ),
    )
    code = "def solution(x):\n    return x\n"

    result = run_problem_tests(problem, code, PythonExecutor(), timeout_sec=3, max_output_chars=4000)

    assert result.compile_ok is True
    assert result.exec_ok is False
    assert result.passed_tests == 0
    assert result.total_tests == 1
    assert "IndentationError" not in result.traceback


def test_run_problem_tests_accepts_passing_humaneval_check_function():
    problem = ProblemSpec(
        task_id="HumanEval/test",
        prompt="def solution(x):\n",
        entry_point="solution",
        tests="\n".join(
            [
                "def check(candidate):",
                "    assert candidate(1) == 1",
                "    assert candidate(2) == 2",
                "",
            ]
        ),
    )
    code = "def solution(x):\n    return x\n"

    result = run_problem_tests(problem, code, PythonExecutor(), timeout_sec=3, max_output_chars=4000)

    assert result.compile_ok is True
    assert result.exec_ok is True
    assert result.passed_tests == 2
    assert result.total_tests == 2


def test_truncate_output_normalizes_bytes():
    assert truncate_output(b"abc\xff", 20) == "abc\ufffd"
