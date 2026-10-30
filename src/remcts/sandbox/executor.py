from __future__ import annotations

from json import dumps
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from remcts.core.state import ExecutionResult

from .safety import truncate_output

RESULT_SENTINEL = "__REMCTS_RESULT__"


def count_asserts(tests: str) -> int:
    return sum(1 for line in tests.splitlines() if line.strip().startswith("assert "))


def _module_loader() -> list[str]:
    """Import ``candidate.py`` as a module (so ``__main__`` blocks do not run) and expose its names."""
    return [
        "from pathlib import Path as __remcts_path",
        "import importlib.util as __remcts_importlib_util",
        "import sys as __remcts_sys",
        "__remcts_candidate_path = __remcts_path(__file__).with_name('candidate.py')",
        "__remcts_spec = __remcts_importlib_util.spec_from_file_location('candidate', __remcts_candidate_path)",
        "__remcts_module = __remcts_importlib_util.module_from_spec(__remcts_spec)",
        "__remcts_sys.modules['candidate'] = __remcts_module",
        "assert __remcts_spec.loader is not None",
        "__remcts_spec.loader.exec_module(__remcts_module)",
        "globals().update({",
        "    __remcts_name: getattr(__remcts_module, __remcts_name)",
        "    for __remcts_name in dir(__remcts_module)",
        "    if not (__remcts_name.startswith('__') and __remcts_name.endswith('__'))",
        "})",
    ]


def _assert_harness(tests: str) -> str | None:
    """Run every top-level ``assert`` separately and report the pass count.

    Returns None for tests that cannot be split line by line (e.g. a HumanEval ``check`` function),
    which then run as one script.
    """
    if count_asserts(tests) <= 0 or "def check(candidate)" in tests:
        return None
    lines = [
        *_module_loader(),
        "import json as __remcts_json",
        "__remcts_passed = 0",
        "__remcts_failed = []",
    ]
    for line_no, line in enumerate(tests.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            lines.append("")
        elif stripped.startswith("#"):
            lines.append(line)
        elif line[:1].isspace() and not stripped.startswith("assert "):
            return None
        elif stripped.startswith("assert "):
            lines.extend(
                [
                    "try:",
                    f"    exec({dumps(stripped)}, globals())",
                    "    __remcts_passed += 1",
                    "except BaseException as __remcts_exc:",
                    (
                        "    __remcts_failed.append("
                        f"'line {line_no}: ' + type(__remcts_exc).__name__ + ': ' + str(__remcts_exc)"
                        ")"
                    ),
                ]
            )
        else:
            lines.append(line)
    lines.extend(
        [
            (
                f"print('{RESULT_SENTINEL}' + __remcts_json.dumps("
                "{'passed': __remcts_passed, 'failed': __remcts_failed}, ensure_ascii=False))"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _script_harness(tests: str) -> str:
    return "\n".join([*_module_loader(), "", tests, ""]) if tests else "\n".join([*_module_loader(), ""])


def _parse_harness_output(stdout: str) -> tuple[str, tuple[int, list[str]] | None]:
    """Split the harness result line from the candidate's own output."""
    result = None
    kept = []
    for line in stdout.splitlines():
        if line.startswith(RESULT_SENTINEL):
            data = json.loads(line[len(RESULT_SENTINEL):])
            result = int(data.get("passed", 0)), [str(item) for item in data.get("failed", [])]
        else:
            kept.append(line)
    return "\n".join(kept), result


class PythonExecutor:
    """Runs a candidate and its tests in a fresh interpreter subprocess with a timeout.

    This is process isolation only, not a security sandbox; run untrusted generations in a
    container or VM.
    """

    def __init__(self, python: str | None = None):
        self.python = python or sys.executable

    def run(
        self,
        code: str,
        tests: str = "",
        timeout_sec: int = 3,
        max_output_chars: int = 4000,
    ) -> ExecutionResult:
        total_tests = count_asserts(tests)
        try:
            compile(code, "<candidate>", "exec")
        except SyntaxError as exc:
            message = f"SyntaxError: {exc.msg} (line {exc.lineno})"
            return ExecutionResult(
                compile_ok=False,
                exec_ok=False,
                timed_out=False,
                stdout="",
                stderr=message,
                traceback=message,
                passed_tests=0,
                total_tests=total_tests,
                failed_cases=["compile_error"],
                coverage_hint={},
            )

        runner_source = (_assert_harness(tests) if tests else None) or _script_harness(tests)
        with tempfile.TemporaryDirectory(prefix="remcts-") as tmpdir:
            (Path(tmpdir) / "candidate.py").write_text(code, encoding="utf-8")
            runner = Path(tmpdir) / "runner.py"
            runner.write_text(runner_source, encoding="utf-8")
            try:
                result = subprocess.run(
                    [self.python, str(runner)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    compile_ok=True,
                    exec_ok=False,
                    timed_out=True,
                    stdout=truncate_output(exc.stdout or "", max_output_chars),
                    stderr=truncate_output(exc.stderr or "", max_output_chars),
                    traceback="TimeoutError: execution exceeded time budget",
                    passed_tests=0,
                    total_tests=total_tests,
                    failed_cases=["timeout"],
                    coverage_hint={},
                )

        own_stdout, harness_result = _parse_harness_output(result.stdout)
        stdout = truncate_output(own_stdout, max_output_chars)
        stderr = truncate_output(result.stderr, max_output_chars)
        exec_ok = result.returncode == 0
        if harness_result is not None:
            passed_tests, failed_cases = harness_result
        else:
            passed_tests = total_tests if exec_ok else 0
            failed_cases = [] if exec_ok else ["execution_failed"]
        if not exec_ok:
            traceback = stderr
        else:
            traceback = "\n".join(failed_cases)
        return ExecutionResult(
            compile_ok=True,
            exec_ok=exec_ok,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
            traceback=traceback,
            passed_tests=passed_tests,
            total_tests=total_tests,
            failed_cases=failed_cases,
            coverage_hint={"return_code": result.returncode},
        )
