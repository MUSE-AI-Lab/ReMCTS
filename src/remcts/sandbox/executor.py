from __future__ import annotations

import subprocess
import tempfile
import json
from json import dumps
from pathlib import Path

from remcts.core.state import ExecutionResult

from .safety import truncate_output


RESULT_SENTINEL = "__REMCTS_RESULT__"


def _estimate_total_tests(tests: str) -> int:
    count = 0
    for line in tests.splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            count += 1
    return count


def _build_module_loader() -> list[str]:
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


def _build_import_harness() -> str:
    return "\n".join(_build_module_loader() + [""])


def _build_assert_harness(tests: str) -> str | None:
    total_tests = _estimate_total_tests(tests)
    if total_tests <= 0:
        return None
    if "def check(candidate)" in tests:
        return None

    lines = [
        *_build_module_loader(),
        "import json as __remcts_json",
        "__remcts_passed = 0",
        "__remcts_failed = []",
    ]
    for line_no, line in enumerate(tests.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("#"):
            lines.append(line)
            continue
        if line[:1].isspace() and not stripped.startswith("assert "):
            return None
        if stripped.startswith("assert "):
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


def _extract_harness_result(stdout: str) -> tuple[int, list[str]] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            payload = line[len(RESULT_SENTINEL) :]
            data = json.loads(payload)
            return int(data.get("passed", 0)), [str(item) for item in data.get("failed", [])]
    return None


def _strip_harness_result(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if not line.startswith(RESULT_SENTINEL)]
    return "\n".join(lines)


class PythonExecutor:
    def run(
        self,
        code: str,
        tests: str = "",
        timeout_sec: int = 3,
        max_output_chars: int = 4000,
    ) -> ExecutionResult:
        total_tests = _estimate_total_tests(tests)
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

        harness = _build_assert_harness(tests) if tests else None
        combined = harness or (_build_import_harness() if not tests else _build_fallback_harness(tests))
        with tempfile.TemporaryDirectory(prefix="remcts-") as tmpdir:
            candidate = Path(tmpdir) / "candidate.py"
            runner = Path(tmpdir) / "runner.py"
            candidate.write_text(code, encoding="utf-8")
            runner.write_text(combined, encoding="utf-8")
            try:
                result = subprocess.run(
                    ["python3", str(runner)],
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
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

        harness_result = _extract_harness_result(result.stdout)
        stdout = truncate_output(_strip_harness_result(result.stdout), max_output_chars)
        stderr = truncate_output(result.stderr, max_output_chars)
        exec_ok = result.returncode == 0
        failed_cases = [] if exec_ok else ["execution_failed"]
        if harness_result is not None:
            passed_tests, failed_cases = harness_result
        else:
            passed_tests = total_tests if exec_ok else 0
        traceback = stderr if not exec_ok else ""
        if exec_ok and failed_cases:
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


def _build_fallback_harness(tests: str) -> str:
    return "\n".join(
        [
            *_build_module_loader(),
            "",
            tests,
            "",
        ]
    )
