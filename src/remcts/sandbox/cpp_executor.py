from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from remcts.core.state import ExecutionResult

from .safety import truncate_output


def estimate_cpp_asserts(tests: str) -> int:
    return len(re.findall(r"\bassert\s*\(", tests))


def strip_cpp_main_or_tests(code: str) -> str:
    markers = ["#undef NDEBUG", "\nint main(", "\nint main()"]
    cut = len(code)
    for marker in markers:
        idx = code.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    return code[:cut].rstrip()


class CppExecutor:
    def run(
        self,
        code: str,
        tests: str = "",
        timeout_sec: int = 8,
        max_output_chars: int = 4000,
    ) -> ExecutionResult:
        total_tests = estimate_cpp_asserts(tests)
        candidate_code = strip_cpp_main_or_tests(code)
        source = f"{candidate_code.rstrip()}\n\n{tests.strip()}\n"

        with tempfile.TemporaryDirectory(prefix="remcts-cpp-") as tmpdir:
            tmp = Path(tmpdir)
            source_path = tmp / "main.cpp"
            binary_path = tmp / "main"
            source_path.write_text(source, encoding="utf-8")

            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O2",
                str(source_path),
                "-o",
                str(binary_path),
            ]
            try:
                compile_result = subprocess.run(
                    compile_cmd,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
            except subprocess.TimeoutExpired as exc:
                return ExecutionResult(
                    compile_ok=False,
                    exec_ok=False,
                    timed_out=True,
                    stdout=truncate_output(exc.stdout or "", max_output_chars),
                    stderr=truncate_output(exc.stderr or "", max_output_chars),
                    traceback="CompileTimeout: g++ exceeded time budget",
                    passed_tests=0,
                    total_tests=total_tests,
                    failed_cases=["compile_timeout"],
                    coverage_hint={"stage": "compile"},
                )

            if compile_result.returncode != 0:
                stderr = truncate_output(compile_result.stderr, max_output_chars)
                return ExecutionResult(
                    compile_ok=False,
                    exec_ok=False,
                    timed_out=False,
                    stdout=truncate_output(compile_result.stdout, max_output_chars),
                    stderr=stderr,
                    traceback=stderr,
                    passed_tests=0,
                    total_tests=total_tests,
                    failed_cases=["compile_error"],
                    coverage_hint={"return_code": compile_result.returncode, "stage": "compile"},
                )

            try:
                run_result = subprocess.run(
                    [str(binary_path)],
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
                    coverage_hint={"stage": "run"},
                )

        exec_ok = run_result.returncode == 0
        stderr = truncate_output(run_result.stderr, max_output_chars)
        stdout = truncate_output(run_result.stdout, max_output_chars)
        failed_cases = [] if exec_ok else ["execution_failed"]
        passed_tests = total_tests if exec_ok else 0
        return ExecutionResult(
            compile_ok=True,
            exec_ok=exec_ok,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
            traceback="" if exec_ok else stderr,
            passed_tests=passed_tests,
            total_tests=total_tests,
            failed_cases=failed_cases,
            coverage_hint={"return_code": run_result.returncode, "stage": "run"},
        )
