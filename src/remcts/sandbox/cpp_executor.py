from __future__ import annotations

from pathlib import Path
import re
import subprocess
import tempfile

from remcts.core.state import ExecutionResult

from .safety import truncate_output

_TEST_MARKERS = ("#undef NDEBUG", "\nint main(", "\nint main()")


def count_cpp_asserts(tests: str) -> int:
    return len(re.findall(r"\bassert\s*\(", tests))


def strip_cpp_main_or_tests(code: str) -> str:
    """Cut a generated ``main``/test section so the official test harness can be appended."""
    found = [index for index in (code.find(marker) for marker in _TEST_MARKERS) if index >= 0]
    return code[: min(found, default=len(code))].rstrip()


class CppExecutor:
    """Compiles ``candidate + tests`` with ``g++ -std=c++17 -O2`` and runs the binary.

    Assertions abort the program, so a run either passes all of them or none.
    """

    def run(
        self,
        code: str,
        tests: str = "",
        timeout_sec: int = 8,
        max_output_chars: int = 4000,
    ) -> ExecutionResult:
        total_tests = count_cpp_asserts(tests)
        source = f"{strip_cpp_main_or_tests(code).rstrip()}\n\n{tests.strip()}\n"

        def timeout_result(exc: subprocess.TimeoutExpired, stage: str) -> ExecutionResult:
            return ExecutionResult(
                compile_ok=stage == "run",
                exec_ok=False,
                timed_out=True,
                stdout=truncate_output(exc.stdout or "", max_output_chars),
                stderr=truncate_output(exc.stderr or "", max_output_chars),
                traceback=(
                    "CompileTimeout: g++ exceeded time budget"
                    if stage == "compile"
                    else "TimeoutError: execution exceeded time budget"
                ),
                passed_tests=0,
                total_tests=total_tests,
                failed_cases=["compile_timeout" if stage == "compile" else "timeout"],
                coverage_hint={"stage": stage},
            )

        with tempfile.TemporaryDirectory(prefix="remcts-cpp-") as tmpdir:
            source_path = Path(tmpdir) / "main.cpp"
            binary_path = Path(tmpdir) / "main"
            source_path.write_text(source, encoding="utf-8")
            try:
                command = ["g++", "-std=c++17", "-O2", str(source_path), "-o", str(binary_path)]
                compiled = _run(command, tmpdir, timeout_sec)
            except subprocess.TimeoutExpired as exc:
                return timeout_result(exc, "compile")
            if compiled.returncode != 0:
                stderr = truncate_output(compiled.stderr, max_output_chars)
                return ExecutionResult(
                    compile_ok=False,
                    exec_ok=False,
                    timed_out=False,
                    stdout=truncate_output(compiled.stdout, max_output_chars),
                    stderr=stderr,
                    traceback=stderr,
                    passed_tests=0,
                    total_tests=total_tests,
                    failed_cases=["compile_error"],
                    coverage_hint={"return_code": compiled.returncode, "stage": "compile"},
                )
            try:
                executed = _run([str(binary_path)], tmpdir, timeout_sec)
            except subprocess.TimeoutExpired as exc:
                return timeout_result(exc, "run")

        exec_ok = executed.returncode == 0
        stderr = truncate_output(executed.stderr, max_output_chars)
        return ExecutionResult(
            compile_ok=True,
            exec_ok=exec_ok,
            timed_out=False,
            stdout=truncate_output(executed.stdout, max_output_chars),
            stderr=stderr,
            traceback="" if exec_ok else stderr,
            passed_tests=total_tests if exec_ok else 0,
            total_tests=total_tests,
            failed_cases=[] if exec_ok else ["execution_failed"],
            coverage_hint={"return_code": executed.returncode, "stage": "run"},
        )


def _run(command: list[str], cwd: str, timeout_sec: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_sec,
    )
