from __future__ import annotations

import ast
from typing import Any

from remcts.core.state import ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.llm.response_parser import parse_json_block
from remcts.sandbox.executor import PythonExecutor

from .contract_checker import expected_entry_point
from .evidence import EvidenceResult


def _arg_counts(code: str, entry_point: str) -> tuple[int, int, bool] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            positional = len(node.args.posonlyargs) + len(node.args.args)
            required = positional - len(node.args.defaults)
            return required, positional, node.args.vararg is not None
    return None


def build_probe_tests(
    problem: ProblemSpec,
    code: str,
    config: SearchConfig | None = None,
    llm_client: BaseLLMClient | None = None,
) -> str:
    entry_point = expected_entry_point(problem)
    arg_counts = _arg_counts(code, entry_point)
    if arg_counts is None:
        return ""
    required_args, positional_args, has_vararg = arg_counts
    lines = [
        f"assert callable({entry_point})",
    ]
    problem_probes = _problem_specific_probe_tests(problem, entry_point, positional_args, has_vararg)
    if problem_probes:
        lines.extend(problem_probes)
    llm_probes = _llm_probe_tests(problem, entry_point, config, llm_client)
    if llm_probes:
        lines.extend(llm_probes)
    elif positional_args == 0 and required_args == 0:
        lines.append(f"{entry_point}()")
    elif not problem_probes and 0 < required_args <= 3:
        args = ", ".join("0" for _ in range(required_args))
        lines.append(f"{entry_point}({args})")
    return "\n".join(lines)


def _llm_probe_tests(
    problem: ProblemSpec,
    entry_point: str,
    config: SearchConfig | None,
    llm_client: BaseLLMClient | None,
) -> list[str]:
    if config is None or not config.production_probe_llm_tests or llm_client is None:
        return []
    cache_key = "_remcts_llm_probe_tests"
    cached = problem.metadata.get(cache_key)
    if isinstance(cached, list):
        return [str(item) for item in cached]

    prompt = f"""
你是一个保守的 Python 单元测试生成器。请根据题目描述，为入口函数 `{entry_point}` 生成 {config.production_probe_llm_count} 条弱测试。

要求：
1. 只使用 Python assert 语句；
2. 只调用入口函数 `{entry_point}`；
3. 使用简单字面量输入，不要 import、不要随机数、不要文件/网络/系统调用；
4. 测试应覆盖常见情况和 1-2 个边界情况；
5. 不要根据某个候选代码生成测试，只根据题目描述推导期望输出；
6. 只输出 JSON：{{"tests": ["assert ...", "..."]}}

题目描述：
{problem.prompt}
""".strip()
    tests: list[str] = []
    try:
        raw = llm_client.generate(
            prompt,
            temperature=config.llm_temperature,
            max_tokens=min(config.llm_max_tokens, 700),
        )
        parsed = parse_json_block(raw)
        tests = _extract_probe_tests(parsed, entry_point, config.production_probe_llm_count)
    except Exception as exc:
        problem.metadata[cache_key] = []
        problem.metadata["_remcts_llm_probe_error"] = f"{type(exc).__name__}: {exc}"
        return []
    problem.metadata[cache_key] = tests
    return tests


def _extract_probe_tests(parsed: dict[str, Any], entry_point: str, limit: int) -> list[str]:
    raw_tests = parsed.get("tests", [])
    if not isinstance(raw_tests, list):
        return []
    tests: list[str] = []
    seen: set[str] = set()
    for item in raw_tests:
        sanitized = _sanitize_probe_test(str(item), entry_point)
        if sanitized is None or sanitized in seen:
            continue
        seen.add(sanitized)
        tests.append(sanitized)
        if len(tests) >= max(1, limit):
            break
    return tests


def _sanitize_probe_test(line: str, entry_point: str) -> str | None:
    stripped = line.strip()
    if "\n" in stripped or "\r" in stripped:
        return None
    if not stripped.startswith("assert "):
        return None
    if f"{entry_point}(" not in stripped:
        return None
    lowered = stripped.lower()
    blocked = [
        "import ",
        "__",
        "open(",
        "input(",
        "eval(",
        "exec(",
        "compile(",
        "os.",
        "sys.",
        "subprocess",
        "socket",
        "requests",
        "urllib",
    ]
    if any(token in lowered for token in blocked):
        return None
    try:
        ast.parse(stripped)
    except SyntaxError:
        return None
    return stripped


def _problem_specific_probe_tests(
    problem: ProblemSpec,
    entry_point: str,
    positional_args: int,
    has_vararg: bool,
) -> list[str]:
    prompt = problem.prompt.lower()
    can_pass_one_arg = positional_args >= 1 or has_vararg
    if not can_pass_one_arg:
        return []
    if _has_any(prompt, ["mex", "minimum excluded", "minimum excludant", "最小未出现", "最小的非负整数"]):
        return [
            f"assert {entry_point}([0, 1, 2]) == 3",
            f"assert {entry_point}([1, 2, 3]) == 0",
            f"assert {entry_point}([0, 2, 3]) == 1",
            f"assert {entry_point}([]) == 0",
        ]
    return []


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def run_probe_tests(
    problem: ProblemSpec,
    code: str,
    executor: PythonExecutor,
    config: SearchConfig,
    llm_client: BaseLLMClient | None = None,
) -> EvidenceResult:
    probe_tests = build_probe_tests(problem, code, config=config, llm_client=llm_client)
    if not probe_tests:
        return EvidenceResult(
            name="probe",
            score=0.0,
            confidence=0.0,
            available=False,
            passed=None,
            details={"reason": "no callable entry point found for probes"},
        )

    result = executor.run(
        code,
        tests=probe_tests,
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )
    if result.total_tests > 0:
        unit_score = result.passed_tests / result.total_tests
    else:
        unit_score = 1.0 if result.exec_ok else 0.0
    score = unit_score if result.exec_ok else 0.0
    return EvidenceResult(
        name="probe",
        score=score,
        confidence=config.production_probe_confidence,
        available=True,
        passed=result.exec_ok and score >= 1.0,
        details={
            "probe_tests": probe_tests,
            "execution": result.to_dict(),
        },
    )
