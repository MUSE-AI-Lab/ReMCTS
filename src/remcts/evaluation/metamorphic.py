from __future__ import annotations

from dataclasses import dataclass

from remcts.core.state import ProblemSpec, SearchConfig
from remcts.sandbox.executor import PythonExecutor

from .contract_checker import entry_point_arity, expected_entry_point, mentions_any
from .evidence import EvidenceResult


@dataclass(slots=True)
class MetamorphicSuite:
    names: list[str]
    tests: list[str]

    def render(self) -> str:
        return "\n".join(self.tests)



def build_metamorphic_suite(problem: ProblemSpec, code: str) -> MetamorphicSuite:
    prompt = problem.prompt.lower()
    entry = expected_entry_point(problem)
    arg_counts = entry_point_arity(code, entry)
    names: list[str] = []
    tests: list[str] = []
    if arg_counts is None:
        return MetamorphicSuite(names, tests)
    required_args, positional_args, has_vararg = arg_counts

    if positional_args >= 1 or has_vararg:
        if mentions_any(prompt, ["unchanged", "same value", "identity", "return input", "returns the input", "原样", "不变"]):
            names.append("identity")
            tests.extend(
                [
                    f"assert {entry}(0) == 0",
                    f"assert {entry}('abc') == 'abc'",
                    f"assert {entry}([1, 2, 3]) == [1, 2, 3]",
                ]
            )
        if mentions_any(prompt, ["uppercase", "upper case", "upper-case", "大写"]):
            names.append("uppercase")
            tests.append(f"assert {entry}('aBc') == 'ABC'")
        if mentions_any(prompt, ["lowercase", "lower case", "lower-case", "小写"]):
            names.append("lowercase")
            tests.append(f"assert {entry}('aBc') == 'abc'")
        if mentions_any(prompt, ["reverse", "reversed", "反转", "倒序"]):
            names.append("reverse")
            tests.extend(
                [
                    f"assert {entry}('abc') == 'cba'",
                    f"assert {entry}([1, 2, 3]) == [3, 2, 1]",
                ]
            )
        if mentions_any(prompt, ["sort", "sorted", "ascending", "升序", "排序"]):
            names.append("sorted")
            tests.extend(
                [
                    f"assert {entry}([3, 1, 2]) == [1, 2, 3]",
                    f"assert len({entry}([3, 1, 2])) == 3",
                ]
            )
        if mentions_any(
            prompt,
            [
                "return length",
                "returns the length",
                "compute length",
                "number of characters",
                "length of",
                "字符数",
                "返回长度",
                "求长度",
                "字符串长度",
            ],
        ):
            names.append("length")
            tests.extend(
                [
                    f"assert {entry}('') == 0",
                    f"assert {entry}('abc') == 3",
                ]
            )
        if mentions_any(prompt, ["palindrome", "回文"]):
            names.append("palindrome")
            tests.extend(
                [
                    f"assert {entry}('aba') is True",
                    f"assert {entry}('abc') is False",
                ]
            )
        if mentions_any(prompt, ["even", "偶数"]):
            names.append("even")
            tests.extend(
                [
                    f"assert {entry}(2) is True",
                    f"assert {entry}(3) is False",
                ]
            )
        if mentions_any(prompt, ["odd", "奇数"]):
            names.append("odd")
            tests.extend(
                [
                    f"assert {entry}(3) is True",
                    f"assert {entry}(2) is False",
                ]
            )
        if mentions_any(prompt, ["mex", "minimum excluded", "minimum excludant", "最小未出现", "最小的非负整数"]):
            names.append("mex")
            tests.extend(
                [
                    f"assert {entry}([]) == 0",
                    f"assert {entry}([0]) == 1",
                    f"assert {entry}([1]) == 0",
                    f"assert {entry}([0, 1, 2]) == 3",
                    f"assert {entry}([0, 2, 3]) == 1",
                    f"assert {entry}([1, 2, 3]) == 0",
                    f"assert {entry}([0, 0, 1, 4]) == 2",
                    f"assert {entry}([2, 1000000, 0]) == 1",
                ]
            )
    if required_args == 2:
        if mentions_any(prompt, ["add", "sum", "addition", "相加", "求和"]):
            names.append("add")
            tests.extend(
                [
                    f"assert {entry}(1, 2) == 3",
                    f"assert {entry}(-1, 1) == 0",
                ]
            )
        if mentions_any(prompt, ["multiply", "product", "乘", "乘积"]):
            names.append("multiply")
            tests.extend(
                [
                    f"assert {entry}(2, 3) == 6",
                    f"assert {entry}(-1, 4) == -4",
                ]
            )
        if mentions_any(prompt, ["maximum", "max", "larger", "最大"]):
            names.append("max")
            tests.extend(
                [
                    f"assert {entry}(2, 3) == 3",
                    f"assert {entry}(-1, -4) == -1",
                ]
            )
    return MetamorphicSuite(_dedupe(names), _dedupe(tests))


def run_metamorphic_tests(
    problem: ProblemSpec,
    code: str,
    executor: PythonExecutor,
    config: SearchConfig,
) -> EvidenceResult:
    suite = build_metamorphic_suite(problem, code)
    if not suite.tests:
        return EvidenceResult(
            name="meta",
            score=0.0,
            confidence=0.0,
            available=False,
            passed=None,
            details={"reason": "no metamorphic properties extracted"},
        )

    result = executor.run(
        code,
        tests=suite.render(),
        timeout_sec=config.timeout_sec,
        max_output_chars=config.max_output_chars,
    )
    score = result.passed_tests / result.total_tests if result.total_tests > 0 else 0.0
    return EvidenceResult(
        name="meta",
        score=score if result.exec_ok else 0.0,
        confidence=config.production_meta_confidence,
        available=True,
        passed=result.exec_ok and score >= 1.0,
        details={
            "properties": suite.names,
            "metamorphic_tests": suite.render(),
            "execution": result.to_dict(),
        },
    )



def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
