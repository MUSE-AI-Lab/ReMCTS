from __future__ import annotations

import ast

from remcts.core.state import ProblemSpec, SearchConfig

from .evidence import EvidenceResult


def expected_entry_point(problem: ProblemSpec) -> str:
    return problem.entry_point or str(problem.metadata.get("entry_point") or "solution")


def check_contract(problem: ProblemSpec, code: str, config: SearchConfig) -> EvidenceResult:
    entry_point = expected_entry_point(problem)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return EvidenceResult(
            name="contract",
            score=0.0,
            confidence=config.production_contract_confidence,
            available=True,
            passed=False,
            details={"entry_point": entry_point, "error": f"SyntaxError: {exc.msg}"},
        )

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = [node.name for node in functions]
    target = next((node for node in functions if node.name == entry_point), None)
    if target is None:
        return EvidenceResult(
            name="contract",
            score=0.0,
            confidence=config.production_contract_confidence,
            available=True,
            passed=False,
            details={"entry_point": entry_point, "defined_functions": names},
        )

    args = target.args
    required_args = len(args.args) - len(args.defaults)
    vararg = args.vararg is not None
    kwarg = args.kwarg is not None
    score = 1.0
    if isinstance(target, ast.AsyncFunctionDef):
        score -= 0.25
    if required_args > 4 and not vararg:
        score -= 0.15
    return EvidenceResult(
        name="contract",
        score=max(0.0, score),
        confidence=config.production_contract_confidence,
        available=True,
        passed=score >= 0.75,
        details={
            "entry_point": entry_point,
            "defined_functions": names,
            "required_args": required_args,
            "has_vararg": vararg,
            "has_kwarg": kwarg,
            "is_async": isinstance(target, ast.AsyncFunctionDef),
        },
    )
