from __future__ import annotations

import ast

from remcts.core.state import ProblemSpec, SearchConfig

from .evidence import EvidenceResult


def expected_entry_point(problem: ProblemSpec) -> str:
    return problem.entry_point or str(problem.metadata.get("entry_point") or "solution")


def entry_point_arity(code: str, entry_point: str) -> tuple[int, int, bool] | None:
    """``(required, positional, has_vararg)`` of the top-level entry function, or None if absent."""
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


def mentions_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def check_contract(problem: ProblemSpec, code: str, config: SearchConfig) -> EvidenceResult:
    """AST check that the entry function exists with a callable, non-async signature."""
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
    is_async = isinstance(target, ast.AsyncFunctionDef)
    score = 1.0
    if is_async:
        score -= 0.25
    if required_args > 4 and args.vararg is None:
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
            "has_vararg": args.vararg is not None,
            "has_kwarg": args.kwarg is not None,
            "is_async": is_async,
        },
    )
