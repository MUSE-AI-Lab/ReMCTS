from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.evaluation.evidence import EvidenceResult
from remcts.evaluation.evidence_builder import build_evidence_bundle, weighted_average
from remcts.sandbox.executor import PythonExecutor


PROXY_WEIGHTS = {
    "contract": 0.35,
    "probe": 0.25,
    "meta": 0.15,
    "static": 0.25,
}


def build_static_evidence(execution_result: ExecutionResult, config: SearchConfig) -> EvidenceResult:
    if not execution_result.compile_ok:
        score = 0.0
    elif execution_result.timed_out:
        score = 0.45
    elif execution_result.exec_ok:
        score = 1.0
    else:
        score = 0.65
    return EvidenceResult(
        name="static",
        score=score,
        confidence=config.production_static_confidence,
        available=True,
        passed=score >= 0.75,
        details={
            "compile_ok": execution_result.compile_ok,
            "exec_ok": execution_result.exec_ok,
            "timed_out": execution_result.timed_out,
        },
    )


def build_meta_evidence(
    problem: ProblemSpec,
    code: str,
    config: SearchConfig,
    executor: PythonExecutor | None = None,
) -> EvidenceResult:
    if executor is not None:
        metamorphic = run_metamorphic_tests(problem, code, executor, config)
        if metamorphic.available:
            return metamorphic

    prompt_terms = [term.lower() for term in problem.prompt.split() if len(term) >= 5]
    if not prompt_terms:
        return EvidenceResult(
            name="meta",
            score=0.0,
            confidence=0.0,
            available=False,
            passed=None,
            details={"reason": "no useful prompt terms"},
        )
    code_lower = code.lower()
    hits = sum(1 for term in set(prompt_terms) if term in code_lower)
    score = min(1.0, hits / max(min(len(set(prompt_terms)), 6), 1))
    return EvidenceResult(
        name="meta",
        score=score,
        confidence=config.production_meta_confidence,
        available=True,
        passed=score >= 0.35,
        details={
            "matched_terms": hits,
            "total_terms": len(set(prompt_terms)),
            "fallback": "keyword_overlap",
        },
    )


def score_proxy(
    problem: ProblemSpec,
    code: str,
    execution_result: ExecutionResult,
    config: SearchConfig,
    executor: PythonExecutor | None = None,
) -> tuple[float, float, list[EvidenceResult]]:
    bundle = build_evidence_bundle(problem, code, execution_result, config, executor=executor)
    evidences: list[EvidenceResult] = []
    if bundle.contract is not None:
        evidences.append(bundle.contract)
    evidences.append(build_static_evidence(execution_result, config))
    if bundle.meta is not None:
        evidences.append(bundle.meta)
    if bundle.probe is not None:
        evidences.append(bundle.probe)

    proxy_score, confidence = weighted_average(evidences, PROXY_WEIGHTS)
    return proxy_score, confidence, evidences
