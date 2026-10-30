from __future__ import annotations

from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.evaluation.evidence import EvidenceResult
from remcts.evaluation.evidence_builder import EvidenceBundle, build_evidence_bundle, weighted_average
from remcts.sandbox.executor import PythonExecutor

# Item weights eta_k of the proxy-correctness reward (paper Eq. 14).
PROXY_WEIGHTS = {
    "contract": 0.35,
    "probe": 0.25,
    "meta": 0.15,
    "static": 0.25,
}


def static_evidence(execution: EvidenceResult, execution_result: ExecutionResult, confidence: float) -> EvidenceResult:
    """Compile/run status as a proxy item: syntax error 0, timeout 0.45, crash 0.65, clean run 1."""
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
        confidence=confidence,
        available=execution.available,
        passed=score >= 0.75,
        details=dict(execution.details),
    )


def score_proxy_bundle(
    bundle: EvidenceBundle,
    execution_result: ExecutionResult,
    config: SearchConfig,
) -> tuple[float, float, list[EvidenceResult]]:
    """Proxy reward, mean confidence of the available items, and the items themselves."""
    evidences: list[EvidenceResult] = []
    if bundle.contract is not None:
        evidences.append(bundle.contract)
    evidences.append(static_evidence(bundle.execution, execution_result, config.production_static_confidence))
    if bundle.meta is not None:
        evidences.append(bundle.meta)
    if bundle.probe is not None:
        evidences.append(bundle.probe)
    score, confidence = weighted_average(evidences, PROXY_WEIGHTS)
    return score, confidence, evidences


def score_proxy(
    problem: ProblemSpec,
    code: str,
    execution_result: ExecutionResult,
    config: SearchConfig,
    executor: PythonExecutor | None = None,
) -> tuple[float, float, list[EvidenceResult]]:
    bundle = build_evidence_bundle(problem, code, execution_result, config, executor=executor)
    return score_proxy_bundle(bundle, execution_result, config)
