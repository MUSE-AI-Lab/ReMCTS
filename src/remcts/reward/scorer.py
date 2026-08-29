from __future__ import annotations

from dataclasses import replace
from typing import Any

from remcts.core.state import ExecutionResult, ProblemSpec, RewardResult, SearchConfig
from remcts.evaluation.evidence import EvidenceResult
from remcts.evaluation.evidence_builder import EvidenceBundle, build_evidence_bundle, weighted_average
from remcts.sandbox.executor import PythonExecutor

from .proxy_score import PROXY_WEIGHTS
from .risk_score import score_risk
from .semantic_score import score_semantic
from .semantic_scorer import SemanticScorer


def score_candidate(
    execution_result: ExecutionResult,
    config: SearchConfig,
    problem: ProblemSpec | None = None,
    code: str = "",
    semantic_scorer: SemanticScorer | None = None,
    executor: PythonExecutor | None = None,
) -> RewardResult:
    evidence_bundle = build_evidence_bundle(
        problem,
        code,
        execution_result,
        config,
        executor=executor,
        probe_llm_client=semantic_scorer.llm_client if semantic_scorer is not None else None,
    )
    r_unit = evidence_bundle.unit.score
    r_exec = evidence_bundle.execution.score
    if config.evaluation_mode.lower() == "production" and problem is not None:
        return _score_production_candidate(
            problem=problem,
            code=code,
            execution_result=execution_result,
            config=config,
            evidence_bundle=evidence_bundle,
            semantic_scorer=semantic_scorer,
        )

    semantic_execution = _with_evidence_summary(
        execution_result,
        evidence_bundle.evidences,
        risk=None,
        summary_key="benchmark_evidence",
    )
    if semantic_scorer is not None and problem is not None:
        r_semantic, semantic_reason, semantic_details = semantic_scorer.score(
            problem,
            code,
            semantic_execution,
        )
    else:
        r_semantic, semantic_reason = score_semantic(semantic_execution)
        semantic_details = {"semantic_mode": "heuristic"}
    total_reward = (
        config.w_unit * r_unit
        + config.w_exec * r_exec
        + config.w_semantic * r_semantic
    )
    return RewardResult(
        r_unit=r_unit,
        r_exec=r_exec,
        r_semantic=r_semantic,
        total_reward=total_reward,
        details={
            "evaluation_mode": config.evaluation_mode,
            "r_unit": r_unit,
            "r_exec": r_exec,
            "r_semantic": r_semantic,
            "benchmark_evidence": evidence_bundle.to_dicts(),
            "semantic_reason": semantic_reason,
            **semantic_details,
        },
    )


def _score_production_candidate(
    problem: ProblemSpec,
    code: str,
    execution_result: ExecutionResult,
    config: SearchConfig,
    evidence_bundle: EvidenceBundle,
    semantic_scorer: SemanticScorer | None,
) -> RewardResult:
    r_proxy, proxy_confidence, proxy_evidences = _score_proxy_from_evidence_bundle(
        evidence_bundle,
        execution_result,
        config,
    )
    risk = score_risk(code, config)
    semantic_execution = _with_evidence_summary(
        execution_result,
        proxy_evidences,
        risk=risk,
        summary_key="production_evidence",
    )
    if semantic_scorer is not None:
        r_semantic, semantic_reason, semantic_details = semantic_scorer.score(
            problem,
            code,
            semantic_execution,
        )
    else:
        r_semantic, semantic_reason = score_semantic(semantic_execution)
        semantic_details = {"semantic_mode": "heuristic"}
    r_exec = evidence_bundle.execution.score
    exec_confidence = evidence_bundle.execution.confidence
    semantic_confidence = _semantic_confidence(semantic_details)
    positive_weight = (
        config.production_w_proxy * proxy_confidence
        + config.production_w_exec * exec_confidence
        + config.production_w_semantic * semantic_confidence
    )
    risk_weight = config.production_w_risk * risk.confidence
    numerator = (
        config.production_w_proxy * proxy_confidence * r_proxy
        + config.production_w_exec * exec_confidence * r_exec
        + config.production_w_semantic * semantic_confidence * r_semantic
        - risk_weight * risk.score
    )
    denominator = positive_weight + risk_weight
    total_reward = _clamp(numerator / denominator if denominator > 0 else 0.0)
    evidence_confidences = [item.confidence for item in proxy_evidences if item.available]
    evidence_confidences.append(risk.confidence)
    evidence_confidences.append(semantic_confidence)
    uncertainty = 1.0 - (sum(evidence_confidences) / len(evidence_confidences))
    deviation_score = _deviation_score(code)
    accept_score = _clamp(
        total_reward
        - config.production_gamma_uncertainty * uncertainty
        - config.production_gamma_risk * risk.score
        - config.production_gamma_deviation * deviation_score
    )
    accepted = (
        accept_score >= config.production_accept_threshold
        and r_proxy >= config.production_proxy_threshold
        and r_exec >= config.production_exec_threshold
        and r_semantic >= config.production_semantic_threshold
        and risk.score <= config.production_risk_threshold
    )
    needs_review = risk.score > config.production_risk_threshold or (
        not accepted and uncertainty > 0.45
    )
    return RewardResult(
        r_unit=r_proxy,
        r_exec=r_exec,
        r_semantic=r_semantic,
        total_reward=total_reward,
        details={
            "evaluation_mode": "production",
            "semantic_reason": semantic_reason,
            **semantic_details,
            "r_proxy": r_proxy,
            "r_risk": risk.score,
            "proxy_confidence": proxy_confidence,
            "exec_confidence": exec_confidence,
            "semantic_confidence": semantic_confidence,
            "risk_confidence": risk.confidence,
            "uncertainty": uncertainty,
            "deviation_score": deviation_score,
            "accept_score": accept_score,
            "accepted": accepted,
            "needs_review": needs_review,
            "evidence": [item.to_dict() for item in proxy_evidences],
            "all_evidence": evidence_bundle.to_dicts(),
            "risk": risk.to_dict(),
        },
    )


def _semantic_confidence(details: dict) -> float:
    subscores = details.get("semantic_subscores")
    if isinstance(subscores, dict) and "confidence" in subscores:
        try:
            return _clamp(float(subscores["confidence"]))
        except (TypeError, ValueError):
            return 0.50
    mode = details.get("semantic_mode")
    if mode in {"production_llm", "llm"}:
        return 0.65
    if mode and str(mode).startswith("shortcut"):
        return 0.85
    return 0.45


def _score_proxy_from_evidence_bundle(
    evidence_bundle: EvidenceBundle,
    execution_result: ExecutionResult,
    config: SearchConfig,
) -> tuple[float, float, list[EvidenceResult]]:
    proxy_evidences: list[EvidenceResult] = []
    if evidence_bundle.contract is not None:
        proxy_evidences.append(evidence_bundle.contract)
    proxy_evidences.append(
        _static_evidence_from_execution_evidence(
            evidence_bundle.execution,
            execution_result,
            confidence=config.production_static_confidence,
        )
    )
    if evidence_bundle.meta is not None:
        proxy_evidences.append(evidence_bundle.meta)
    if evidence_bundle.probe is not None:
        proxy_evidences.append(evidence_bundle.probe)
    proxy_score, confidence = weighted_average(proxy_evidences, PROXY_WEIGHTS)
    return proxy_score, confidence, proxy_evidences


def _static_evidence_from_execution_evidence(
    execution: EvidenceResult,
    execution_result: ExecutionResult,
    confidence: float,
) -> EvidenceResult:
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


def _with_evidence_summary(
    execution_result: ExecutionResult,
    evidences: list[Any],
    risk: Any | None,
    summary_key: str,
) -> ExecutionResult:
    coverage_hint = dict(execution_result.coverage_hint)
    coverage_hint[summary_key] = [_summarize_evidence(item) for item in evidences]
    if risk is not None:
        coverage_hint["production_risk"] = {
            "score": risk.score,
            "confidence": risk.confidence,
            "passed": risk.passed,
            "findings": risk.details.get("findings", []),
        }
    return replace(execution_result, coverage_hint=coverage_hint)


def _summarize_evidence(evidence: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": evidence.name,
        "score": evidence.score,
        "confidence": evidence.confidence,
        "available": evidence.available,
        "passed": evidence.passed,
    }
    details = evidence.details if isinstance(evidence.details, dict) else {}
    if evidence.name == "contract":
        summary["details"] = {
            "entry_point": details.get("entry_point"),
            "required_args": details.get("required_args"),
            "has_vararg": details.get("has_vararg"),
            "has_kwarg": details.get("has_kwarg"),
        }
    elif evidence.name == "meta":
        execution = details.get("execution", {})
        summary["details"] = {
            "properties": details.get("properties", []),
            "passed_tests": execution.get("passed_tests"),
            "total_tests": execution.get("total_tests"),
            "failed_cases": execution.get("failed_cases", []),
        }
    elif evidence.name == "probe":
        execution = details.get("execution", {})
        summary["details"] = {
            "probe_tests": details.get("probe_tests"),
            "passed_tests": execution.get("passed_tests"),
            "total_tests": execution.get("total_tests"),
            "failed_cases": execution.get("failed_cases", []),
        }
    elif evidence.name == "static":
        summary["details"] = {
            "compile_ok": details.get("compile_ok"),
            "exec_ok": details.get("exec_ok"),
            "timed_out": details.get("timed_out"),
        }
    return summary


def _deviation_score(code: str) -> float:
    line_count = len([line for line in code.splitlines() if line.strip()])
    if line_count <= 80:
        return 0.0
    if line_count <= 200:
        return 0.15
    return 0.35


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
