from __future__ import annotations

from dataclasses import dataclass

from remcts.core.state import ExecutionResult, ProblemSpec, SearchConfig
from remcts.llm.client import BaseLLMClient
from remcts.reward.exec_score import score_execution
from remcts.reward.unit_score import score_unit
from remcts.sandbox.executor import PythonExecutor

from .contract_checker import check_contract
from .evidence import EvidenceResult
from .metamorphic import run_metamorphic_tests
from .probe_generator import run_probe_tests


@dataclass(slots=True)
class EvidenceBundle:
    evidences: list[EvidenceResult]
    unit: EvidenceResult
    execution: EvidenceResult
    contract: EvidenceResult | None = None
    probe: EvidenceResult | None = None
    meta: EvidenceResult | None = None

    def get(self, name: str) -> EvidenceResult | None:
        return next((item for item in self.evidences if item.name == name), None)

    def to_dicts(self) -> list[dict]:
        return [item.to_dict() for item in self.evidences]


def build_evidence_bundle(
    problem: ProblemSpec | None,
    code: str,
    execution_result: ExecutionResult,
    config: SearchConfig,
    executor: PythonExecutor | None = None,
    probe_llm_client: BaseLLMClient | None = None,
) -> EvidenceBundle:
    unit = build_unit_evidence(execution_result)
    execution = build_execution_evidence(execution_result)
    evidences = [unit, execution]
    contract = None
    probe = None
    meta = None

    if problem is not None:
        contract = check_contract(problem, code, config)
        evidences.append(contract)
        if executor is not None:
            meta = run_metamorphic_tests(problem, code, executor, config)
            probe = run_probe_tests(problem, code, executor, config, llm_client=probe_llm_client)
        else:
            meta = EvidenceResult(
                name="meta",
                score=0.0,
                confidence=0.0,
                available=False,
                passed=None,
                details={"reason": "missing executor"},
            )
            probe = EvidenceResult(
                name="probe",
                score=0.0,
                confidence=0.0,
                available=False,
                passed=None,
                details={"reason": "missing executor"},
            )
        evidences.extend([meta, probe])

    return EvidenceBundle(
        evidences=evidences,
        unit=unit,
        execution=execution,
        contract=contract,
        probe=probe,
        meta=meta,
    )


def build_unit_evidence(execution_result: ExecutionResult) -> EvidenceResult:
    available = execution_result.total_tests > 0
    score = score_unit(execution_result) if available else 0.0
    return EvidenceResult(
        name="unit",
        score=score,
        confidence=1.0 if available else 0.0,
        available=available,
        passed=execution_result.exec_ok and available and score >= 1.0,
        details={
            "passed_tests": execution_result.passed_tests,
            "total_tests": execution_result.total_tests,
            "failed_cases": execution_result.failed_cases,
        },
    )


def build_execution_evidence(execution_result: ExecutionResult) -> EvidenceResult:
    score = score_execution(execution_result)
    return EvidenceResult(
        name="exec",
        score=score,
        confidence=0.85 if execution_result.compile_ok else 0.50,
        available=True,
        passed=score >= 0.75,
        details={
            "compile_ok": execution_result.compile_ok,
            "exec_ok": execution_result.exec_ok,
            "timed_out": execution_result.timed_out,
            "return_code": execution_result.coverage_hint.get("return_code"),
        },
    )


def weighted_average(evidences: list[EvidenceResult], weights: dict[str, float]) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    confidences: list[float] = []
    for evidence in evidences:
        weight = weights.get(evidence.name, 0.0)
        numerator += evidence.weighted_value(weight)
        denominator += evidence.weighted_mass(weight)
        if evidence.available:
            confidences.append(evidence.confidence)
    score = numerator / denominator if denominator > 0 else 0.0
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return score, confidence
