from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProblemSpec:
    task_id: str
    prompt: str
    entry_point: str | None = None
    tests: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CodeState:
    code: str
    language: str = "python"
    stdout: str = ""
    stderr: str = ""
    compile_ok: bool = False
    exec_ok: bool = False
    passed_tests: int = 0
    total_tests: int = 0

    def success(self) -> bool:
        return self.total_tests > 0 and self.passed_tests == self.total_tests and self.exec_ok

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExecutionResult:
    compile_ok: bool
    exec_ok: bool
    timed_out: bool
    stdout: str
    stderr: str
    traceback: str
    passed_tests: int
    total_tests: int
    failed_cases: list[str] = field(default_factory=list)
    coverage_hint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RewardResult:
    r_unit: float
    r_exec: float
    r_semantic: float
    total_reward: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReflectionRecord:
    reflection_id: str
    summary: str
    error_type: str
    root_cause: str
    fix_hint: str
    severity: float
    reward_delta: float = 0.0
    embedding: list[float] | None = None
    source_task_id: str | None = None
    source_node_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchConfig:
    max_iterations: int = 64
    branch_factor: int = 3
    max_depth: int = 8
    exploration_c: float = 1.4
    path_memory_k: int = 4
    global_memory_top_t: int = 3
    success_threshold: float = 0.999
    no_improve_patience: int = 16
    w_unit: float = 0.6
    w_exec: float = 0.2
    w_semantic: float = 0.2
    alpha: float = 0.4
    beta: float = 0.6
    timeout_sec: int = 3
    max_output_chars: int = 4000
    llm_model_name: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key_env: str = "REMCTS_API_KEY"
    llm_credentials_file: str | None = None
    llm_api_alias: str | None = None
    llm_timeout_sec: int = 60
    llm_max_retries: int = 2
    llm_enable_thinking: bool | None = None
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    reflection_mode: str = "heuristic"
    reflection_max_feedback_chars: int = 3000
    reflection_fallback_on_error: bool = True
    reward_semantic_mode: str = "heuristic"
    reward_semantic_on_success: bool = False
    reward_semantic_on_failure: bool = True
    reward_semantic_max_feedback_chars: int = 3000
    evaluation_mode: str = "benchmark"
    oracle_policy: str = "explicit_tests"
    production_w_proxy: float = 0.4
    production_w_exec: float = 0.2
    production_w_semantic: float = 0.3
    production_w_risk: float = 0.1
    production_accept_threshold: float = 0.70
    production_proxy_threshold: float = 0.50
    production_exec_threshold: float = 0.60
    production_semantic_threshold: float = 0.55
    production_risk_threshold: float = 0.25
    production_gamma_uncertainty: float = 0.15
    production_gamma_risk: float = 0.35
    production_gamma_deviation: float = 0.10
    production_probe_confidence: float = 0.45
    production_contract_confidence: float = 0.75
    production_static_confidence: float = 0.80
    production_meta_confidence: float = 0.35
    production_complete_branch_before_accept: bool = False
    production_min_accept_iterations: int = 1
    production_probe_llm_tests: bool = False
    production_probe_llm_count: int = 4
    run_output_dir: str = "outputs/runs"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> "SearchConfig":
        mapping = mapping or {}
        search = mapping.get("search", {}) or {}
        reward = mapping.get("reward", {}) or {}
        semantic = mapping.get("semantic", {}) or {}
        sandbox = mapping.get("sandbox", {}) or {}
        llm = mapping.get("llm", {}) or {}
        reflection = mapping.get("reflection", {}) or {}
        evaluation = mapping.get("evaluation", {}) or {}
        production = mapping.get("production", {}) or {}
        storage = mapping.get("storage", {}) or {}
        return cls(
            max_iterations=search.get("max_iterations", 64),
            branch_factor=search.get("branch_factor", 3),
            max_depth=search.get("max_depth", 8),
            exploration_c=search.get("exploration_c", 1.4),
            path_memory_k=search.get("path_memory_k", 4),
            global_memory_top_t=search.get("global_memory_top_t", 3),
            success_threshold=search.get("success_threshold", 0.999),
            no_improve_patience=search.get("no_improve_patience", 16),
            w_unit=reward.get("w_unit", 0.6),
            w_exec=reward.get("w_exec", 0.2),
            w_semantic=reward.get("w_semantic", 0.2),
            alpha=semantic.get("alpha", 0.4),
            beta=semantic.get("beta", 0.6),
            timeout_sec=sandbox.get("timeout_sec", 3),
            max_output_chars=sandbox.get("max_output_chars", 4000),
            llm_model_name=llm.get("model_name", "mock"),
            llm_base_url=llm.get("base_url", "https://api.openai.com/v1"),
            llm_api_key_env=llm.get("api_key_env", "REMCTS_API_KEY"),
            llm_credentials_file=llm.get("credentials_file"),
            llm_api_alias=llm.get("api_alias") or llm.get("alias"),
            llm_timeout_sec=llm.get("timeout_sec", 60),
            llm_max_retries=llm.get("max_retries", 2),
            llm_enable_thinking=llm.get("enable_thinking"),
            llm_temperature=llm.get("temperature", 0.2),
            llm_max_tokens=llm.get("max_tokens", 1024),
            reflection_mode=reflection.get("mode", "heuristic"),
            reflection_max_feedback_chars=reflection.get("max_feedback_chars", 3000),
            reflection_fallback_on_error=reflection.get("fallback_on_error", True),
            reward_semantic_mode=reward.get("semantic_mode", "heuristic"),
            reward_semantic_on_success=reward.get("semantic_on_success", False),
            reward_semantic_on_failure=reward.get("semantic_on_failure", True),
            reward_semantic_max_feedback_chars=reward.get("semantic_max_feedback_chars", 3000),
            evaluation_mode=evaluation.get("mode", "benchmark"),
            oracle_policy=evaluation.get("oracle_policy", "explicit_tests"),
            production_w_proxy=production.get("w_proxy", 0.4),
            production_w_exec=production.get("w_exec", 0.2),
            production_w_semantic=production.get("w_semantic", 0.3),
            production_w_risk=production.get("w_risk", 0.1),
            production_accept_threshold=production.get("accept_threshold", 0.70),
            production_proxy_threshold=production.get("proxy_threshold", 0.50),
            production_exec_threshold=production.get("exec_threshold", 0.60),
            production_semantic_threshold=production.get("semantic_threshold", 0.55),
            production_risk_threshold=production.get("risk_threshold", 0.25),
            production_gamma_uncertainty=production.get("gamma_uncertainty", 0.15),
            production_gamma_risk=production.get("gamma_risk", 0.35),
            production_gamma_deviation=production.get("gamma_deviation", 0.10),
            production_probe_confidence=production.get("probe_confidence", 0.45),
            production_contract_confidence=production.get("contract_confidence", 0.75),
            production_static_confidence=production.get("static_confidence", 0.80),
            production_meta_confidence=production.get("meta_confidence", 0.35),
            production_complete_branch_before_accept=production.get("complete_branch_before_accept", False),
            production_min_accept_iterations=production.get("min_accept_iterations", 1),
            production_probe_llm_tests=production.get("probe_llm_tests", False),
            production_probe_llm_count=production.get("probe_llm_count", 4),
            run_output_dir=storage.get("run_output_dir", "outputs/runs"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResult:
    success: bool
    best_node_id: str | None
    best_code: str
    iterations: int
    tree_size: int
    run_dir: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    needs_review: bool = False
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
