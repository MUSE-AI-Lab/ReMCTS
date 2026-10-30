from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProblemSpec":
        return cls(
            task_id=str(data.get("task_id", "unknown")),
            prompt=str(data.get("prompt") or ""),
            entry_point=data.get("entry_point"),
            tests=str(data.get("tests") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


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

    def all_passed(self) -> bool:
        return self.total_tests > 0 and self.exec_ok and self.passed_tests == self.total_tests

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
        """Build a config from the nested YAML layout (``search``, ``reward``, ``llm``, ...)."""
        mapping = mapping or {}
        values: dict[str, Any] = {}
        for name, (section, key) in _YAML_KEYS.items():
            section_values = mapping.get(section) or {}
            if key in section_values:
                values[name] = section_values[key]
        llm = mapping.get("llm") or {}
        alias = llm.get("api_alias") or llm.get("alias")
        if alias:
            values["llm_api_alias"] = alias
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _yaml_key(name: str) -> tuple[str, str]:
    for prefix, section in _PREFIX_SECTIONS:
        if name.startswith(prefix):
            return section, name[len(prefix):]
    return "search", name


# Field-name prefix -> YAML section; fields without a prefix live in ``search``.
_PREFIX_SECTIONS = [
    ("llm_", "llm"),
    ("reflection_", "reflection"),
    ("reward_", "reward"),
    ("production_", "production"),
]
_SPECIAL_KEYS = {
    "w_unit": ("reward", "w_unit"),
    "w_exec": ("reward", "w_exec"),
    "w_semantic": ("reward", "w_semantic"),
    "alpha": ("semantic", "alpha"),
    "beta": ("semantic", "beta"),
    "timeout_sec": ("sandbox", "timeout_sec"),
    "max_output_chars": ("sandbox", "max_output_chars"),
    "evaluation_mode": ("evaluation", "mode"),
    "oracle_policy": ("evaluation", "oracle_policy"),
    "reflection_mode": ("reflection", "mode"),
    "run_output_dir": ("storage", "run_output_dir"),
}
_YAML_KEYS: dict[str, tuple[str, str]] = {
    item.name: _SPECIAL_KEYS.get(item.name) or _yaml_key(item.name)
    for item in fields(SearchConfig)
    if item.name != "llm_api_alias"
}


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
