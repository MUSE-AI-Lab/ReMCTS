import json

import pytest

from remcts.experiments import worker
from remcts.experiments.runner import BudgetExceeded, RunOptions, run_experiments
from remcts.experiments.specs import BASELINE_METHODS, ablation_specs, baseline_specs, main_specs
from remcts.llm.client import BaseLLMClient
from remcts.llm.errors import LLMClientError

DATASETS = ["humaneval_full_known", "humaneval_full_no_tests", "mbpp_sanitized_full_known", "mbpp_sanitized_full_no_tests"]
PROMPT = "Write a Python function solution(x) that returns the input unchanged."


class IdentityLLM(BaseLLMClient):
    """Answers every prompt type of ReMCTS and the baselines; code prompts get a correct solution."""

    def generate(self, prompt: str, **kwargs) -> str:
        if '"tests"' in prompt:
            return '{"tests": ["assert solution(1) == 1", "assert solution(2) == 2"]}'
        if "requirement_coverage" in prompt:
            return (
                '{"requirement_coverage":0.95,"api_contract_match":0.95,"context_compatibility":0.9,'
                '"maintainability":0.9,"risk":0.0,"confidence":0.9,"reason":"identity"}'
            )
        if '"score"' in prompt:
            return '{"score": 0.9, "reason": "identity"}'
        if "JSON" in prompt:
            return '{"error_type": "AssertionError", "root_cause": "wrong value", "fix_hint": "return x", "severity": 0.5}'
        return "```python\ndef solution(x):\n    return x\n```"


class RejectingLLM(BaseLLMClient):
    def generate(self, prompt: str, **kwargs) -> str:
        raise LLMClientError("LLM HTTP 400: invalid request")


def write_datasets(root):
    humaneval_tests = "def check(candidate):\n    assert candidate(1) == 1\n    assert candidate(5) == 5\n"
    rows = [{"task_id": f"HumanEval/{i}", "prompt": PROMPT, "entry_point": "solution"} for i in range(2)]
    files = {
        "humaneval/HumanEval.jsonl": [{**row, "test": humaneval_tests} for row in rows],
        "humaneval/HumanEval_eval.jsonl": [{**row, "test": humaneval_tests} for row in rows],
        "humaneval/HumanEval_no_tests.jsonl": [{**row, "test": ""} for row in rows],
    }
    for name, items in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
    mbpp = [{"task_id": i, "text": PROMPT, "prompt": PROMPT, "entry_point": "solution"} for i in (11, 12)]
    splits = {
        "mbpp/mbpp_sanitized.json": ["assert solution(3) == 3", "assert solution(4) == 4"],
        "mbpp/mbpp_sanitized_eval_split.json": ["assert solution(4) == 4"],
        "mbpp/mbpp_sanitized_no_tests.json": [],
    }
    for name, tests in splits.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([{**item, "test_list": tests} for item in mbpp]), encoding="utf-8")
    return root


def options(tmp_path, **kwargs):
    defaults = dict(
        report_dir=tmp_path / "reports",
        data_dir=write_datasets(tmp_path / "data"),
        workers=1,
        budget_cny=0,
        credentials_file=None,
        key_rotation=False,
    )
    return RunOptions(**{**defaults, **kwargs})


def test_main_suite_scores_on_hidden_tests_and_reuses_direct_generations(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "create_llm_client", lambda config: IdentityLLM())
    run = options(tmp_path)
    specs = main_specs(["fake-model"], DATASETS)
    outputs = {output["label"]: output for output in run_experiments(specs, run)}

    assert len(outputs) == 8
    assert {output["status"] for output in outputs.values()} == {"complete"}
    assert {output["pass_rate"] for output in outputs.values()} == {1.0}
    assert {output["pass_rate_metric"] for output in outputs.values()} == {"hidden_solve_rate"}
    reused = outputs["fake_model_direct_humaneval_full_no_tests"]
    assert reused["tasks"][0]["stop_reason"] == "direct_reused_for_hidden_eval"
    assert reused["tasks"][0]["reused_llm_from_label"] == "fake_model_direct_humaneval_full_known"
    remcts = outputs["fake_model_remcts_mbpp_sanitized_full_no_tests"]
    assert remcts["tasks"][0]["hidden_total_tests"] == 1
    assert remcts["tasks"][0]["code"].startswith("def solution")

    summary = json.loads((run.report_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["rows"]) == 8
    assert "Direct vs. ReMCTS" in (run.report_dir / "summary.md").read_text(encoding="utf-8")

    # A rerun with the same report directory skips complete experiments.
    before = (run.report_dir / "fake_model_remcts_humaneval_full_known.json").stat().st_mtime_ns
    run_experiments(specs, run)
    assert (run.report_dir / "fake_model_remcts_humaneval_full_known.json").stat().st_mtime_ns == before


def test_baselines_and_ablations_run_in_their_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "create_llm_client", lambda config: IdentityLLM())
    run = options(tmp_path)
    specs = baseline_specs(["fake-model"], list(BASELINE_METHODS), DATASETS)
    specs += ablation_specs(["fake-model"], ["no_global_memory", "no_path_memory"], DATASETS)
    methods = {(spec.method, spec.dataset.setting) for spec in specs}
    assert ("best_of_8", "known") not in methods and ("reflection_style", "no_tests") not in methods
    assert ("mcts_style", "known") in methods and ("pure_mcts", "no_tests") in methods

    outputs = {output["label"]: output for output in run_experiments(specs, run)}
    assert len(outputs) == len(specs)
    assert {output["pass_rate"] for output in outputs.values()} == {1.0}
    best_of = outputs["fake_model_best_of_8_humaneval_full_no_tests"]
    assert best_of["oracle_pool_hidden_success_rate"] == 1.0
    assert best_of["tasks"][0]["generated_tests"]["count"] == 2
    reflection = outputs["fake_model_reflection_style_mbpp_sanitized_full_known"]["tasks"][0]
    assert (reflection["selected_trial"], reflection["stop_reason"]) == (1, "reflection_visible_tests_passed")
    ablation = outputs["fake_model_remcts_no_global_memory_mbpp_sanitized_full_known"]
    assert ablation["config_overrides"] == {"search": {"global_memory_top_t": 0}}


def test_failed_requests_are_retried_on_resume(tmp_path, monkeypatch):
    run = options(tmp_path)
    specs = main_specs(["fake-model"], ["humaneval_full_no_tests"], include_direct=False)
    monkeypatch.setattr(worker, "create_llm_client", lambda config: RejectingLLM())
    assert run_experiments(specs, run) == []
    partial = json.loads((run.report_dir / "fake_model_remcts_humaneval_full_no_tests.json").read_text(encoding="utf-8"))
    assert partial["status"] == "partial"
    assert partial["request_failed_tasks"] == 2

    monkeypatch.setattr(worker, "create_llm_client", lambda config: IdentityLLM())
    (output,) = run_experiments(specs, run)
    assert (output["status"], output["pass_rate"], output["resumed_task_count"]) == ("complete", 1.0, 0)


def test_budget_guard_stops_before_scheduling(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "create_llm_client", lambda config: IdentityLLM())
    run = options(tmp_path, budget_cny=10.0, initial_spent_cny=9.0, stop_margin_cny=5.0)
    with pytest.raises(BudgetExceeded):
        run_experiments(main_specs(["fake-model"], ["humaneval_full_known"]), run)
    progress = json.loads((run.report_dir / "fake_model_direct_humaneval_full_known.progress.json").read_text("utf-8"))
    assert progress["completed"] == 0


def test_worker_processes_with_the_mock_client(tmp_path):
    run = options(tmp_path, workers=2)
    outputs = run_experiments(main_specs(["mock"], ["humaneval_full_known", "mbpp_sanitized_full_no_tests"]), run)
    assert {output["status"] for output in outputs} == {"complete"}
    assert {output["pass_rate"] for output in outputs} == {0.0}  # the mock emits placeholder code
