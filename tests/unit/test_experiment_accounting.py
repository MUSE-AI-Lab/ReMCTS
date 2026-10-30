from remcts.experiments.keys import (
    DEEPSEEK_BASE_URL,
    QWEN_BASE_URL,
    is_retryable_error,
    load_key_pool,
    select_profile,
    stable_key_slot,
)
from remcts.experiments.reporting import hidden_summary
from remcts.experiments.usage import aggregate_usage, compact_usage, estimate_cost_cny, merge_usages
from remcts.llm.errors import LLMClientError


def test_key_pool_groups_keys_by_provider(tmp_path):
    path = tmp_path / "apikeys.txt"
    path.write_text(
        "# provider-labelled keys\nqwen: sk-qwenAAA\ndeepseek: sk-deepBBB\nqwen sk-qwenCCC\nother: sk-otherDDD\n",
        encoding="utf-8",
    )
    qwen = load_key_pool(path, "qwen3-8b")
    assert [profile.api_key for profile in qwen] == ["sk-qwenAAA", "sk-qwenCCC"]
    assert {profile.base_url for profile in qwen} == {QWEN_BASE_URL}
    assert {profile.model_name for profile in qwen} == {"qwen3-8b"}
    deepseek = load_key_pool(path, "deepseek-v4-flash")
    assert [(profile.api_key, profile.base_url) for profile in deepseek] == [("sk-deepBBB", DEEPSEEK_BASE_URL)]
    assert load_key_pool(path, "gpt-4o") == []
    assert load_key_pool(tmp_path / "missing.txt", "qwen3-8b") == []
    assert "sk-" not in str(qwen[0].public_dict())


def test_rotation_is_stable_and_moves_to_the_next_key(tmp_path):
    path = tmp_path / "apikeys.txt"
    path.write_text("qwen: sk-a\nqwen: sk-b\nqwen: sk-c\n", encoding="utf-8")
    profiles = load_key_pool(path, "qwen3-32b")
    slot = stable_key_slot("qwen3_32b_remcts_humaneval_full_known", 7)
    assert slot == stable_key_slot("qwen3_32b_remcts_humaneval_full_known", 7)
    first, second = select_profile(profiles, slot, 0), select_profile(profiles, slot, 1)
    assert first is not None and second is not None
    assert (second.index - first.index) % len(profiles) == 1
    assert select_profile([], slot) is None


def test_retryable_errors():
    assert is_retryable_error(LLMClientError("LLM HTTP 429: rate limit exceeded"))
    assert is_retryable_error(TimeoutError("timed out"))
    assert not is_retryable_error(LLMClientError("LLM HTTP 400: invalid request"))


def test_usage_merge_aggregate_and_cost():
    first = {"model": "qwen3-8b", "api_alias": "a", "calls": 2, "successful_calls": 1, "failed_calls": 1,
             "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110, "elapsed_sec": 1.5, "records": [1]}
    second = {"model": None, "calls": 1, "successful_calls": 1, "prompt_tokens": 50, "completion_tokens": 5,
              "total_tokens": 55, "elapsed_sec": 0.5}
    assert "records" not in compact_usage(first)
    merged = merge_usages([first, second])
    assert (merged["model"], merged["calls"], merged["total_tokens"], merged["elapsed_sec"]) == ("qwen3-8b", 3, 165, 2.0)

    totals = aggregate_usage([{"llm_usage": merged}, {"llm_usage": compact_usage(None), "model_alias": "x"}])
    assert totals["totals"]["prompt_tokens"] == 150
    assert set(totals["by_model"]) == {"qwen3-8b", "x"}
    assert estimate_cost_cny("qwen3-8b", 1_000_000, 1_000_000) == 2.5
    assert estimate_cost_cny("unknown-model", 1_000_000, 0) == 2.0  # priced like qwen3-32b


def test_hidden_summary_counts_acceptance_quality():
    rows = [
        {"accepted": True, "hidden_success": True},
        {"accepted": True, "hidden_success": False},
        {"accepted": False, "needs_review": True, "hidden_success": True},
        {"accepted": False, "hidden_success": False},
        {"accepted": True},  # not hidden-evaluated
    ]
    summary = hidden_summary(rows)
    assert summary["tasks"] == 4
    assert summary["hidden_solve_rate"] == 0.5
    assert summary["accepted_rate"] == 0.5
    assert summary["acceptance_precision"] == 0.5
    assert summary["false_acceptance_rate"] == 0.5
    assert summary["needs_review"] == 1
    assert hidden_summary([{"accepted": True}]) is None
