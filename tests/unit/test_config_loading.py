import pytest

from remcts.config import available_configs, load_config, resolve_config_path
from remcts.core.state import SearchConfig


def test_packaged_configs_load():
    names = set(available_configs())
    assert {
        "default",
        "remcts_known",
        "remcts_no_tests",
        "pure_mcts_no_tests",
        "mcts_style_known",
        "mcts_style_no_tests",
        "remcts_cpp_known",
    } <= names
    for name in names:
        assert isinstance(load_config(name), SearchConfig)


def test_main_configs_match_the_paper_settings():
    known = load_config("remcts_known")
    no_tests = load_config("remcts_no_tests")
    for config in (known, no_tests):
        assert (config.max_iterations, config.branch_factor, config.max_depth) == (8, 5, 8)
        assert (config.exploration_c, config.alpha, config.beta) == (1.4, 0.4, 0.6)
        assert (config.path_memory_k, config.global_memory_top_t) == (3, 2)
        assert config.production_min_accept_iterations == 3
        assert config.production_complete_branch_before_accept is True
        assert (config.llm_temperature, config.llm_max_tokens, config.timeout_sec) == (0.2, 1024, 5)
    assert known.evaluation_mode == "benchmark"
    assert (known.w_unit, known.w_exec, known.w_semantic) == (0.6, 0.2, 0.2)
    assert no_tests.evaluation_mode == "production"
    assert (
        no_tests.production_w_proxy,
        no_tests.production_w_exec,
        no_tests.production_w_semantic,
        no_tests.production_w_risk,
    ) == (0.4, 0.2, 0.3, 0.1)
    assert (
        no_tests.production_accept_threshold,
        no_tests.production_proxy_threshold,
        no_tests.production_exec_threshold,
        no_tests.production_semantic_threshold,
        no_tests.production_risk_threshold,
    ) == (0.62, 0.5, 0.6, 0.45, 0.25)
    assert (
        no_tests.production_gamma_uncertainty,
        no_tests.production_gamma_risk,
        no_tests.production_gamma_deviation,
    ) == (0.10, 0.35, 0.10)
    assert (no_tests.production_probe_llm_tests, no_tests.production_probe_llm_count) == (True, 4)


def test_baseline_configs_disable_memory():
    for name in ("pure_mcts_no_tests", "mcts_style_no_tests", "mcts_style_known"):
        config = load_config(name)
        assert (config.path_memory_k, config.global_memory_top_t, config.alpha, config.beta) == (0, 0, 0.0, 0.0)
    assert load_config("pure_mcts_no_tests").reward_semantic_mode == "disabled"
    assert load_config("mcts_style_no_tests").reward_semantic_mode == "llm"


def test_overrides_aliases_and_paths(tmp_path):
    config = load_config("remcts_known", {"search": {"global_memory_top_t": 0}})
    assert (config.path_memory_k, config.global_memory_top_t) == (3, 0)

    path = tmp_path / "custom.yaml"
    path.write_text("llm:\n  alias: my-profile\n  model_name: my-model\n", encoding="utf-8")
    config = load_config(path)
    assert config.llm_api_alias == "my-profile"
    assert config.llm_model_name == "my-model"
    assert config.max_iterations == SearchConfig().max_iterations

    with pytest.raises(FileNotFoundError, match="Packaged configs"):
        resolve_config_path("does-not-exist")
