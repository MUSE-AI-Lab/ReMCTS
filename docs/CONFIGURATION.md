# Configuration

A run is described by a YAML file that `remcts.config.load_config` turns into a `SearchConfig`.
Every command accepts either a packaged config name or a path to your own YAML file.

## Packaged Configs

| Name | Use |
| --- | --- |
| `default` | Offline smoke test with the mock client |
| `remcts_known` | ReMCTS, `known` setting (visible tests guide the search) |
| `remcts_no_tests` | ReMCTS, `no_tests` setting (proxy evidence and acceptance gate) |
| `pure_mcts_no_tests` | Pure-MCTS baseline: no memory, semantic UCB, LLM review or LLM probes |
| `mcts_style_no_tests`, `mcts_style_known` | MCTS-style baseline: pure-MCTS plus LLM semantic review |
| `remcts_cpp_known` | HumanEval-X C++ pilot (3 iterations, branch factor 3, C++17 executor) |

The paper runs use 8 iterations, branch factor 5, maximum depth 8, minimum acceptance iteration 3,
`C = 1.4`, `k = 3` path reflections, `t = 2` retrieved global memories, `alpha = 0.4`, `beta = 0.6`,
temperature 0.2, at most 1024 output tokens, and a 5-second execution timeout.

## Sections

| Section | Keys |
| --- | --- |
| `search` | `max_iterations`, `branch_factor`, `max_depth`, `exploration_c`, `path_memory_k`, `global_memory_top_t`, `no_improve_patience` |
| `semantic` | `alpha` (repair-confidence bonus), `beta` (high-risk similarity penalty) of the UCB memory term |
| `evaluation` | `mode: benchmark` (`known`) or `mode: production` (`no_tests`), `oracle_policy` |
| `reward` | `known` weights `w_unit`, `w_exec`, `w_semantic`; `semantic_mode` (`heuristic`, `llm`, `disabled`) |
| `production` | `no_tests` weights `w_proxy`, `w_exec`, `w_semantic`, `w_risk`; thresholds `*_threshold`; penalties `gamma_*`; fixed evidence confidences `*_confidence`; `complete_branch_before_accept`, `min_accept_iterations`; LLM probes `probe_llm_tests`, `probe_llm_count` |
| `reflection` | `mode` (`heuristic` or `llm`), `max_feedback_chars`, `fallback_on_error` |
| `sandbox` | `timeout_sec`, `max_output_chars` |
| `llm` | `model_name`, `base_url`, `api_key_env`, `credentials_file`, `api_alias`, `timeout_sec`, `max_retries`, `enable_thinking`, `temperature`, `max_tokens` |
| `storage` | `run_output_dir` |

## Models and Credentials

The client speaks the OpenAI-compatible `/chat/completions` protocol. For another provider, change
`llm.model_name` and `llm.base_url`, then validate one task before a full run (providers differ in
reasoning and token-limit parameters; Qwen3 and DeepSeek-V4 thinking is switched off as in the paper).

For single runs (`remcts-run-search`, `remcts-evaluate`, the Python API), the key is resolved in this order:

1. a profile in `llm.credentials_file` (selected by `llm.api_alias` or `--api-alias`);
2. the environment variable named by `llm.api_key_env` (default `REMCTS_API_KEY`);
3. `OPENAI_API_KEY`.

A credentials file holds one profile as `key: value` lines, several profiles separated by blank lines
or `[alias]` headers, or the same in JSON:

```text
alias: qwen
api_key: YOUR_QWEN_KEY
base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
model_name: qwen3-8b

alias: deepseek
api_key: YOUR_DEEPSEEK_KEY
base_url: https://api.deepseek.com
model_name: deepseek-v4-flash
```

The experiment drivers (`remcts-experiment`) read `--credentials-file` (default `apikeys.txt`). If it
contains provider-labelled keys (`qwen: sk-...`, `deepseek: sk-...`), tasks are spread over those keys
and retried on the next key after a retryable error; otherwise it is used as a profile file whose
aliases are the model names. Reports only contain 8-character key fingerprints. `apikeys.txt`,
`my_api_key.txt`, `.env` and `*.credentials*` are ignored by Git; never put keys in a tracked file.

## Settings

- **`known`** (`evaluation.mode: benchmark`): the search executes the tests in the task file. A candidate
  is accepted when all of them pass, subject to `min_accept_iterations` and `complete_branch_before_accept`.
- **`no_tests`** (`evaluation.mode: production`, `oracle_policy: no_oracle`): no explicit tests. The
  reward combines contract, static, probe and metamorphic evidence, semantic review and risk; unavailable
  evidence is left out of the weighted average instead of counting as a failure.

In both settings the reported score comes from the separate hidden-evaluation split.
