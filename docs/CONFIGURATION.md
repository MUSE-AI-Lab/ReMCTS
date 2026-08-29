# Configuration

ReMCTS reads YAML configuration into `SearchConfig`. The default configuration
uses the mock LLM client and is safe for local smoke tests.

## Main Sections

- `search`: iteration, branching, depth, exploration, memory, and stopping
  parameters.
- `evaluation`: explicit-test (`benchmark`) or no-oracle (`production`) mode.
- `reward`: benchmark reward weights and semantic-scoring behavior.
- `production`: proxy, execution, semantic, risk, uncertainty, and acceptance
  settings used when explicit tests are unavailable.
- `reflection`: heuristic or LLM-generated structured reflection.
- `sandbox`: timeout and captured-output limits.
- `llm`: model, endpoint, credential source, retry, and decoding settings.
- `storage`: local run-artifact directory.

## Credentials

The recommended public-repository workflow is an environment variable:

```bash
export REMCTS_API_KEY="YOUR_API_KEY"
```

The client resolves credentials in this order:

1. A profile selected from the local file in `llm.credentials_file`.
2. The environment variable named by `llm.api_key_env`.
3. `OPENAI_API_KEY`.

Credential files are supported for local multi-provider work, but are ignored
by Git. A profile file may use YAML-like blocks:

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

Select a profile with `--api-alias qwen`. Never place real values in a tracked
configuration file.

Paper experiment drivers that rotate multiple keys expect a local
`apikeys.txt` containing one provider-labelled key per line, for example:

```text
qwen: YOUR_QWEN_KEY
deepseek: YOUR_DEEPSEEK_KEY
```

`apikeys.txt` and `my_api_key.txt` are both excluded by `.gitignore`.

## Evaluation Modes

### Explicit-test / `known`

Set `evaluation.mode: benchmark` and provide visible tests. Passing visible
tests may stop search, but any reported final score must be computed using the
separate hidden-evaluation split.

### No-explicit-test / `no_tests`

Set `evaluation.mode: production` and `evaluation.oracle_policy: no_oracle`.
Search then uses contracts, static checks, probes, semantic review, and risk
evidence. Hidden tests remain offline-only.

## Provider Portability

The client targets the OpenAI-compatible `/chat/completions` interface. To use a
different provider, change only `llm.model_name`, `llm.base_url`, and the local
credential source. Some providers expose different reasoning or token-limit
parameters; validate a one-task smoke run before starting a full experiment.
