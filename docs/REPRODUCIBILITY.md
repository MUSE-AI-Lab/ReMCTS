# Reproducing the Experiments

## 1. Environment and Data

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/prepare_datasets.py
```

The C++ pilot needs `g++` with C++17 on `PATH`. Compare `data/dataset_manifest.json` with the
committed version to confirm that the derived splits are identical.

## 2. Credentials

Create `apikeys.txt` (ignored by Git) with provider-labelled keys, one per line:

```text
qwen: YOUR_DASHSCOPE_KEY
deepseek: YOUR_DEEPSEEK_KEY
```

Qwen3 models are called on DashScope and DeepSeek-V4-Flash on the DeepSeek API, both with thinking
disabled. See [CONFIGURATION.md](CONFIGURATION.md) for other providers and profile files.

## 3. Plan, Then Run

Every command prints the planned runs and a rough cost estimate with `--dry-run`, without API calls:

```bash
remcts-experiment main --dry-run
```

Main table (Direct and ReMCTS on HumanEval and MBPP-Sanitized, `known` and `no_tests`; Direct
`no_tests` re-scores the Direct `known` generations, since Direct does not use tests):

```bash
remcts-experiment main \
  --models qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b deepseek-v4-flash \
  --workers 8 --budget-cny 500 --report-dir outputs/reports/main
```

Supplementary baselines and budget-up controls:

```bash
remcts-experiment baselines --report-dir outputs/reports/baselines
remcts-experiment baselines --methods best_of_n reflexion --best-of-n 32 --reflexion-trials 16 \
  --models qwen3-8b qwen3-32b --datasets mbpp_sanitized_full_no_tests --report-dir outputs/reports/budget-up
```

Best-of-N and Reflexion-style generate their own tests and run on the `no_tests` datasets;
Reflection-style uses the visible tests and runs on the `known` datasets; pure-MCTS runs on `no_tests`;
MCTS-style runs in both settings.

Memory ablations (without global memory retrieval, without path reflection memory):

```bash
remcts-experiment ablations --report-dir outputs/reports/ablations
```

HumanEval-X C++ pilot on the first 30 tasks in benchmark order:

```bash
remcts-experiment main --datasets humanevalx_cpp --limit 30 \
  --models qwen3-8b qwen3-32b deepseek-v4-flash --report-dir outputs/reports/cpp
```

## 4. Reports

A report directory contains:

- `<label>.json`: one run (model × method × dataset) with per-task rows: returned code, hidden-test
  results, acceptance decision, iterations, tree size, token usage and key fingerprint;
- `<label>.progress.json`: rows written as tasks finish; a rerun with the same `--report-dir` resumes
  from it and retries failed requests (`--force` starts over);
- `summary.md` / `summary.json`: hidden solve rate, acceptance rate, acceptance precision, false
  acceptance rate, average iterations and tree size, calls, tokens and estimated cost per run;
- `runs/`: the search traces (`nodes.jsonl`, `memory.jsonl`, `summary.json`) of every task.

`remcts-experiment status <report-dir>` lists the progress of each run. The budget guard
(`--budget-cny`, `--initial-spent-cny`, `--stop-margin-cny`) stops scheduling new tasks once the
estimated spend reaches the budget minus the margin; the run can be resumed later.

## 5. Run Record

For every published number keep the Git commit, the report directory (it records the config, model
alias, dataset files and token usage), the dataset manifest and the run date. API-served models change
over time, so exact replication of hosted-model results is not guaranteed.
