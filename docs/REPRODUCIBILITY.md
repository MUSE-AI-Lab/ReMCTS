# Reproducing ReMCTS Experiments

## 1. Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python scripts/prepare_datasets.py
```

For the C++ pilot, install a compiler with C++17 support and verify `g++` is on
`PATH`.

## 2. Credentials

Create a local `apikeys.txt` as described in [Configuration](CONFIGURATION.md).
This file is ignored by Git. The experiment drivers print only short key
fingerprints and must never persist full credentials.

## 3. Main Multi-model Runs

Inspect the planned jobs and cost estimate before making API calls:

```bash
python scripts/run_qwen_multi_model_experiment.py \
  --models qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b deepseek-v4-flash \
  --dry-run
```

Then run with an explicit worker count, budget, and output directory:

```bash
python scripts/run_qwen_multi_model_experiment.py \
  --models qwen3-1.7b qwen3-8b qwen3-14b qwen3-32b deepseek-v4-flash \
  --workers 5 \
  --budget-cny 1000 \
  --report-dir outputs/reports/main
```

The driver evaluates final candidates on the hidden split offline. Hidden tests
must never be used to generate, select, or repair candidates.

## 4. Baselines and Ablations

```bash
python scripts/run_pure_mcts_baselines.py --force
python scripts/run_mcts_style_baselines.py --force

python scripts/run_memory_ablation_experiments.py \
  --models qwen3-8b qwen3-32b deepseek-v4-flash \
  --settings known no_tests \
  --credentials-file apikeys.txt \
  --report-dir outputs/reports/memory-ablation

python scripts/run_known_baseline_completions.py \
  --models qwen3-8b qwen3-32b deepseek-v4-flash \
  --credentials-file apikeys.txt \
  --report-dir outputs/reports/known-baselines
```

Use `--limit` for smoke runs where the script supports it. Do not use `--force`
until the destination directory has been checked; it may overwrite completed
task artifacts.

## 5. HumanEval-X C++ Pilot

```bash
python scripts/run_humanevalx_cpp_pilot.py \
  --models qwen3-8b \
  --methods direct remcts \
  --limit 30 \
  --key-pool-file apikeys.txt \
  --report-dir outputs/reports/humanevalx-cpp-pilot30
```

This is a limited compiler-backed pilot, not evidence of broad multilingual or
repository-level performance.

## 6. Run Record

For every published result, retain:

- Git commit and release tag
- effective YAML configuration
- source dataset checksums
- model name, provider, and invocation date
- task IDs and task count
- decoding parameters and search budget
- aggregate calls, tokens, elapsed time, and estimated cost
- final hidden-evaluation policy

Keep raw traces outside Git. Release only reviewed aggregate tables and, when
appropriate, a separately archived artifact with its own license and checksum.
