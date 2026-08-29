# ReMCTS

Official implementation of **ReMCTS: Reflection-Enhanced Monte Carlo Tree
Search for Code Generation**.

ReMCTS is an execution-grounded, memory-augmented, LLM-guided search framework
for code generation. It represents candidate programs as tree states, evaluates
them with execution and proxy evidence, and reuses structured failure
reflections within and across branches.

## Features

- MCTS-style multi-branch program search with UCB selection
- Path-local and global reflection memory
- Explicit-test (`known`) and no-explicit-test (`no_tests`) settings
- Python and C++ execution backends
- OpenAI-compatible LLM API client
- HumanEval, MBPP-Sanitized, and HumanEval-X preparation utilities
- Search traces, reward decompositions, and usage accounting

## Repository Layout

```text
src/remcts/           Core package
scripts/              Dataset preparation and experiment drivers
tests/                Unit and integration tests
data/                 Dataset documentation and generated local data
docs/                 Public configuration and reproduction guides
outputs/              Local run artifacts (ignored by Git)
```

Paper drafts, review material, credentials, and full local experiment traces
are intentionally excluded from the public repository.

## Installation

ReMCTS requires Python 3.10 or newer. A C++17 compiler is additionally required
for the HumanEval-X C++ pilot.

```bash
git clone https://github.com/YOUR_GITHUB_ACCOUNT/ReMCTS.git
cd ReMCTS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Replace `YOUR_GITHUB_ACCOUNT` after the public repository has been created.

## Quick Smoke Test

The default configuration uses a deterministic mock client. It exercises the
search, execution, reward, memory, and artifact-writing pipeline without making
network requests:

```bash
remcts-run-search \
  --config src/remcts/config/default.yaml \
  --task-id smoke-test \
  --prompt "Write a Python function solution(x) that returns x." \
  --tests "assert solution(1) == 1"
```

The mock client intentionally emits placeholder code, so this command validates
the pipeline rather than model accuracy. Run artifacts are written below
`outputs/runs/`.

## Run with an LLM API

ReMCTS works with OpenAI-compatible Chat Completions endpoints. Keep credentials
in environment variables; never add keys to YAML files or commit them to Git.

```bash
export REMCTS_API_KEY="YOUR_API_KEY"

remcts-run-search \
  --config src/remcts/config/production.yaml \
  --task-id no-tests-example \
  --prompt "Write a Python function solution(x) that returns x." \
  --entry-point solution
```

The public `production.yaml` uses a Qwen-compatible endpoint as an example.
Change `llm.model_name` and `llm.base_url` for another OpenAI-compatible
provider. See [Configuration](docs/CONFIGURATION.md) for credential precedence,
API profiles, and the distinction between `known` and `no_tests`.

> [!WARNING]
> ReMCTS executes model-generated code. The included executor applies timeouts
> and static checks, but it is not a hardened security boundary. Run untrusted
> generations only inside an isolated container or virtual machine with no
> secrets, network access, or valuable writable files.

## Prepare Datasets

The repository does not duplicate third-party benchmark data. Download the
official sources and generate the local splits with:

```bash
python scripts/prepare_datasets.py
```

Source URLs, generated files, split semantics, and checksums are documented in
[data/README.md](data/README.md) and `data/dataset_manifest.json`. Review the
upstream dataset licenses before redistributing derived files.

## Evaluation

After preparing the datasets, the generic dataset runners can be invoked as:

```bash
remcts-evaluate \
  --dataset humaneval \
  --data-path data/humaneval/HumanEval_feedback.jsonl \
  --config src/remcts/config/humaneval.yaml

remcts-evaluate \
  --dataset mbpp \
  --data-path data/mbpp/mbpp_sanitized_feedback_split.json \
  --config src/remcts/config/mbpp.yaml
```

The paper-specific multi-model, baseline, ablation, and C++ commands are listed
in [Reproducibility](docs/REPRODUCIBILITY.md). These experiments call paid APIs;
start with `--dry-run` or `--limit` where supported.

## Outputs

Each search writes a timestamped directory containing the effective
configuration, tree nodes, memory records, and final summary:

```text
outputs/runs/<task-id>-<timestamp>/
|-- config.json
|-- nodes.jsonl
|-- memory.jsonl
`-- summary.json
```

Generated artifacts are ignored by Git. Publish only aggregate results or a
separately reviewed artifact archive; traces may contain benchmark prompts,
generated code, provider errors, and other information that should be audited
before release.

## Reproducibility Notes

- Hidden tests are used only for offline final evaluation, never for search or
  candidate selection.
- The public configs preserve the search and acceptance parameters used by the
  experiment drivers. Provider endpoints and credentials are deployment
  settings and may be overridden locally.
- API model revisions, server load, and provider-side decoding can affect exact
  replication. Record the model identifier, provider, date, config, token use,
  and run manifest for every experiment.

## Citation

GitHub renders citation metadata from [`CITATION.cff`](CITATION.cff). The
proceedings DOI and final repository DOI should be added after they are issued.

```bibtex
@inproceedings{wang2026remcts,
  title     = {ReMCTS: Reflection-Enhanced Monte Carlo Tree Search for Code Generation},
  author    = {Wang, Huifei and Huang, Xinying and Sun, Yiheng and Yuan, Yifan},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```

## License

The ReMCTS source code is released under the [MIT License](LICENSE). Dataset and
model licenses remain with their respective owners. Confirm the intended code
license with all authors and Shenzhen University before the first public
release.

## Acknowledgments

This work is supported by the Intelligent Computing Center of Shenzhen
University.
