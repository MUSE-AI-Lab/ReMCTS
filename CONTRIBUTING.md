# Contributing

Thank you for helping improve ReMCTS.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

## Pull Requests

1. Keep changes focused and explain the behavioral motivation.
2. Add or update tests for changed search, reward, memory, execution, or parsing
   behavior.
3. Run the complete test suite before opening a pull request.
4. Do not commit API keys, generated run directories, benchmark copies, model
   outputs, or unpublished review material.
5. Document changes that affect experiment comparability, including defaults,
   model identifiers, prompts, acceptance thresholds, and hidden-evaluation
   policy.

## Reporting Results

Include the effective configuration, dataset checksum, model and provider
identifier, run date, random or decoding settings, task count, and aggregate
token/call usage. Hidden tests must not be used during search or candidate
selection.

## Security Issues

Do not open a public issue for a vulnerability that could expose credentials or
enable unsafe code execution. Follow [SECURITY.md](SECURITY.md) instead.
