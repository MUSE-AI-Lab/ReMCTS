# ReMCTS Experiment Datasets

This directory documents the datasets prepared for the ReMCTS experiments.

Regenerate everything with:

```bash
python3 scripts/prepare_datasets.py
```

Use `--force` to re-download the official source files.

## Main Files

| File | Rows | Use |
| --- | ---: | --- |
| `humaneval/HumanEval.jsonl` | 164 | Original HumanEval oracle-feedback runs. |
| `humaneval/HumanEval_feedback.jsonl` | 164 | Standard-comparable search feedback from prompt doctests only. |
| `humaneval/HumanEval_eval.jsonl` | 164 | Original HumanEval tests for offline final evaluation. |
| `humaneval/HumanEval_no_tests.jsonl` | 164 | Production-mode no-explicit-test search. |
| `mbpp/mbpp.json` | 974 | Full MBPP converted to the current loader format. |
| `mbpp/mbpp_test_11_510.json` | 500 | Official MBPP test split by task id. |
| `mbpp/mbpp_sanitized.json` | 427 | Google Research sanitized MBPP converted to the current loader format. |
| `mbpp/mbpp_sanitized_test_11_510.json` | 257 | Sanitized MBPP tasks whose ids fall in the official test range. |
| `evalplus/HumanEvalPlus.jsonl` | 164 | EvalPlus HumanEval+ raw JSONL for robustness evaluation. |
| `evalplus/MbppPlus.jsonl` | 378 | EvalPlus MBPP+ raw JSONL for robustness evaluation. |
| `evalplus/MbppPlus_base_assertions.json` | 378 | MBPP+ base assertions converted to the current MBPP loader format. |

Each MBPP variant also has:

- `*_feedback_split.json`: setup/import lines plus the first assertion for search feedback.
- `*_eval_split.json`: setup/import lines plus the remaining assertions for offline evaluation.
- `*_no_tests.json`: no explicit assertions for production-mode search.

See `dataset_manifest.json` for source URLs, SHA-256 hashes, byte sizes, row counts, and split notes.

Generated benchmark files are intentionally excluded from the public Git
repository. Run the preparation script locally and review each upstream
dataset's license before redistributing original or derived benchmark data.

## Notes

- MBPP source files usually do not provide an `entry_point`; `scripts/prepare_datasets.py` infers it from the reference solution or assertions so the current prompt builder does not fall back to `solution`.
- HumanEval feedback tests are parsed only from doctest-style examples in the prompt. Some tasks have no such examples, so `HumanEval_feedback.jsonl` intentionally contains empty feedback tests for those tasks.
- EvalPlus files are provided for optional robustness evaluation. Full EvalPlus scoring needs an EvalPlus-compatible evaluator; the current repository loader can directly consume `MbppPlus_base_assertions.json` for base assertions only.
