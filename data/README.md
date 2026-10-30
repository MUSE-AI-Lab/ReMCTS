# ReMCTS Experiment Datasets

Regenerate everything with:

```bash
python scripts/prepare_datasets.py            # --force re-downloads, --skip-download only re-derives
```

Generated files are not tracked by Git; `dataset_manifest.json` records the source URLs, byte sizes,
row counts and SHA-256 hashes, so a regenerated copy can be checked against it. Review the upstream
licenses before redistributing original or derived benchmark data.

## Files

| File | Rows | Use |
| --- | ---: | --- |
| `humaneval/HumanEval.jsonl` | 164 | Original HumanEval with its test field. |
| `humaneval/HumanEval_feedback.jsonl` | 164 | Feedback tests parsed from the doctest examples in the prompt. |
| `humaneval/HumanEval_eval.jsonl` | 164 | Original HumanEval tests for offline hidden evaluation. |
| `humaneval/HumanEval_no_tests.jsonl` | 164 | No explicit tests (`no_tests` search). |
| `mbpp/mbpp.json` | 974 | Full MBPP in the loader format. |
| `mbpp/mbpp_test_11_510.json` | 500 | Official MBPP test split (task ids 11–510). |
| `mbpp/mbpp_sanitized.json` | 427 | Sanitized MBPP in the loader format. |
| `mbpp/mbpp_sanitized_test_11_510.json` | 257 | Sanitized MBPP tasks in the official test id range. |
| `evalplus/HumanEvalPlus.jsonl` | 164 | EvalPlus HumanEval+ (needs an EvalPlus-compatible evaluator). |
| `evalplus/MbppPlus.jsonl` | 378 | EvalPlus MBPP+. |
| `evalplus/MbppPlus_base_assertions.json` | 378 | MBPP+ base assertions in the MBPP loader format. |
| `humaneval_x/humaneval_cpp_feedback.jsonl` | 164 | HumanEval-X C++ with the public `example_test`. |
| `humaneval_x/humaneval_cpp_eval.jsonl` | 164 | HumanEval-X C++ with the official `test` for hidden evaluation. |

Each MBPP variant also has:

- `*_feedback_split.json`: setup/import lines plus the first assertion;
- `*_eval_split.json`: setup/import lines plus the remaining assertions, for offline evaluation;
- `*_no_tests.json`: no explicit assertions.

The dataset registry of the experiment drivers is in `src/remcts/experiments/datasets.py`.

## Notes

- MBPP files rarely provide an `entry_point`; it is inferred from the reference solution or the first
  assertion, so prompts do not fall back to a generic `solution` name.
- HumanEval feedback tests come only from single-line doctest examples in the prompt. Some prompts have
  none, and the last example of a docstring is skipped because its expected output runs into the closing
  quotes, so some tasks have empty feedback tests.
- HumanEval-X C++ entry points are parsed from the declaration; the pilot uses the first 30 tasks in
  benchmark order (`--limit 30`).
