# Data Release Policy

ReMCTS uses HumanEval, MBPP, EvalPlus, and a limited HumanEval-X C++ subset.
These benchmarks originate in third-party repositories and retain their
upstream licenses and attribution requirements.

The public code repository should contain:

- `scripts/prepare_datasets.py`
- `data/README.md`
- `data/dataset_manifest.json`

Generated benchmark copies under `data/` are intentionally ignored. Users can
recreate them from the recorded official URLs and transformations.

Before publishing any derived dataset separately:

1. Verify every upstream license permits redistribution and transformation.
2. Preserve source attribution and document the exact split transformation.
3. Remove hidden test material if the benchmark policy prohibits publication.
4. Include checksums, schema, intended use, limitations, and contamination
   considerations in a dataset card.
5. Publish it as a separate versioned dataset rather than adding large files to
   the code repository.
