"""Benchmark splits produced by ``scripts/prepare_datasets.py``.

Each dataset pairs the file the search reads (``data_file``) with the file used only for the
final offline evaluation of the returned program (``hidden_eval_file``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from remcts.core.state import ProblemSpec
from remcts.tasks.humaneval import HumanEvalDataset
from remcts.tasks.mbpp import MBPPDataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: str  # "humaneval" (JSONL) or "mbpp" (JSON list)
    setting: str  # "known" (visible tests during search) or "no_tests"
    data_file: str
    hidden_eval_file: str
    language: str = "python"
    full_size: int = 0


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        DatasetSpec(
            "humaneval_full_known", "humaneval", "known",
            "humaneval/HumanEval.jsonl", "humaneval/HumanEval_eval.jsonl", full_size=164,
        ),
        DatasetSpec(
            "humaneval_full_no_tests", "humaneval", "no_tests",
            "humaneval/HumanEval_no_tests.jsonl", "humaneval/HumanEval_eval.jsonl", full_size=164,
        ),
        DatasetSpec(
            "mbpp_sanitized_full_known", "mbpp", "known",
            "mbpp/mbpp_sanitized.json", "mbpp/mbpp_sanitized_eval_split.json", full_size=427,
        ),
        DatasetSpec(
            "mbpp_sanitized_full_no_tests", "mbpp", "no_tests",
            "mbpp/mbpp_sanitized_no_tests.json", "mbpp/mbpp_sanitized_eval_split.json", full_size=427,
        ),
        DatasetSpec(
            "humanevalx_cpp", "humaneval", "known",
            "humaneval_x/humaneval_cpp_feedback.jsonl", "humaneval_x/humaneval_cpp_eval.jsonl",
            language="cpp", full_size=164,
        ),
    ]
}
PYTHON_DATASETS = [name for name, spec in DATASETS.items() if spec.language == "python"]


def load_problems(spec: DatasetSpec, data_dir: Path, *, hidden: bool = False) -> list[ProblemSpec]:
    path = Path(data_dir) / (spec.hidden_eval_file if hidden else spec.data_file)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `python scripts/prepare_datasets.py` first.")
    dataset = HumanEvalDataset(str(path)) if spec.loader == "humaneval" else MBPPDataset(str(path))
    return list(dataset.iter_tasks())
