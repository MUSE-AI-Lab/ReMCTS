import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("prepare_datasets", ROOT / "scripts" / "prepare_datasets.py")
prepare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prepare)


def test_humaneval_feedback_comes_from_prompt_doctests():
    prompt = '''def add(a, b):
    """Add two numbers.
    >>> add(1, 2)
    3
    >>> add(-1, 1)
    0
    """
'''
    # The expected output of the last example runs into the closing quotes, so that example is
    # skipped; HumanEval_feedback.jsonl (pinned by data/dataset_manifest.json) is built this way.
    assert prepare.extract_humaneval_feedback_tests(prompt) == ["assert add(1, 2) == 3"]
    assert prepare.extract_humaneval_feedback_tests(prompt.replace('    0\n    """', '    0\n\n    """')) == [
        "assert add(1, 2) == 3",
        "assert add(-1, 1) == 0",
    ]


def test_mbpp_feedback_keeps_the_first_assertion():
    item = prepare.normalize_mbpp_item(
        {
            "task_id": 11,
            "prompt": "Write a function to add two numbers.",
            "code": "import math\ndef add(a, b):\n    return a + b",
            "test_imports": ["import math"],
            "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 0) == 0"],
        }
    )
    assert item["entry_point"] == "add"
    feedback, evaluation, no_tests = prepare.split_mbpp_feedback_eval([item])
    assert feedback[0]["test_list"] == ["import math", "assert add(1, 2) == 3"]
    assert evaluation[0]["test_list"] == ["import math", "assert add(2, 2) == 4", "assert add(0, 0) == 0"]
    assert no_tests[0]["test_list"] == []


def test_cpp_entry_point_is_the_declared_function():
    declaration = "#include<stdio.h>\nusing namespace std;\nbool has_close_elements(vector<float> numbers, float threshold){\n"
    assert prepare.cpp_entry_point({"declaration": declaration}) == "has_close_elements"
    assert prepare.cpp_entry_point({"declaration": "", "prompt": ""}) == "solution"
