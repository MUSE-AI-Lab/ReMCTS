#!/usr/bin/env python3
"""Download the official benchmark files and derive the search / hidden-evaluation splits.

    python scripts/prepare_datasets.py [--force] [--skip-download]

Outputs go to data/ (ignored by Git); data/dataset_manifest.json records URLs, sizes and SHA-256.
"""
from __future__ import annotations

import argparse
import ast
import doctest
import gzip
import hashlib
import json
import re
import shutil
import urllib.request
from ast import FunctionDef
from collections.abc import Iterable
from pathlib import Path
from typing import Any
import warnings


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

SOURCES = {
    "humaneval": {
        "url": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "path": DATA_DIR / "humaneval" / "HumanEval.jsonl.gz",
    },
    "mbpp_full": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/mbpp.jsonl",
        "path": DATA_DIR / "mbpp" / "mbpp.jsonl",
    },
    "mbpp_sanitized": {
        "url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
        "path": DATA_DIR / "mbpp" / "sanitized-mbpp.json",
    },
    "humaneval_plus": {
        "url": "https://raw.githubusercontent.com/evalplus/humanevalplus_release/main/HumanEvalPlus.jsonl.gz",
        "path": DATA_DIR / "evalplus" / "HumanEvalPlus.jsonl.gz",
    },
    "mbpp_plus": {
        "url": "https://raw.githubusercontent.com/evalplus/mbppplus_release/main/MbppPlus.jsonl.gz",
        "path": DATA_DIR / "evalplus" / "MbppPlus.jsonl.gz",
    },
    "humanevalx_cpp": {
        "url": (
            "https://raw.githubusercontent.com/THUDM/CodeGeeX/main/"
            "codegeex/benchmark/humaneval-x/cpp/data/humaneval_cpp.jsonl.gz"
        ),
        "path": DATA_DIR / "humaneval_x" / "humaneval_cpp.jsonl.gz",
    },
}

IGNORED_CALL_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "float",
    "frozenset",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}


def download(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def gunzip_to_jsonl(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8") as src, destination.open("w", encoding="utf-8", newline="\n") as dst:
        shutil.copyfileobj(src, dst)


def parse(code: str) -> ast.Module:
    # Reference solutions contain regex strings such as "\w"; their SyntaxWarnings are noise here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(code)


def first_function_name(code: str) -> str | None:
    try:
        tree = parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, FunctionDef):
            return node.name
    return None


def first_assert_call_name(assertion: str) -> str | None:
    try:
        tree = parse(assertion)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        func = getattr(node, "func", None)
        name = getattr(func, "id", None)
        if name and name not in IGNORED_CALL_NAMES:
            return name
    return None


def split_setup_and_asserts(lines: Iterable[str]) -> tuple[list[str], list[str]]:
    setup: list[str] = []
    asserts: list[str] = []
    for raw_line in lines:
        line = str(raw_line).strip()
        if not line:
            continue
        if line.startswith("assert "):
            asserts.append(line)
        else:
            setup.append(line)
    return setup, asserts


def mbpp_setup_lines(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    setup_code = str(item.get("test_setup_code") or "").strip()
    if setup_code:
        lines.extend(line for line in setup_code.splitlines() if line.strip())
    imports = item.get("test_imports") or []
    if isinstance(imports, list):
        lines.extend(str(line).strip() for line in imports if str(line).strip())
    return lines


def normalize_mbpp_item(item: dict[str, Any], *, test_list: list[str] | None = None) -> dict[str, Any]:
    setup, assertions = split_setup_and_asserts(test_list if test_list is not None else item.get("test_list", []))
    setup = [*mbpp_setup_lines(item), *setup]
    code = str(item.get("code") or item.get("canonical_solution") or "")
    entry_point = item.get("entry_point") or first_function_name(code)
    if not entry_point and assertions:
        entry_point = first_assert_call_name(assertions[0])
    prompt = item.get("prompt") or item.get("text") or ""
    normalized = {
        "task_id": item.get("task_id"),
        "prompt": prompt,
        "text": prompt,
        "entry_point": entry_point,
        "test_list": [*setup, *assertions],
        "source": "google-research/mbpp",
    }
    challenge = item.get("challenge_test_list")
    if isinstance(challenge, list):
        normalized["challenge_test_list"] = challenge
    return normalized


def split_mbpp_feedback_eval(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feedback_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    no_test_rows: list[dict[str, Any]] = []
    for item in items:
        setup, assertions = split_setup_and_asserts(item.get("test_list", []))
        feedback_asserts = assertions[:1]
        eval_asserts = assertions[1:]
        feedback = dict(item)
        feedback["test_list"] = [*setup, *feedback_asserts]
        feedback["split"] = "feedback_first_assertion"
        feedback_rows.append(feedback)

        eval_item = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "text": item["text"],
            "entry_point": item.get("entry_point"),
            "test_list": [*setup, *eval_asserts],
            "split": "eval_remaining_assertions",
            "source": item.get("source"),
        }
        eval_rows.append(eval_item)

        no_test = dict(item)
        no_test["test_list"] = []
        no_test["split"] = "no_explicit_tests"
        no_test_rows.append(no_test)
    return feedback_rows, eval_rows, no_test_rows


def extract_humaneval_feedback_tests(prompt: str) -> list[str]:
    parser = doctest.DocTestParser()
    tests: list[str] = []
    for example in parser.get_examples(prompt):
        source = example.source.strip()
        expected = example.want.strip()
        if not source or not expected:
            continue
        if "\n" in source or "\n" in expected:
            continue
        tests.append(f"assert {source} == {expected}")
    return tests


def prepare_humaneval(manifest: dict[str, Any]) -> None:
    gz_path = SOURCES["humaneval"]["path"]
    jsonl_path = DATA_DIR / "humaneval" / "HumanEval.jsonl"
    gunzip_to_jsonl(gz_path, jsonl_path)

    rows = read_jsonl(jsonl_path)
    feedback_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    no_test_rows: list[dict[str, Any]] = []
    for item in rows:
        base = {
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "entry_point": item.get("entry_point"),
        }
        feedback_tests = extract_humaneval_feedback_tests(item.get("prompt", ""))
        feedback_rows.append(
            {
                **base,
                "test": "\n".join(feedback_tests),
                "feedback_source": "prompt_doctest",
                "feedback_tests_count": len(feedback_tests),
            }
        )
        eval_rows.append({**base, "test": item.get("test", ""), "split": "hidden_eval"})
        no_test_rows.append({**base, "test": "", "split": "no_explicit_tests"})

    outputs = {
        "HumanEval_feedback.jsonl": feedback_rows,
        "HumanEval_eval.jsonl": eval_rows,
        "HumanEval_no_tests.jsonl": no_test_rows,
    }
    for name, out_rows in outputs.items():
        path = DATA_DIR / "humaneval" / name
        write_jsonl(path, out_rows)
        manifest["generated"].append(describe_file(path, rows=len(out_rows)))


def prepare_mbpp(manifest: dict[str, Any]) -> None:
    full_raw = read_jsonl(SOURCES["mbpp_full"]["path"])
    sanitized_raw = json.loads(SOURCES["mbpp_sanitized"]["path"].read_text(encoding="utf-8"))

    datasets = {
        "mbpp": [normalize_mbpp_item(item) for item in full_raw],
        "mbpp_test_11_510": [
            normalize_mbpp_item(item) for item in full_raw if 11 <= int(item.get("task_id", -1)) <= 510
        ],
        "mbpp_sanitized": [normalize_mbpp_item(item) for item in sanitized_raw],
        "mbpp_sanitized_test_11_510": [
            normalize_mbpp_item(item) for item in sanitized_raw if 11 <= int(item.get("task_id", -1)) <= 510
        ],
    }

    for stem, items in datasets.items():
        oracle_path = DATA_DIR / "mbpp" / f"{stem}.json"
        write_json(oracle_path, items)
        manifest["generated"].append(describe_file(oracle_path, rows=len(items)))

        feedback, eval_rows, no_tests = split_mbpp_feedback_eval(items)
        for suffix, rows in {
            "feedback_split": feedback,
            "eval_split": eval_rows,
            "no_tests": no_tests,
        }.items():
            path = DATA_DIR / "mbpp" / f"{stem}_{suffix}.json"
            write_json(path, rows)
            manifest["generated"].append(describe_file(path, rows=len(rows)))


def prepare_evalplus(manifest: dict[str, Any]) -> None:
    for source_name, output_name in {
        "humaneval_plus": "HumanEvalPlus.jsonl",
        "mbpp_plus": "MbppPlus.jsonl",
    }.items():
        gz_path = SOURCES[source_name]["path"]
        jsonl_path = DATA_DIR / "evalplus" / output_name
        gunzip_to_jsonl(gz_path, jsonl_path)
        rows = read_jsonl(jsonl_path)
        manifest["generated"].append(describe_file(jsonl_path, rows=len(rows)))

    mbpp_plus_jsonl = DATA_DIR / "evalplus" / "MbppPlus.jsonl"
    mbpp_plus_rows = read_jsonl(mbpp_plus_jsonl)
    mbpp_plus_for_loader = []
    for item in mbpp_plus_rows:
        mbpp_plus_for_loader.append(
            {
                "task_id": item.get("task_id"),
                "prompt": item.get("prompt", ""),
                "text": item.get("prompt", ""),
                "entry_point": item.get("entry_point"),
                "test_list": [line.strip() for line in str(item.get("assertion", "")).splitlines() if line.strip()],
                "source": "evalplus/mbppplus_release",
            }
        )
    path = DATA_DIR / "evalplus" / "MbppPlus_base_assertions.json"
    write_json(path, mbpp_plus_for_loader)
    manifest["generated"].append(describe_file(path, rows=len(mbpp_plus_for_loader)))


def cpp_entry_point(item: dict[str, Any]) -> str:
    """Name of the last function defined in the HumanEval-X declaration (or prompt)."""
    pattern = r"([A-Za-z_][A-Za-z0-9_]*)\s*\([^;{}]*\)\s*\{"
    for text in (item.get("declaration"), item.get("prompt")):
        matches = re.findall(pattern, str(text or ""))
        if matches:
            return matches[-1]
    return "solution"


def prepare_humanevalx_cpp(manifest: dict[str, Any]) -> None:
    with gzip.open(SOURCES["humanevalx_cpp"]["path"], "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    feedback_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    for item in rows:
        base = {
            "task_id": item["task_id"],
            "prompt": item.get("prompt", ""),
            "declaration": item.get("declaration", ""),
            "entry_point": cpp_entry_point(item),
            "language": "cpp",
        }
        feedback_rows.append({**base, "test": item.get("example_test", ""), "split": "feedback_example_test"})
        eval_rows.append({**base, "test": item.get("test", ""), "split": "hidden_eval"})
    for name, out_rows in (("humaneval_cpp_feedback.jsonl", feedback_rows), ("humaneval_cpp_eval.jsonl", eval_rows)):
        path = DATA_DIR / "humaneval_x" / name
        write_jsonl(path, out_rows)
        manifest["generated"].append(describe_file(path, rows=len(out_rows)))


def describe_file(path: Path, *, rows: int | None = None, source_url: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    if rows is not None:
        item["rows"] = rows
    if source_url is not None:
        item["source_url"] = source_url
    return item


def build_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "description": "Datasets prepared for the ReMCTS experiments.",
        "sources": [],
        "generated": [],
        "split_notes": [
            "HumanEval_feedback uses only doctest-style examples parsed from the prompt.",
            "HumanEval_eval keeps the original HumanEval test field for offline hidden evaluation.",
            "MBPP feedback_split keeps setup/import lines plus the first assertion as feedback.",
            "MBPP eval_split keeps setup/import lines plus the remaining assertions for offline evaluation.",
            "no_tests variants remove all explicit assertions for production-mode experiments.",
            "EvalPlus raw files are intended for robustness evaluation with an EvalPlus-compatible evaluator.",
            "HumanEval-X C++ feedback keeps the public example_test; eval keeps the official test field.",
        ],
    }
    for name, source in SOURCES.items():
        path = source["path"]
        manifest["sources"].append(describe_file(path, source_url=source["url"]))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and derive datasets for ReMCTS experiments.")
    parser.add_argument("--force", action="store_true", help="Re-download official source files.")
    parser.add_argument("--skip-download", action="store_true", help="Only regenerate derived files from local sources.")
    args = parser.parse_args()

    if not args.skip_download:
        for source in SOURCES.values():
            download(source["url"], source["path"], force=args.force)

    for source in SOURCES.values():
        if not source["path"].exists():
            raise FileNotFoundError(f"Missing source dataset: {source['path']}")

    manifest = build_manifest()
    prepare_humaneval(manifest)
    prepare_mbpp(manifest)
    prepare_evalplus(manifest)
    prepare_humanevalx_cpp(manifest)
    manifest_path = DATA_DIR / "dataset_manifest.json"
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    print(f"Prepared {len(manifest['sources'])} source files and {len(manifest['generated'])} derived files.")


if __name__ == "__main__":
    main()
