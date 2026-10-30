from __future__ import annotations

import argparse
import json
from pathlib import Path

from remcts.config import load_config
from remcts.core.search_engine import SearchEngine
from remcts.core.state import ProblemSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ReMCTS on a single task.")
    parser.add_argument("--config", default="default", help="Packaged config name or path to a YAML file.")
    parser.add_argument("--task-id", default="demo-task")
    parser.add_argument("--prompt", help="Task prompt text.")
    parser.add_argument("--prompt-file", help="Path to a prompt file.")
    parser.add_argument("--tests", default="", help="Visible tests (assert lines); omit for the no_tests setting.")
    parser.add_argument("--tests-file", help="Path to a test script.")
    parser.add_argument("--entry-point", default=None)
    parser.add_argument("--api-alias", help="Alias of the API profile to load from llm.credentials_file.")
    return parser


def _load_text(inline_value: str | None, file_path: str | None) -> str:
    if inline_value:
        return inline_value
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    return ""


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    if args.api_alias:
        config.llm_api_alias = args.api_alias
    prompt = _load_text(args.prompt, args.prompt_file)
    if not prompt:
        parser.error("You must provide --prompt or --prompt-file")

    problem = ProblemSpec(
        task_id=args.task_id,
        prompt=prompt,
        entry_point=args.entry_point,
        tests=_load_text(args.tests, args.tests_file),
    )
    result = SearchEngine(config).run(problem)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
