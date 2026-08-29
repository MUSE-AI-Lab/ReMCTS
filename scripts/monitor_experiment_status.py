#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path


MEMORY_SPECS = [
    (model, ablation, dataset, size)
    for model in ["qwen3_8b", "qwen3_32b", "deepseek_v4_flash"]
    for ablation in ["no_global_memory", "no_path_memory"]
    for dataset, size in [
        ("humaneval_full_known", 164),
        ("humaneval_full_no_tests", 164),
        ("mbpp_sanitized_full_known", 427),
        ("mbpp_sanitized_full_no_tests", 427),
    ]
]


KNOWN_SPECS = [
    (model, method, dataset, size)
    for model in ["qwen3_8b", "qwen3_32b", "deepseek_v4_flash"]
    for method in ["reflection_style", "mcts_style"]
    for dataset, size in [
        ("humaneval_full_known", 164),
        ("mbpp_sanitized_full_known", 427),
    ]
]


def read_completed(report_dir: Path, label: str, size: int) -> tuple[int, str]:
    final = report_dir / f"{label}.json"
    progress = report_dir / f"{label}.progress.json"
    try:
        if final.exists():
            data = json.loads(final.read_text(encoding="utf-8"))
            completed = int(data.get("completed") or size)
            status = str(data.get("status") or ("complete" if completed >= size else "partial"))
            return min(completed, size), status
        if progress.exists():
            data = json.loads(progress.read_text(encoding="utf-8"))
            completed = int(data.get("completed") or 0)
            return min(completed, size), "running" if completed else "started"
    except Exception as exc:
        return 0, f"read_error:{type(exc).__name__}"
    return 0, "pending"


def summarize_specs(title: str, report_dir: Path, specs: list[tuple[str, str, str, int]]) -> None:
    print(f"[{title}] report={report_dir}")
    if not report_dir.exists():
        print("  report_dir_missing")
        return
    for group in ["qwen", "deepseek"]:
        total = 0
        done = 0
        spec_done = 0
        active: list[tuple[str, int, int, str]] = []
        for model, method, dataset, size in specs:
            is_qwen = model.startswith("qwen")
            if (group == "qwen") != is_qwen:
                continue
            total += size
            if title == "memory":
                label = f"{model}_{method}_remcts_{dataset}"
            else:
                label = f"{model}_{method}_{dataset}"
            completed, status = read_completed(report_dir, label, size)
            done += completed
            if completed >= size and status == "complete":
                spec_done += 1
            elif completed > 0:
                active.append((label, completed, size, status))
        pct = (done / total * 100.0) if total else 0.0
        print(f"  [{group}] {done}/{total} {pct:.1f}% remaining={total-done} specs_done={spec_done}")
        for label, completed, size, status in active[:20]:
            print(f"    {completed}/{size} {status} {label}")


def print_log_tail(label: str, path: Path, tail: int) -> None:
    print(f"[{label} log] {path}")
    if not path.exists():
        print("  log_missing")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    task_lines = [line for line in lines if line.startswith("[task]")]
    done_lines = [line for line in lines if line.startswith("[done]")]
    print(f"  task_lines={len(task_lines)} done_lines={len(done_lines)}")
    for line in lines[-tail:]:
        print(f"  {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ReMCTS experiment monitor status.")
    parser.add_argument("--memory-report", type=Path, required=True)
    parser.add_argument("--known-report", type=Path, required=True)
    parser.add_argument("--qwen-log", type=Path, required=True)
    parser.add_argument("--deepseek-log", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=6)
    args = parser.parse_args()

    print(f"===== monitor tick {datetime.now().strftime('%F %T')} =====")
    summarize_specs("memory", args.memory_report, MEMORY_SPECS)
    summarize_specs("known", args.known_report, KNOWN_SPECS)
    print_log_tail("qwen", args.qwen_log, args.tail)
    print_log_tail("deepseek", args.deepseek_log, args.tail)


if __name__ == "__main__":
    main()
