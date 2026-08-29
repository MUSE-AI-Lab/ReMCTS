from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from remcts.core.node import SearchNode
from remcts.core.state import ReflectionRecord

from .jsonl_writer import JsonlWriter


class RunLogger:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.node_writer = JsonlWriter(self.run_dir / "nodes.jsonl")
        self.memory_writer = JsonlWriter(self.run_dir / "memory.jsonl")

    def write_config(self, config: dict[str, Any]) -> None:
        (self.run_dir / "config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def log_node(self, node: SearchNode) -> None:
        self.node_writer.write(node.to_dict())

    def log_memory(self, record: ReflectionRecord) -> None:
        self.memory_writer.write(record.to_dict())

    def write_summary(self, summary: dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
