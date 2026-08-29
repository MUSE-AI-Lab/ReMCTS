from __future__ import annotations

import json

from remcts.core.state import ProblemSpec

from .base import TaskDataset


class HumanEvalDataset(TaskDataset):
    def load(self) -> list[ProblemSpec]:
        tasks: list[ProblemSpec] = []
        for line in self.data_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            tasks.append(
                ProblemSpec(
                    task_id=item.get("task_id", "unknown"),
                    prompt=item.get("prompt", ""),
                    entry_point=item.get("entry_point"),
                    tests=item.get("test", ""),
                    metadata=item,
                )
            )
        return tasks
