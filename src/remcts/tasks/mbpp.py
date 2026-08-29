from __future__ import annotations

import json

from remcts.core.state import ProblemSpec

from .base import TaskDataset


class MBPPDataset(TaskDataset):
    def load(self) -> list[ProblemSpec]:
        raw = self.data_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        tasks: list[ProblemSpec] = []
        for item in data:
            prompt = item.get("text") or item.get("prompt") or ""
            tests = "\n".join(item.get("test_list", [])) if isinstance(item.get("test_list"), list) else item.get("tests", "")
            tasks.append(
                ProblemSpec(
                    task_id=str(item.get("task_id", item.get("id", "unknown"))),
                    prompt=prompt,
                    entry_point=item.get("entry_point"),
                    tests=tests,
                    metadata=item,
                )
            )
        return tasks
