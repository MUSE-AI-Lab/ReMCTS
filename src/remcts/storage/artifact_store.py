from __future__ import annotations

from pathlib import Path

from remcts.utils.time_utils import utc_timestamp


class ArtifactStore:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, task_id: str) -> Path:
        safe_task_id = task_id.replace("/", "_")
        run_dir = self.root_dir / f"{safe_task_id}-{utc_timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
