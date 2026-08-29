from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def write(self, item: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
