from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from remcts.core.state import ProblemSpec


class TaskDataset(ABC):
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    @abstractmethod
    def load(self) -> list[ProblemSpec]:
        raise NotImplementedError

    def iter_tasks(self):
        yield from self.load()
