from __future__ import annotations

from remcts.core.search_engine import SearchEngine
from remcts.core.state import SearchResult
from remcts.tasks.base import TaskDataset


class BenchmarkRunner:
    def __init__(self, engine: SearchEngine):
        self.engine = engine

    def run_dataset(self, dataset: TaskDataset) -> list[SearchResult]:
        results: list[SearchResult] = []
        for problem in dataset.iter_tasks():
            results.append(self.engine.run(problem))
        return results
