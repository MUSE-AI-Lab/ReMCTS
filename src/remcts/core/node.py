from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .state import CodeState, ReflectionRecord


@dataclass(slots=True)
class SearchNode:
    node_id: str
    parent_id: str | None
    depth: int
    state: CodeState
    total_value: float = 0.0
    visits: int = 0
    local_reward: float = 0.0
    path_memory: list[ReflectionRecord] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    is_terminal: bool = False
    ucb_score: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def mean_value(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children:
            self.children.append(child_id)

    def mark_terminal(self) -> None:
        self.is_terminal = True

    def update_stats(self, reward: float) -> None:
        self.visits += 1
        self.total_value += reward

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
