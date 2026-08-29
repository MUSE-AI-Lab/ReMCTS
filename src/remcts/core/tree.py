from __future__ import annotations

from typing import Iterable
from uuid import uuid4

from remcts.utils.hashing import stable_hash

from .node import SearchNode
from .state import CodeState, ProblemSpec, ReflectionRecord, SearchConfig


class SearchTree:
    def __init__(self, problem: ProblemSpec, config: SearchConfig):
        self.problem = problem
        self.config = config
        self.nodes: dict[str, SearchNode] = {}
        self.code_hashes: set[str] = set()
        self.root_id: str | None = None
        self.iteration: int = 0

    @property
    def root(self) -> SearchNode:
        if self.root_id is None:
            raise RuntimeError("Root node has not been initialized.")
        return self.nodes[self.root_id]

    def add_root(self) -> SearchNode:
        root = SearchNode(
            node_id="root",
            parent_id=None,
            depth=0,
            state=CodeState(code=""),
            meta={"task_id": self.problem.task_id, "prompt": self.problem.prompt},
        )
        self.nodes[root.node_id] = root
        self.root_id = root.node_id
        return root

    def get_node(self, node_id: str) -> SearchNode:
        return self.nodes[node_id]

    def add_child(self, parent: SearchNode, child: SearchNode) -> None:
        self.nodes[child.node_id] = child
        if child.state.code:
            self.code_hashes.add(stable_hash(child.state.code))
        parent.add_child(child.node_id)

    def has_code(self, code: str) -> bool:
        return bool(code) and stable_hash(code) in self.code_hashes

    def create_child(
        self,
        parent: SearchNode,
        state: CodeState,
        path_memory: list[ReflectionRecord] | None = None,
        local_reward: float = 0.0,
        meta: dict | None = None,
    ) -> SearchNode:
        child = SearchNode(
            node_id=f"node-{uuid4().hex[:12]}",
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            state=state,
            local_reward=local_reward,
            path_memory=path_memory or [],
            meta=meta or {},
        )
        self.add_child(parent, child)
        return child

    def get_path(self, node_id: str) -> list[SearchNode]:
        path: list[SearchNode] = []
        current = self.get_node(node_id)
        while True:
            path.append(current)
            if current.parent_id is None:
                break
            current = self.get_node(current.parent_id)
        return list(reversed(path))

    def iter_leaves(self) -> Iterable[SearchNode]:
        for node in self.nodes.values():
            if not node.children:
                yield node

    def best_solution(self) -> SearchNode:
        def sort_key(node: SearchNode) -> tuple[float, float, float, float]:
            success = float(node.state.success())
            return (success, node.local_reward, node.mean_value(), -float(node.depth))

        return max(self.nodes.values(), key=sort_key)

    def size(self) -> int:
        return len(self.nodes)

    def summary(self) -> dict:
        return {
            "task_id": self.problem.task_id,
            "iteration": self.iteration,
            "tree_size": self.size(),
            "best_node_id": self.best_solution().node_id if self.nodes else None,
        }
