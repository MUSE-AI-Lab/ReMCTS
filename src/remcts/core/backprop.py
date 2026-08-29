from __future__ import annotations

from .tree import SearchTree


def backpropagate(tree: SearchTree, node_id: str, reward: float) -> None:
    current = tree.get_node(node_id)
    while True:
        current.update_stats(reward)
        if current.parent_id is None:
            break
        current = tree.get_node(current.parent_id)
