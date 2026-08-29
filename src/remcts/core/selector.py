from __future__ import annotations

import math

from remcts.memory.global_memory import GlobalMemoryStore
from remcts.memory.path_memory import render_memory_for_prompt

from .node import SearchNode
from .state import ReflectionRecord, SearchConfig
from .tree import SearchTree


def compute_mean_value(node: SearchNode) -> float:
    return node.mean_value()


def compute_exploration_bonus(node: SearchNode, parent: SearchNode, config: SearchConfig) -> float:
    return config.exploration_c * math.sqrt(math.log(parent.visits + 1.0) / (node.visits + 1.0))


def compute_confidence_score(path_memory: list[ReflectionRecord]) -> float:
    if not path_memory:
        return 0.0
    severity_term = 1.0 - min(1.0, sum(m.severity for m in path_memory) / max(len(path_memory), 1))
    reward_term = sum(max(m.reward_delta, 0.0) for m in path_memory) / max(len(path_memory), 1)
    fix_hint_term = sum(1.0 for m in path_memory if m.fix_hint.strip()) / max(len(path_memory), 1)
    score = 0.5 * severity_term + 0.3 * min(reward_term, 1.0) + 0.2 * fix_hint_term
    return max(0.0, min(score, 1.0))


def compute_risk_penalty(
    path_memory: list[ReflectionRecord],
    global_memory: GlobalMemoryStore,
    top_k: int = 3,
) -> float:
    if not path_memory or len(global_memory) == 0:
        return 0.0
    query_text = render_memory_for_prompt(path_memory)
    retrieved = global_memory.retrieve(query_text, top_k=top_k, min_severity=0.7)
    if not retrieved:
        return 0.0
    return max(score for _, score in retrieved)


def compute_semantic_bonus(
    path_memory: list[ReflectionRecord],
    global_memory: GlobalMemoryStore,
    config: SearchConfig,
) -> tuple[float, float, float]:
    confidence = compute_confidence_score(path_memory)
    risk = compute_risk_penalty(path_memory, global_memory, top_k=config.global_memory_top_t)
    bonus = config.alpha * confidence - config.beta * risk
    return bonus, confidence, risk


def compute_ucb(
    node: SearchNode,
    parent: SearchNode,
    global_memory: GlobalMemoryStore,
    config: SearchConfig,
) -> float:
    mean_value = compute_mean_value(node)
    exploration_bonus = compute_exploration_bonus(node, parent, config)
    semantic_bonus, confidence, risk = compute_semantic_bonus(node.path_memory, global_memory, config)
    node.meta["mean_value"] = mean_value
    node.meta["exploration_bonus"] = exploration_bonus
    node.meta["confidence_score"] = confidence
    node.meta["risk_penalty"] = risk
    node.meta["semantic_bonus"] = semantic_bonus
    node.ucb_score = mean_value + exploration_bonus + semantic_bonus
    return node.ucb_score


def select_best_child(
    parent: SearchNode,
    tree: SearchTree,
    global_memory: GlobalMemoryStore,
    config: SearchConfig,
) -> SearchNode:
    children = [tree.get_node(child_id) for child_id in parent.children]
    if not children:
        return parent
    return max(children, key=lambda child: compute_ucb(child, parent, global_memory, config))


def select_expandable_node(
    tree: SearchTree,
    global_memory: GlobalMemoryStore,
    config: SearchConfig,
) -> SearchNode:
    node = tree.root
    while True:
        if node.depth >= config.max_depth:
            return node
        if not node.is_terminal and len(node.children) < config.branch_factor:
            return node
        if not node.children:
            return node
        node = select_best_child(node, tree, global_memory, config)
