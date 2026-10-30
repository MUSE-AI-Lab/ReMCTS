from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .node import SearchNode
from .state import SearchConfig
from .tree import SearchTree


@dataclass(slots=True)
class StopDecision:
    should_stop: bool
    reason: str = ""
    accepted: bool = False
    needs_review: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def _reward_details(node: SearchNode) -> dict[str, Any]:
    return node.meta.get("reward", {}).get("details", {})


def make_stop_decision(
    tree: SearchTree,
    iterations: int,
    no_improve_rounds: int,
    config: SearchConfig,
) -> StopDecision:
    min_accept_iterations = max(1, int(config.production_min_accept_iterations))
    is_production = config.evaluation_mode.lower() == "production"
    if iterations >= config.max_iterations:
        return StopDecision(True, reason="max_iterations")
    if no_improve_rounds >= config.no_improve_patience:
        return StopDecision(True, reason="no_improve_patience")
    best = tree.best_solution() if tree.nodes else None
    if best and is_production:
        details = _reward_details(best)
        if details.get("accepted") and iterations >= min_accept_iterations:
            return StopDecision(
                True,
                reason="accepted_by_confidence",
                accepted=True,
                details={"accept_score": details.get("accept_score"), "risk_score": details.get("r_risk")},
            )
        if details.get("needs_review") and no_improve_rounds > 0:
            return StopDecision(
                True,
                reason="risk_or_uncertainty_requires_review",
                needs_review=True,
                details={
                    "accept_score": details.get("accept_score"),
                    "risk_score": details.get("r_risk"),
                    "uncertainty": details.get("uncertainty"),
                },
            )
    if best and best.state.success() and iterations >= min_accept_iterations:
        return StopDecision(True, reason="tests_passed", accepted=True)
    if tree.nodes:
        leaves = list(tree.iter_leaves())
        if leaves and all(leaf.is_terminal or leaf.depth >= config.max_depth for leaf in leaves):
            # Keep searching until the minimum iteration when an acceptable candidate exists.
            if best and iterations < min_accept_iterations:
                acceptable = _reward_details(best).get("accepted") if is_production else best.state.success()
                if acceptable:
                    return StopDecision(False)
            return StopDecision(True, reason="tree_exhausted")
    return StopDecision(False)
