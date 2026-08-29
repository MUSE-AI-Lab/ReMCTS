from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import SearchConfig
from .tree import SearchTree


@dataclass(slots=True)
class StopDecision:
    should_stop: bool
    reason: str = ""
    accepted: bool = False
    needs_review: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def should_stop(
    tree: SearchTree,
    iterations: int,
    no_improve_rounds: int,
    config: SearchConfig,
) -> bool:
    return make_stop_decision(tree, iterations, no_improve_rounds, config).should_stop


def make_stop_decision(
    tree: SearchTree,
    iterations: int,
    no_improve_rounds: int,
    config: SearchConfig,
) -> StopDecision:
    min_accept_iterations = max(1, int(config.production_min_accept_iterations))
    if iterations >= config.max_iterations:
        return StopDecision(True, reason="max_iterations")
    if no_improve_rounds >= config.no_improve_patience:
        return StopDecision(True, reason="no_improve_patience")
    best = tree.best_solution() if tree.nodes else None
    if best and config.evaluation_mode.lower() == "production":
        reward_details = best.meta.get("reward", {}).get("details", {})
        if reward_details.get("accepted") and iterations >= min_accept_iterations:
            return StopDecision(
                True,
                reason="accepted_by_confidence",
                accepted=True,
                details={
                    "accept_score": reward_details.get("accept_score"),
                    "risk_score": reward_details.get("r_risk"),
                },
            )
        if reward_details.get("needs_review") and no_improve_rounds > 0:
            return StopDecision(
                True,
                reason="risk_or_uncertainty_requires_review",
                needs_review=True,
                details={
                    "accept_score": reward_details.get("accept_score"),
                    "risk_score": reward_details.get("r_risk"),
                    "uncertainty": reward_details.get("uncertainty"),
                },
            )
    if best and best.state.success() and iterations >= min_accept_iterations:
        return StopDecision(True, reason="tests_passed", accepted=True)
    if tree.nodes:
        leaves = list(tree.iter_leaves())
        if leaves and all(leaf.is_terminal or leaf.depth >= config.max_depth for leaf in leaves):
            if best and iterations < min_accept_iterations:
                if config.evaluation_mode.lower() == "production":
                    reward_details = best.meta.get("reward", {}).get("details", {})
                    if reward_details.get("accepted"):
                        return StopDecision(False)
                elif best.state.success():
                    return StopDecision(False)
            return StopDecision(True, reason="tree_exhausted")
    return StopDecision(False)
