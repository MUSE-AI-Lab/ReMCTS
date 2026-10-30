from __future__ import annotations

from typing import Any

from remcts.llm.client import BaseLLMClient, create_llm_client
from remcts.memory.global_memory import GlobalMemoryStore
from remcts.memory.path_memory import build_path_memory
from remcts.memory.reflection_generator import ReflectionGenerator
from remcts.reward.scorer import score_candidate
from remcts.reward.semantic_scorer import SemanticScorer
from remcts.sandbox.executor import PythonExecutor
from remcts.sandbox.test_runner import run_problem_tests
from remcts.storage.artifact_store import ArtifactStore
from remcts.storage.run_logger import RunLogger

from .backprop import backpropagate
from .expander import NodeExpander
from .node import SearchNode
from .selector import select_expandable_node
from .state import ProblemSpec, SearchConfig, SearchResult
from .stop_criteria import make_stop_decision
from .tree import SearchTree


class SearchEngine:
    """Reflection-enhanced MCTS over candidate programs.

    Every iteration selects a node with the memory-augmented UCB score, expands it with
    ``branch_factor`` LLM candidates, executes and scores each candidate, writes a
    reflection for every candidate that is not accepted, and backpropagates the reward.
    """

    def __init__(
        self,
        config: SearchConfig,
        llm_client: BaseLLMClient | None = None,
        global_memory: GlobalMemoryStore | None = None,
        executor: PythonExecutor | None = None,
    ):
        self.config = config
        self.llm_client = llm_client or create_llm_client(config)
        self.global_memory = global_memory or GlobalMemoryStore()
        self.executor = executor or PythonExecutor()
        self.expander = NodeExpander(self.llm_client)
        self.reflection_generator = ReflectionGenerator(self.llm_client, config)
        self.semantic_scorer = SemanticScorer(self.llm_client, config)

    def run(self, problem: ProblemSpec) -> SearchResult:
        config = self.config
        self.llm_client.reset_usage()
        tree = SearchTree(problem, config)
        root = tree.add_root()

        run_dir = ArtifactStore(config.run_output_dir).create_run_dir(problem.task_id)
        logger = RunLogger(run_dir)
        logger.write_config(config.to_dict())
        logger.log_node(root)

        is_production = config.evaluation_mode.lower() == "production"
        complete_branch_before_accept = bool(config.production_complete_branch_before_accept)
        min_accept_iterations = max(1, int(config.production_min_accept_iterations))
        iterations = 0
        no_improve_rounds = 0
        best_reward = float("-inf")
        best_accepted: tuple[float, SearchNode, Any] | None = None
        stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, config)

        def finish(
            node: SearchNode,
            *,
            status: str,
            accepted: bool,
            needs_review: bool,
            stop_reason: str,
            best_reward: dict[str, Any] | None,
        ) -> SearchResult:
            summary = {
                "status": status,
                "iterations": iterations,
                "tree": tree.summary(),
                "memory_size": len(self.global_memory),
                "accepted": accepted,
                "needs_review": needs_review,
                "stop_reason": stop_reason,
                "best_reward": best_reward,
                "llm_usage": self.llm_client.usage_snapshot(),
            }
            logger.write_summary(summary)
            return SearchResult(
                success=node.state.success(),
                best_node_id=node.node_id,
                best_code=node.state.code,
                iterations=iterations,
                tree_size=tree.size(),
                run_dir=str(run_dir),
                summary=summary,
                accepted=accepted,
                needs_review=needs_review,
                stop_reason=stop_reason,
            )

        while not stop_decision.should_stop:
            iterations += 1
            tree.iteration = iterations
            parent = select_expandable_node(tree, self.global_memory, config)
            expandable = not parent.is_terminal and parent.depth < config.max_depth
            candidates = self.expander.expand(problem, parent, self.global_memory, config) if expandable else []
            if not candidates:
                parent.mark_terminal()
                no_improve_rounds += 1
                stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, config)
                continue

            improved = False
            produced_child = False
            for candidate in candidates:
                if tree.has_code(candidate.code):
                    continue
                exec_result = run_problem_tests(
                    problem,
                    candidate.code,
                    self.executor,
                    timeout_sec=config.timeout_sec,
                    max_output_chars=config.max_output_chars,
                )
                candidate.stdout = exec_result.stdout
                candidate.stderr = exec_result.stderr
                candidate.compile_ok = exec_result.compile_ok
                candidate.exec_ok = exec_result.exec_ok
                candidate.passed_tests = exec_result.passed_tests
                candidate.total_tests = exec_result.total_tests

                reward_result = score_candidate(
                    exec_result,
                    config,
                    problem=problem,
                    code=candidate.code,
                    semantic_scorer=self.semantic_scorer,
                    executor=self.executor,
                )
                accepted = bool(reward_result.details.get("accepted")) if is_production else candidate.success()
                reflection = None
                if not accepted:
                    reflection = self.reflection_generator.generate(
                        problem,
                        candidate.code,
                        exec_result,
                        reward_delta=reward_result.total_reward - parent.mean_value(),
                        source_task_id=problem.task_id,
                        source_node_id=parent.node_id,
                    )

                child = tree.create_child(
                    parent,
                    candidate,
                    path_memory=build_path_memory(
                        parent.path_memory,
                        reflection,
                        config.path_memory_k,
                        depth=parent.depth + 1,
                    ),
                    local_reward=reward_result.total_reward,
                    meta={
                        "execution": exec_result.to_dict(),
                        "reward": reward_result.to_dict(),
                        "prompt": problem.prompt,
                    },
                )
                produced_child = True
                backpropagate(tree, child.node_id, reward_result.total_reward)

                if reflection is not None:
                    self.global_memory.add(reflection)
                    logger.log_memory(reflection)
                child.meta["reflection"] = reflection.to_dict() if reflection is not None else None

                if reward_result.total_reward > best_reward:
                    best_reward = reward_result.total_reward
                    improved = True
                if accepted and iterations >= min_accept_iterations:
                    child.mark_terminal()
                logger.log_node(child)

                if not accepted:
                    continue
                if best_accepted is None or reward_result.total_reward > best_accepted[0]:
                    best_accepted = (reward_result.total_reward, child, reward_result)
                if complete_branch_before_accept or iterations < min_accept_iterations:
                    continue
                return finish(
                    child,
                    status="accepted" if is_production else "success",
                    accepted=True,
                    needs_review=False,
                    stop_reason="accepted_by_confidence" if is_production else "tests_passed",
                    best_reward=reward_result.to_dict(),
                )

            # With complete_branch_before_accept, the best accepted sibling is returned
            # once the whole branch has been generated and the minimum iteration is reached.
            if best_accepted is not None and iterations >= min_accept_iterations:
                _, node, reward_result = best_accepted
                node.mark_terminal()
                return finish(
                    node,
                    status="accepted",
                    accepted=True,
                    needs_review=False,
                    stop_reason="accepted_by_confidence_after_branch",
                    best_reward=reward_result.to_dict(),
                )

            if produced_child:
                no_improve_rounds = 0 if improved else no_improve_rounds + 1
            else:
                parent.mark_terminal()
                no_improve_rounds += 1
            stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, config)

        best = tree.best_solution()
        best_details = best.meta.get("reward", {}).get("details", {})
        needs_review = bool(stop_decision.needs_review) or (is_production and not best_details.get("accepted"))
        return finish(
            best,
            status="stopped",
            accepted=bool(stop_decision.accepted or best_details.get("accepted", False)),
            needs_review=needs_review,
            stop_reason=stop_decision.reason,
            best_reward=best.meta.get("reward"),
        )
