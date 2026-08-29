from __future__ import annotations

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
from .selector import select_expandable_node
from .state import ProblemSpec, SearchConfig, SearchResult
from .stop_criteria import make_stop_decision
from .tree import SearchTree


class SearchEngine:
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
        self.llm_client.reset_usage()
        tree = SearchTree(problem, self.config)
        root = tree.add_root()

        artifact_store = ArtifactStore(self.config.run_output_dir)
        run_dir = artifact_store.create_run_dir(problem.task_id)
        logger = RunLogger(run_dir)
        logger.write_config(self.config.to_dict())
        logger.log_node(root)

        iterations = 0
        no_improve_rounds = 0
        best_reward = float("-inf")
        is_production = self.config.evaluation_mode.lower() == "production"
        complete_branch_before_accept = bool(self.config.production_complete_branch_before_accept)
        min_accept_iterations = max(1, int(self.config.production_min_accept_iterations))
        best_accepted_node_id: str | None = None
        best_accepted_reward: float = float("-inf")
        best_accepted_reward_result = None
        stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)

        while not stop_decision.should_stop:
            iterations += 1
            tree.iteration = iterations
            parent = select_expandable_node(tree, self.global_memory, self.config)
            if parent.is_terminal or parent.depth >= self.config.max_depth:
                parent.mark_terminal()
                no_improve_rounds += 1
                stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)
                continue
            candidates = self.expander.expand(problem, parent, self.global_memory, self.config)

            if not candidates:
                parent.mark_terminal()
                no_improve_rounds += 1
                stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)
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
                    timeout_sec=self.config.timeout_sec,
                    max_output_chars=self.config.max_output_chars,
                )
                candidate.stdout = exec_result.stdout
                candidate.stderr = exec_result.stderr
                candidate.compile_ok = exec_result.compile_ok
                candidate.exec_ok = exec_result.exec_ok
                candidate.passed_tests = exec_result.passed_tests
                candidate.total_tests = exec_result.total_tests

                reward_result = score_candidate(
                    exec_result,
                    self.config,
                    problem=problem,
                    code=candidate.code,
                    semantic_scorer=self.semantic_scorer,
                    executor=self.executor,
                )
                reward_details = reward_result.details
                candidate_accepted = bool(reward_details.get("accepted")) if is_production else candidate.success()
                reflection = None
                if not candidate_accepted:
                    reflection = self.reflection_generator.generate(
                        problem,
                        candidate.code,
                        exec_result,
                        reward_delta=reward_result.total_reward - parent.mean_value(),
                        source_task_id=problem.task_id,
                        source_node_id=parent.node_id,
                    )

                path_memory = build_path_memory(
                    parent.path_memory,
                    reflection,
                    self.config.path_memory_k,
                    depth=parent.depth + 1,
                )
                child = tree.create_child(
                    parent,
                    candidate,
                    path_memory=path_memory,
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

                if candidate_accepted and iterations >= min_accept_iterations:
                    child.mark_terminal()
                logger.log_node(child)

                if candidate_accepted:
                    if reward_result.total_reward > best_accepted_reward:
                        best_accepted_reward = reward_result.total_reward
                        best_accepted_node_id = child.node_id
                        best_accepted_reward_result = reward_result
                    if complete_branch_before_accept or iterations < min_accept_iterations:
                        continue
                    status = "accepted" if is_production else "success"
                    summary = {
                        "status": status,
                        "iterations": iterations,
                        "tree": tree.summary(),
                        "memory_size": len(self.global_memory),
                        "accepted": True,
                        "needs_review": False,
                        "stop_reason": "accepted_by_confidence" if is_production else "tests_passed",
                        "best_reward": reward_result.to_dict(),
                        "llm_usage": self.llm_client.usage_snapshot(),
                    }
                    logger.write_summary(summary)
                    return SearchResult(
                        success=candidate.success(),
                        best_node_id=child.node_id,
                        best_code=child.state.code,
                        iterations=iterations,
                        tree_size=tree.size(),
                        run_dir=str(run_dir),
                        summary=summary,
                        accepted=True,
                        needs_review=False,
                        stop_reason=summary["stop_reason"],
                    )

            if best_accepted_node_id is not None and iterations >= min_accept_iterations:
                child = tree.get_node(best_accepted_node_id)
                child.mark_terminal()
                summary = {
                    "status": "accepted",
                    "iterations": iterations,
                    "tree": tree.summary(),
                    "memory_size": len(self.global_memory),
                    "accepted": True,
                    "needs_review": False,
                    "stop_reason": "accepted_by_confidence_after_branch",
                    "best_reward": best_accepted_reward_result.to_dict() if best_accepted_reward_result else None,
                    "llm_usage": self.llm_client.usage_snapshot(),
                }
                logger.write_summary(summary)
                return SearchResult(
                    success=child.state.success(),
                    best_node_id=child.node_id,
                    best_code=child.state.code,
                    iterations=iterations,
                    tree_size=tree.size(),
                    run_dir=str(run_dir),
                    summary=summary,
                    accepted=True,
                    needs_review=False,
                    stop_reason=summary["stop_reason"],
                )

            if not produced_child:
                parent.mark_terminal()
                no_improve_rounds += 1
                stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)
                continue
            no_improve_rounds = 0 if improved else no_improve_rounds + 1
            stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)

        stop_decision = make_stop_decision(tree, iterations, no_improve_rounds, self.config)
        best = tree.best_solution()
        best_reward_details = best.meta.get("reward", {}).get("details", {})
        needs_review = bool(stop_decision.needs_review)
        if is_production and not bool(best_reward_details.get("accepted")):
            needs_review = True
        summary = {
            "status": "stopped",
            "iterations": iterations,
            "tree": tree.summary(),
            "memory_size": len(self.global_memory),
            "accepted": bool(stop_decision.accepted or best_reward_details.get("accepted", False)),
            "needs_review": needs_review,
            "stop_reason": stop_decision.reason,
            "best_reward": best.meta.get("reward"),
            "llm_usage": self.llm_client.usage_snapshot(),
        }
        logger.write_summary(summary)
        return SearchResult(
            success=best.state.success(),
            best_node_id=best.node_id,
            best_code=best.state.code,
            iterations=iterations,
            tree_size=tree.size(),
            run_dir=str(run_dir),
            summary=summary,
            accepted=summary["accepted"],
            needs_review=needs_review,
            stop_reason=stop_decision.reason,
        )
