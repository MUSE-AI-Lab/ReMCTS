from __future__ import annotations

from remcts.llm.client import BaseLLMClient
from remcts.llm.prompt_builder import build_expand_prompt, is_cpp
from remcts.llm.response_parser import extract_code
from remcts.memory.global_memory import GlobalMemoryStore
from remcts.sandbox.cpp_executor import strip_cpp_main_or_tests
from remcts.utils.hashing import stable_hash

from .node import SearchNode
from .state import CodeState, ProblemSpec, SearchConfig


class NodeExpander:
    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def expand(
        self,
        problem: ProblemSpec,
        node: SearchNode,
        global_memory: GlobalMemoryStore,
        config: SearchConfig,
    ) -> list[CodeState]:
        """Sample ``branch_factor`` distinct candidates conditioned on path and global memory."""
        query_text = node.state.stderr or node.state.code or problem.prompt
        retrieved = global_memory.retrieve(query_text, top_k=config.global_memory_top_t)
        prompt = build_expand_prompt(problem, node, retrieved, config.path_memory_k)

        candidates: list[CodeState] = []
        seen: set[str] = set()
        for _ in range(config.branch_factor):
            raw = self.llm_client.generate(
                prompt,
                temperature=config.llm_temperature,
                max_tokens=config.llm_max_tokens,
            )
            code = _normalize_candidate_code(problem, extract_code(raw).strip())
            if not code:
                continue
            digest = stable_hash(code)
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append(CodeState(code=code))
        return candidates


def _normalize_candidate_code(problem: ProblemSpec, code: str) -> str:
    """For C++ tasks, drop any test ``main`` and restore the required declaration and headers."""
    if not is_cpp(problem):
        return code
    code = strip_cpp_main_or_tests(code)
    entry = problem.entry_point or "solution"
    declaration = str(problem.metadata.get("declaration") or problem.prompt or "")
    if entry and f"{entry}(" not in code:
        code = f"{declaration.rstrip()}\n{code.strip()}"
    elif "#include" not in code:
        code = "#include <bits/stdc++.h>\nusing namespace std;\n" + code
    return code.strip()
