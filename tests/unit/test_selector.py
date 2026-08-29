from remcts.core.node import SearchNode
from remcts.core.selector import compute_ucb
from remcts.core.state import CodeState, SearchConfig
from remcts.memory.global_memory import GlobalMemoryStore


def test_compute_ucb_runs():
    config = SearchConfig()
    parent = SearchNode(node_id="p", parent_id=None, depth=0, state=CodeState(code=""), visits=3)
    child = SearchNode(node_id="c", parent_id="p", depth=1, state=CodeState(code=""), visits=1, total_value=0.5)
    score = compute_ucb(child, parent, GlobalMemoryStore(), config)
    assert isinstance(score, float)
