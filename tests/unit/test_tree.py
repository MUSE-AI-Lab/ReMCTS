from remcts.core.state import CodeState, ProblemSpec, SearchConfig
from remcts.core.tree import SearchTree


def test_tree_add_root_and_child():
    tree = SearchTree(ProblemSpec(task_id="t1", prompt="demo"), SearchConfig())
    root = tree.add_root()
    child = tree.create_child(root, CodeState(code="print('hello')"), local_reward=0.5)
    assert tree.root_id == "root"
    assert child.parent_id == root.node_id
    assert tree.get_path(child.node_id)[0].node_id == "root"
