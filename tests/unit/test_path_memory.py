from remcts.core.state import ReflectionRecord
from remcts.memory.path_memory import build_path_memory


def make_reflection(idx: int) -> ReflectionRecord:
    return ReflectionRecord(
        reflection_id=f"r{idx}",
        summary=f"summary {idx}",
        error_type="AssertionError",
        root_cause=f"cause {idx}",
        fix_hint=f"fix {idx}",
        severity=0.5,
    )


def test_path_memory_respects_depth_and_k():
    memory = [make_reflection(1), make_reflection(2), make_reflection(3)]
    new_reflection = make_reflection(4)
    result = build_path_memory(memory, new_reflection, k=4, depth=2)
    assert len(result) == 3
    assert result[-1].reflection_id == "r4"
