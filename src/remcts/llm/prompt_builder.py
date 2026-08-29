from __future__ import annotations

from remcts.memory.path_memory import render_memory_for_prompt

from remcts.core.node import SearchNode
from remcts.core.state import ProblemSpec, ReflectionRecord


def _render_retrieved_memories(retrieved: list[tuple[ReflectionRecord, float]]) -> str:
    if not retrieved:
        return "无全局相似失败经验。"
    lines = []
    for idx, (record, score) in enumerate(retrieved, start=1):
        lines.append(
            f"[{idx}] sim={score:.3f}; error_type={record.error_type}; root_cause={record.root_cause}; fix_hint={record.fix_hint}"
        )
    return "\n".join(lines)


def build_expand_prompt(
    problem: ProblemSpec,
    node: SearchNode,
    retrieved: list[tuple[ReflectionRecord, float]],
    k: int,
) -> str:
    language = str(problem.metadata.get("language") or "").lower()
    if language in {"cpp", "c++", "cxx"}:
        return _build_cpp_expand_prompt(problem, node, retrieved, k)

    local_memory = render_memory_for_prompt(node.path_memory[-k:])
    global_memory = _render_retrieved_memories(retrieved)
    entry_point = problem.entry_point or "solution"
    return f"""
你是一个代码修复与生成助手。请根据任务、当前候选代码、路径记忆和全局失败经验，生成一个新的完整可执行 Python 解答。

# 任务描述
{problem.prompt}

# 入口函数
{entry_point}

# 当前代码
{node.state.code or '# 当前尚无候选代码'}

# 局部路径反思记忆
{local_memory}

# 全局相似失败经验
{global_memory}

要求：
1. 只输出完整可执行代码；
2. 优先修复已知错误；
3. 处理边界条件；
4. 必须定义入口函数 `{entry_point}`；
5. 入口函数是主要验收接口，优先写成可直接调用的纯函数；
6. 不要添加不必要的标准输入、命令行主程序或交互逻辑；
7. 如果确实使用 sys、math、collections、heapq 等标准库，必须显式 import。
""".strip()


def _build_cpp_expand_prompt(
    problem: ProblemSpec,
    node: SearchNode,
    retrieved: list[tuple[ReflectionRecord, float]],
    k: int,
) -> str:
    local_memory = render_memory_for_prompt(node.path_memory[-k:])
    global_memory = _render_retrieved_memories(retrieved)
    entry_point = problem.entry_point or "solution"
    declaration = str(problem.metadata.get("declaration") or "")
    return f"""
You are a careful C++17 code repair and generation assistant. Generate a new complete C++17 solution for the task.

# Task
{problem.prompt}

# Required entry point
{entry_point}

# Required declaration / prefix
{declaration}

# Current candidate code
{node.state.code or '// no candidate yet'}

# Local reflection memory
{local_memory}

# Retrieved global failure memory
{global_memory}

Requirements:
1. Return only complete C++17 source code, with no Markdown fences and no explanation.
2. Define the required entry function `{entry_point}` with the exact signature implied by the declaration.
3. Include all required standard headers and `using namespace std;` when useful.
4. Do not define `main`, do not include tests, and do not read from stdin unless the prompt explicitly requires it.
5. Prefer simple deterministic code and handle edge cases.
""".strip()


def build_reflect_prompt(problem: ProblemSpec, code: str, execution_summary: str) -> str:
    return f"""
你是 ReMCTS 的失败反思模块。请根据任务、候选代码和执行反馈，生成一条短小但信息密度高的结构化反思。

只输出一个 JSON 对象，不要 Markdown，不要解释文字。JSON 必须包含：
{{
  "error_type": "错误类别，例如 AssertionError/TypeError/SyntaxError/LogicError",
  "failure_summary": "一句话概括失败现象",
  "root_cause": "指出代码为什么失败，不要只复述报错",
  "fix_hint": "给下一次生成的具体修复建议",
  "severity": 0.0到1.0之间的小数，越严重越接近1
}}

任务：{problem.prompt}
代码：
{code}
执行反馈：
{execution_summary}
""".strip()


def build_semantic_score_prompt(problem: ProblemSpec, code: str, execution_summary: str) -> str:
    return f"""
你是 ReMCTS 的语义奖励评估器。请评估当前代码思路与任务要求的接近程度。

只输出一个 JSON 对象，不要 Markdown，不要解释文字。JSON 格式：
{{"score": 0.0到1.0之间的小数, "reason": "一句话说明评分原因"}}

评分准则：
- 1.0 表示代码思路和边界处理都与任务高度匹配；
- 0.5 表示核心方向部分正确，但明显漏掉边界或细节；
- 0.0 表示代码与任务基本无关或完全不可用。

任务：{problem.prompt}
代码：
{code}
执行反馈：
{execution_summary}
""".strip()


def build_production_semantic_score_prompt(problem: ProblemSpec, code: str, execution_summary: str) -> str:
    entry = problem.entry_point or problem.metadata.get("entry_point") or "solution"
    return f"""
你是 ReMCTS 的生产环境代码验收评审器。当前任务没有人工编写的完整测试用例，因此你必须基于需求、接口约束、代码和执行证据进行保守评估。

只输出一个 JSON 对象，不要 Markdown，不要解释文字。JSON 格式：
{{
  "requirement_coverage": 0.0到1.0之间的小数,
  "api_contract_match": 0.0到1.0之间的小数,
  "context_compatibility": 0.0到1.0之间的小数,
  "maintainability": 0.0到1.0之间的小数,
  "risk": 0.0到1.0之间的小数,
  "confidence": 0.0到1.0之间的小数,
  "reason": "一句话说明评分原因"
}}

评分要求：
- requirement_coverage 衡量代码是否覆盖需求本身；
- api_contract_match 衡量是否定义并遵守入口函数 `{entry}` 及接口约定；
- context_compatibility 衡量是否容易嵌入现有代码和调用方式；
- maintainability 衡量实现是否简洁、可读、改动最小；
- risk 衡量安全、隐私、破坏性操作、不可控依赖和复杂度风险；
- confidence 衡量你对以上判断的置信度。证据不足时必须降低 confidence。

验收边界：
- 入口函数 `{entry}` 是主要生产接口；
- 若任务没有明确要求命令行或标准输入，`if __name__ == "__main__"`、stdin 读取等附加路径不应覆盖入口函数证据；
- 如果附加路径存在缺失 import 或输入解析问题，可以降低 context_compatibility 或 maintainability，但不要在入口函数证据充分通过时把 requirement_coverage 评为失败；
- 如果缺失 import 会影响入口函数的正常调用，则必须明显降低 api_contract_match 和 requirement_coverage。

任务：{problem.prompt}
入口函数：{entry}
代码：
{code}
执行与证据摘要：
{execution_summary}
""".strip()
