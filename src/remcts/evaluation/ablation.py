from __future__ import annotations

from dataclasses import replace

from remcts.core.state import SearchConfig


def build_ablation_configs(base: SearchConfig) -> dict[str, SearchConfig]:
    return {
        "full": base,
        "no_global_memory": replace(base, global_memory_top_t=0),
        "no_path_memory": replace(base, path_memory_k=0),
        "sparse_reward": replace(base, w_unit=1.0, w_exec=0.0, w_semantic=0.0),
        "no_semantic_bonus": replace(base, alpha=0.0, beta=0.0),
    }
