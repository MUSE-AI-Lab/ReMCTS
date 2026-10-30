from __future__ import annotations

import numpy as np


def cosine_similarity(vec_a: list[float] | None, vec_b: list[float] | None) -> float:
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)