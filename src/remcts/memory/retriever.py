from __future__ import annotations

from typing import Callable, Iterable, TypeVar

import numpy as np

T = TypeVar("T")


def cosine_similarity(vec_a: list[float] | None, vec_b: list[float] | None) -> float:
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=float)
    b = np.array(vec_b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def top_k_by_similarity(
    query_vector: list[float],
    items: Iterable[T],
    embedding_getter: Callable[[T], list[float] | None],
    top_k: int,
) -> list[tuple[T, float]]:
    scored = []
    for item in items:
        score = cosine_similarity(query_vector, embedding_getter(item))
        scored.append((item, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
