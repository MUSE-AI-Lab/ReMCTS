from __future__ import annotations

from remcts.core.state import SearchResult


def pass_rate(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for result in results if result.success) / len(results)


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def average_iterations(results: list[SearchResult]) -> float:
    if not results:
        return 0.0
    return sum(result.iterations for result in results) / len(results)
