"""Token usage and cost accounting for experiment reports."""

from __future__ import annotations

from typing import Any

# (input, output) CNY per million tokens; the estimate is operational accounting only.
MODEL_PRICES_CNY_PER_MILLION: dict[str, tuple[float, float]] = {
    "deepseek-v4-flash": (0.952, 1.904),
    "qwen3-1.7b": (0.3, 1.2),
    "qwen3-8b": (0.5, 2.0),
    "qwen3-14b": (1.0, 4.0),
    "qwen3-32b": (2.0, 8.0),
}
_FALLBACK_PRICE_MODEL = "qwen3-32b"
COUNTERS = ("calls", "successful_calls", "failed_calls", "prompt_tokens", "completion_tokens", "total_tokens")


def compact_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Client usage snapshot without the per-call records."""
    usage = usage or {}
    compact: dict[str, Any] = {"model": usage.get("model"), "api_alias": usage.get("api_alias")}
    compact.update({key: int(usage.get(key) or 0) for key in COUNTERS})
    compact["elapsed_sec"] = round(float(usage.get("elapsed_sec") or 0.0), 6)
    return compact


def merge_usages(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum the usage of several attempts of one task (e.g. after switching API keys)."""
    merged = compact_usage(None)
    for usage in map(compact_usage, usages):
        merged["model"] = usage["model"] or merged["model"]
        merged["api_alias"] = usage["api_alias"] or merged["api_alias"]
        for key in COUNTERS:
            merged[key] += usage[key]
        merged["elapsed_sec"] = round(merged["elapsed_sec"] + usage["elapsed_sec"], 6)
    return merged


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over task rows, overall and per model."""
    empty = {**{key: 0 for key in COUNTERS}, "elapsed_sec": 0.0}
    totals = dict(empty)
    by_model: dict[str, dict[str, Any]] = {}
    for row in rows:
        usage = row.get("llm_usage") or {}
        model_totals = by_model.setdefault(str(usage.get("model") or row.get("model_alias") or "unknown"), dict(empty))
        for target in (totals, model_totals):
            for key in COUNTERS:
                target[key] += int(usage.get(key) or 0)
            target["elapsed_sec"] += float(usage.get("elapsed_sec") or 0.0)
    for target in (totals, *by_model.values()):
        target["elapsed_sec"] = round(target["elapsed_sec"], 6)
    return {"totals": totals, "by_model": by_model}


def estimate_cost_cny(model_alias: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_price, output_price = MODEL_PRICES_CNY_PER_MILLION.get(
        model_alias, MODEL_PRICES_CNY_PER_MILLION[_FALLBACK_PRICE_MODEL]
    )
    return round(prompt_tokens * input_price / 1_000_000 + completion_tokens * output_price / 1_000_000, 6)


def rows_cost_cny(model_alias: str, rows: list[dict[str, Any]]) -> float:
    totals = aggregate_usage(rows)["totals"]
    return estimate_cost_cny(model_alias, totals["prompt_tokens"], totals["completion_tokens"])
