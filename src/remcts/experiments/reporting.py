"""Per-experiment reports (``<label>.json``) and the report-directory summary (``summary.json/md``)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from remcts.core.state import ProblemSpec

from .specs import ExperimentSpec
from .usage import MODEL_PRICES_CNY_PER_MILLION, aggregate_usage, estimate_cost_cny


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rate(numerator: int, denominator: int, digits: int = 4) -> float:
    return round(numerator / denominator, digits) if denominator else 0.0


def hidden_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Hidden solve rate and the quality of the method's own acceptance decisions."""
    rows = [row for row in rows if "hidden_success" in row]
    if not rows:
        return None
    total = len(rows)
    accepted = sum(1 for row in rows if row.get("accepted"))
    needs_review = sum(1 for row in rows if row.get("needs_review"))
    solved = sum(1 for row in rows if row.get("hidden_success"))
    accepted_solved = sum(1 for row in rows if row.get("accepted") and row.get("hidden_success"))
    return {
        "tasks": total,
        "accepted": accepted,
        "needs_review": needs_review,
        "hidden_success": solved,
        "accepted_and_hidden_success": accepted_solved,
        "accepted_but_hidden_failed": accepted - accepted_solved,
        "accepted_rate": _rate(accepted, total),
        "needs_review_rate": _rate(needs_review, total),
        "hidden_solve_rate": _rate(solved, total),
        "acceptance_precision": _rate(accepted_solved, accepted),
        "false_acceptance_rate": _rate(accepted - accepted_solved, accepted),
    }


def build_output(
    spec: ExperimentSpec,
    tasks: list[ProblemSpec],
    rows: list[dict[str, Any]],
    *,
    wall_elapsed_sec: float,
    resumed_count: int,
    workers: int,
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row.get("task_index") or 0))
    n = len(rows)
    usage = aggregate_usage(rows)
    hidden = hidden_summary(rows)
    failed = sum(1 for row in rows if row.get("request_failed"))
    search_success = _rate(sum(1 for row in rows if row.get("success")), n, 6)
    output: dict[str, Any] = {
        "label": spec.label,
        "model_alias": spec.model_alias,
        "method": spec.method,
        "runner": spec.runner,
        "dataset": spec.dataset.name,
        "setting": spec.dataset.setting,
        "data_path": spec.dataset.data_file,
        "hidden_eval_path": spec.dataset.hidden_eval_file,
        "config": spec.config,
        "config_overrides": spec.config_overrides,
        "parameters": spec.params,
        "reuse_from_label": spec.reuse_from_label,
        "status": "complete" if n == len(tasks) and failed == 0 else "partial",
        "size": len(tasks),
        "completed": n,
        "pass_rate": hidden["hidden_solve_rate"] if hidden else search_success,
        "pass_rate_metric": "hidden_solve_rate" if hidden else "explicit_test_pass_rate",
        "raw_search_success_rate": search_success,
        "avg_iterations": _rate(sum(int(row.get("iterations") or 0) for row in rows), n, 6),
        "avg_tree_size": _rate(sum(int(row.get("tree_size") or 0) for row in rows), n, 6),
        "request_failed_tasks": failed,
        "wall_elapsed_sec": round(wall_elapsed_sec, 6),
        "summed_task_wall_elapsed_sec": round(sum(float(row.get("wall_elapsed_sec") or 0.0) for row in rows), 6),
        "resumed_task_count": resumed_count,
        "workers": workers,
        "llm_usage": usage,
        "estimated_cost_cny": estimate_cost_cny(
            spec.model_alias, usage["totals"]["prompt_tokens"], usage["totals"]["completion_tokens"]
        ),
        "hidden_summary": hidden,
    }
    if any("oracle_pool_hidden_success" in row for row in rows):
        output["oracle_pool_hidden_success_rate"] = _rate(
            sum(1 for row in rows if row.get("oracle_pool_hidden_success")), n, 6
        )
    visible = [row for row in rows if "visible_success" in row]
    if visible:
        output["visible_pass_rate"] = _rate(sum(1 for row in visible if row["visible_success"]), len(visible), 6)
    output["tasks"] = rows
    return output


def load_outputs(report_dir: Path) -> list[dict[str, Any]]:
    """Complete experiment reports in ``report_dir``."""
    outputs = []
    for path in sorted(report_dir.glob("*.json")):
        if path.name == "summary.json" or path.name.endswith(".progress.json"):
            continue
        data = read_json(path)
        if isinstance(data, dict) and data.get("label") and data.get("status") == "complete":
            outputs.append(data)
    return outputs


def write_summary(report_dir: Path, outputs: list[dict[str, Any]], *, started_at: str, budget: dict[str, Any]) -> dict:
    rows = [_summary_row(report_dir, output) for output in outputs]
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("calls", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_cny")
    }
    totals["estimated_cost_cny"] = round(totals["estimated_cost_cny"], 6)
    summary = {
        "report_dir": str(report_dir),
        "started_at_utc": started_at,
        "finished_at_utc": utc_stamp(),
        "price_assumption_cny_per_million": {
            model: {"input": prices[0], "output": prices[1]} for model, prices in MODEL_PRICES_CNY_PER_MILLION.items()
        },
        "budget": {**budget, "report_estimated_cost_cny": totals["estimated_cost_cny"]},
        "rows": rows,
        "totals": totals,
    }
    write_json(report_dir / "summary.json", summary)
    (report_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def _summary_row(report_dir: Path, output: dict[str, Any]) -> dict[str, Any]:
    usage = output["llm_usage"]["totals"]
    return {
        "label": output["label"],
        "file": str(report_dir / f"{output['label']}.json"),
        **{key: output.get(key) for key in ("model_alias", "method", "dataset", "setting", "size", "completed")},
        "pass_rate": output["pass_rate"],
        "pass_rate_metric": output["pass_rate_metric"],
        "visible_pass_rate": output.get("visible_pass_rate"),
        "oracle_pool_hidden_success_rate": output.get("oracle_pool_hidden_success_rate"),
        "avg_iterations": output["avg_iterations"],
        "avg_tree_size": output["avg_tree_size"],
        "wall_elapsed_sec": output["wall_elapsed_sec"],
        "calls": usage["calls"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "estimated_cost_cny": output["estimated_cost_cny"],
        "hidden_summary": output.get("hidden_summary"),
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.2f}"


def render_summary_md(summary: dict[str, Any]) -> str:
    rows = sorted(summary["rows"], key=lambda row: (row["model_alias"], row["dataset"], row["method"]))
    show_visible = any(row["visible_pass_rate"] is not None for row in rows)
    show_oracle = any(row["oracle_pool_hidden_success_rate"] is not None for row in rows)
    header = ["Model", "Method", "Dataset", "Tasks", "Hidden solve (%)"]
    header += ["Visible pass (%)"] * show_visible + ["Oracle pool (%)"] * show_oracle
    header += ["Accepted (%)", "Acc. prec. (%)", "False acc. (%)", "Avg iter", "Avg tree"]
    header += ["Calls", "Tokens", "Cost (CNY)"]
    lines = [
        "# ReMCTS experiment summary",
        "",
        f"- Report dir: `{summary['report_dir']}`",
        "- Hidden solve rate is computed with the hidden-evaluation split only; acceptance columns describe "
        "each method's own stopping decision.",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * 3 + ["---:"] * (len(header) - 3)) + " |",
    ]
    for row in rows:
        hidden = row["hidden_summary"] or {}
        cells = [row["model_alias"], row["method"], row["dataset"], f"{row['completed']}/{row['size']}"]
        cells.append(_percent(row["pass_rate"]))
        if show_visible:
            cells.append(_percent(row["visible_pass_rate"]))
        if show_oracle:
            cells.append(_percent(row["oracle_pool_hidden_success_rate"]))
        cells += [
            _percent(hidden.get("accepted_rate")),
            _percent(hidden.get("acceptance_precision")),
            _percent(hidden.get("false_acceptance_rate")),
            f"{row['avg_iterations']:.3f}",
            f"{row['avg_tree_size']:.3f}",
            str(row["calls"]),
            str(row["total_tokens"]),
            f"{row['estimated_cost_cny']:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lift = _direct_vs_remcts(rows)
    if lift:
        lines += ["", "## Direct vs. ReMCTS", "", "| Model | Dataset | Direct | ReMCTS | Delta (pp) |"]
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for model, dataset, direct, remcts in lift:
            delta = 100 * (remcts - direct)
            lines.append(f"| {model} | {dataset} | {_percent(direct)} | {_percent(remcts)} | {delta:+.2f} |")

    totals = summary["totals"]
    budget = summary["budget"]
    lines += [
        "",
        "## Totals",
        "",
        f"- calls={totals['calls']}, input={totals['prompt_tokens']}, output={totals['completion_tokens']}, "
        f"total={totals['total_tokens']}, cost=CNY {totals['estimated_cost_cny']:.4f}",
        f"- budget=CNY {budget.get('budget_cny', 0):.2f}, initial_spent=CNY {budget.get('initial_spent_cny', 0):.2f}, "
        f"stop_margin=CNY {budget.get('stop_margin_cny', 0):.2f}",
        "",
    ]
    return "\n".join(lines)


def _direct_vs_remcts(rows: list[dict[str, Any]]) -> list[tuple[str, str, float, float]]:
    rates: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        rates.setdefault((row["model_alias"], row["dataset"]), {})[row["method"]] = float(row["pass_rate"])
    return [
        (model, dataset, values["direct"], values["remcts"])
        for (model, dataset), values in sorted(rates.items())
        if "direct" in values and "remcts" in values
    ]


def status_lines(report_dir: Path) -> list[str]:
    """``completed/size status label`` for every experiment in a report directory."""
    entries: dict[str, tuple[int, int, str]] = {}
    for path in sorted(report_dir.glob("*.progress.json")):
        data = read_json(path)
        if isinstance(data, dict) and data.get("label"):
            entries[data["label"]] = (int(data.get("completed") or 0), int(data.get("size") or 0), "running")
    for output in (read_json(path) for path in sorted(report_dir.glob("*.json"))):
        if isinstance(output, dict) and output.get("label") and "status" in output and "tasks" in output:
            done, size = int(output.get("completed") or 0), int(output.get("size") or 0)
            entries[output["label"]] = (done, size, output["status"])
    return [f"{done:>4}/{size:<4} {status:<8} {label}" for label, (done, size, status) in sorted(entries.items())]
