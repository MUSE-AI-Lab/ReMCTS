#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${REMCTS_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
WORKERS="${WORKERS:-4}"
CREDENTIALS_FILE="${CREDENTIALS_FILE:-${ROOT}/apikeys.txt}"
MEM_BUDGET_CNY="${MEM_BUDGET_CNY:-500}"
KNOWN_BUDGET_CNY="${KNOWN_BUDGET_CNY:-260}"

MEM_REPORT="${MEM_REPORT:-${ROOT}/outputs/reports/memory-ablation-${RUN_ID}}"
KNOWN_REPORT="${KNOWN_REPORT:-${ROOT}/outputs/reports/known-baseline-completions-${RUN_ID}}"

cd "$ROOT"

printf '[%s] RUN_ID=%s\n' "$(date -Is)" "$RUN_ID"
printf '[%s] memory report: %s\n' "$(date -Is)" "$MEM_REPORT"
printf '[%s] known baseline report: %s\n' "$(date -Is)" "$KNOWN_REPORT"

printf '[%s] starting memory ablation experiments\n' "$(date -Is)"
python3 -u scripts/run_memory_ablation_experiments.py \
  --models qwen3-8b qwen3-32b deepseek-v4-flash \
  --settings no_tests known \
  --ablations no_global_memory no_path_memory \
  --workers "$WORKERS" \
  --budget-cny "$MEM_BUDGET_CNY" \
  --credentials-file "$CREDENTIALS_FILE" \
  --report-dir "$MEM_REPORT"

printf '[%s] starting known baseline completions\n' "$(date -Is)"
python3 -u scripts/run_known_baseline_completions.py \
  --models qwen3-8b qwen3-32b deepseek-v4-flash \
  --methods reflection_style mcts_style \
  --datasets humaneval_full_known mbpp_sanitized_full_known \
  --workers "$WORKERS" \
  --budget-cny "$KNOWN_BUDGET_CNY" \
  --credentials-file "$CREDENTIALS_FILE" \
  --report-dir "$KNOWN_REPORT"

printf '[%s] all supplemental experiments finished\n' "$(date -Is)"
