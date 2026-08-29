#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m remcts.cli.evaluate --dataset humaneval --data-path "$1" --config src/remcts/config/humaneval.yaml
