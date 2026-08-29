#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m remcts.cli.evaluate --dataset mbpp --data-path "$1" --config src/remcts/config/mbpp.yaml
