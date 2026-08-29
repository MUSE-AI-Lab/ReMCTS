from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a ReMCTS run directory.")
    parser.add_argument("run_dir")
    args = parser.parse_args()

    summary_path = Path(args.run_dir) / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary file: {summary_path}")
    print(json.dumps(json.loads(summary_path.read_text(encoding='utf-8')), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
