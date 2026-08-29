from __future__ import annotations

import re


ERROR_TYPES = [
    "SyntaxError",
    "TypeError",
    "IndexError",
    "KeyError",
    "ValueError",
    "AssertionError",
    "TimeoutError",
    "NameError",
]


def parse_traceback(traceback_text: str) -> dict[str, str]:
    if not traceback_text:
        return {"error_type": "", "message": "", "stack_summary": ""}
    error_type = "ExecutionError"
    for candidate in ERROR_TYPES:
        if candidate in traceback_text:
            error_type = candidate
            break
    lines = [line.strip() for line in traceback_text.strip().splitlines() if line.strip()]
    message = lines[-1] if lines else traceback_text.strip()
    line_match = re.search(r"line (\d+)", traceback_text)
    return {
        "error_type": error_type,
        "error_line": line_match.group(1) if line_match else "",
        "message": message,
        "stack_summary": " | ".join(lines[-3:]),
    }
