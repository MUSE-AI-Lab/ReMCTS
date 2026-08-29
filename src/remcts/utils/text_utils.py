from __future__ import annotations

import re


def strip_code_fences(text: str) -> str:
    return re.sub(r"^```(?:python)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def truncate_text(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
