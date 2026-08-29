from __future__ import annotations


def normalize_output(text: object) -> str:
    if isinstance(text, bytes):
        return text.decode("utf-8", errors="replace")
    if text is None:
        return ""
    return str(text)


def truncate_output(text: object, max_chars: int) -> str:
    text = normalize_output(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
