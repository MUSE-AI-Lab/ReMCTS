from __future__ import annotations

import json
import re


CODE_BLOCK_PATTERN = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    match = CODE_BLOCK_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    if "def " in stripped:
        return stripped[stripped.find("def ") :].strip()
    return stripped


def parse_json_block(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise
