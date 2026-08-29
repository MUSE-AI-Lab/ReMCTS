from __future__ import annotations

import ast

from remcts.core.state import SearchConfig
from remcts.evaluation.evidence import EvidenceResult, clamp_score


DANGEROUS_IMPORTS = {
    "os",
    "subprocess",
    "socket",
    "shutil",
    "ftplib",
    "telnetlib",
    "requests",
    "urllib",
}

DANGEROUS_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
}

DANGEROUS_ATTRS = {
    "system",
    "popen",
    "remove",
    "unlink",
    "rmdir",
    "rmtree",
    "chmod",
    "chown",
    "connect",
    "request",
    "urlopen",
}


def score_risk(code: str, config: SearchConfig) -> EvidenceResult:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return EvidenceResult(
            name="risk",
            score=0.25,
            confidence=0.70,
            available=True,
            passed=False,
            details={"error": "syntax error prevents full risk scan"},
        )

    findings: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DANGEROUS_IMPORTS:
                    findings.append(f"dangerous import: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in DANGEROUS_IMPORTS:
                findings.append(f"dangerous import: {node.module}")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in DANGEROUS_CALLS:
                findings.append(f"dangerous call: {func.id}")
            elif isinstance(func, ast.Attribute) and func.attr in DANGEROUS_ATTRS:
                findings.append(f"dangerous attribute call: {func.attr}")

    line_count = len(code.splitlines())
    complexity_penalty = 0.0
    if line_count > 200:
        complexity_penalty = 0.15
        findings.append(f"large code size: {line_count} lines")

    score = clamp_score(0.20 * len(findings) + complexity_penalty)
    return EvidenceResult(
        name="risk",
        score=score,
        confidence=0.80,
        available=True,
        passed=score <= config.production_risk_threshold,
        details={"findings": findings, "line_count": line_count},
    )
