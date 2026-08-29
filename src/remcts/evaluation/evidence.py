from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class EvidenceResult:
    name: str
    score: float
    confidence: float
    available: bool
    passed: bool | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def weighted_value(self, weight: float) -> float:
        if not self.available:
            return 0.0
        return weight * self.confidence * self.score

    def weighted_mass(self, weight: float) -> float:
        if not self.available:
            return 0.0
        return weight * self.confidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))
