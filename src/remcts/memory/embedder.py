from __future__ import annotations

import hashlib
import re

import numpy as np


class SimpleTextEmbedder:
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = np.zeros(self.dim, dtype=float)
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[idx] += 1.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.tolist()
