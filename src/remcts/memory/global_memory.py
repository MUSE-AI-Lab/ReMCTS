from __future__ import annotations

from remcts.core.state import ReflectionRecord

from .embedder import SimpleTextEmbedder
from .retriever import cosine_similarity


class GlobalMemoryStore:
    def __init__(self, embedder: SimpleTextEmbedder | None = None):
        self.embedder = embedder or SimpleTextEmbedder()
        self.records: list[ReflectionRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, reflection: ReflectionRecord) -> None:
        if reflection.embedding is None:
            text = " ".join([reflection.summary, reflection.root_cause, reflection.fix_hint]).strip()
            reflection.embedding = self.embedder.embed(text)
        self.records.append(reflection)

    def retrieve(
        self,
        query_text: str,
        top_k: int,
        min_severity: float | None = None,
    ) -> list[tuple[ReflectionRecord, float]]:
        if not self.records:
            return []
        query_vector = self.embedder.embed(query_text)
        scored: list[tuple[ReflectionRecord, float]] = []
        for record in self.records:
            if min_severity is not None and record.severity < min_severity:
                continue
            scored.append((record, cosine_similarity(query_vector, record.embedding)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]
