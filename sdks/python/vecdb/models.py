from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CollectionConfig:
    name: str
    dimension: int
    metric: str = "cosine"
    index_type: str = "hnsw"


@dataclass
class CollectionInfo:
    name: str
    dimension: int
    metric: str
    index_type: str
    vector_count: int
    created_at: str

    @classmethod
    def from_dict(cls, d: dict) -> "CollectionInfo":
        return cls(
            name=d["name"],
            dimension=d["dimension"],
            metric=d["metric"],
            index_type=d["index_type"],
            vector_count=d["vector_count"],
            created_at=d["created_at"],
        )


@dataclass
class VectorRecord:
    id: str
    vector: list[float]
    payload: dict | None = None
    text: str | None = None

    def to_dict(self) -> dict:
        d: dict = {"id": self.id, "vector": self.vector}
        if self.payload is not None:
            d["payload"] = self.payload
        if self.text is not None:
            d["text"] = self.text
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VectorRecord":
        return cls(
            id=d["id"],
            vector=d["vector"],
            payload=d.get("payload"),
            text=d.get("text"),
        )


@dataclass
class SearchResult:
    id: str
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    payload: dict | None = None
    text: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResult":
        payload = d.get("payload")
        if payload is not None and not isinstance(payload, dict):
            payload = None
        return cls(
            id=d["id"],
            score=d["score"],
            dense_score=d.get("dense_score"),
            sparse_score=d.get("sparse_score"),
            payload=payload,
            text=d.get("text"),
        )


@dataclass
class UpsertResponse:
    inserted: int
    updated: int
    errors: list = field(default_factory=list)
    time_ms: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "UpsertResponse":
        return cls(
            inserted=d["inserted"],
            updated=d["updated"],
            errors=d.get("errors", []),
            time_ms=float(d.get("time_ms", 0)),
        )


@dataclass
class SearchResponse:
    results: list[SearchResult]
    count: int
    time_ms: float

    @classmethod
    def from_dict(cls, d: dict) -> "SearchResponse":
        results = [SearchResult.from_dict(r) for r in d.get("results", [])]
        return cls(
            results=results,
            count=d.get("count", len(results)),
            time_ms=float(d.get("time_ms", 0)),
        )
