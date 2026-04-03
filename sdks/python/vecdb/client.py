from __future__ import annotations

from typing import Any

import httpx

from .exceptions import ConnectionError, raise_for_status
from .models import (
    CollectionInfo,
    SearchResponse,
    UpsertResponse,
    VectorRecord,
)


class VecDbClient:
    """Synchronous HTTP client for vecdb."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._client = httpx.Client(headers=headers, timeout=timeout)

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, self._url(path), **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConnectionError(str(exc)) from exc
        raise_for_status(response)
        return response

    # ── Collections ──────────────────────────────────────────────────────────

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
    ) -> CollectionInfo:
        """Create a new collection."""
        body = {"name": name, "dimension": dimension, "metric": metric, "index_type": index_type}
        r = self._request("POST", "/collections", json=body)
        return CollectionInfo.from_dict(r.json())

    def list_collections(self) -> list[CollectionInfo]:
        """List all collections."""
        r = self._request("GET", "/collections")
        return [CollectionInfo.from_dict(c) for c in r.json()]

    def get_collection(self, name: str) -> CollectionInfo:
        """Get a single collection by name."""
        r = self._request("GET", f"/collections/{name}")
        return CollectionInfo.from_dict(r.json())

    def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True on success."""
        self._request("DELETE", f"/collections/{name}")
        return True

    # ── Vectors ───────────────────────────────────────────────────────────────

    def upsert(self, collection: str, records: list[VectorRecord]) -> UpsertResponse:
        """Upsert vectors into a collection."""
        body = {"records": [r.to_dict() for r in records]}
        r = self._request("POST", f"/collections/{collection}/vectors", json=body)
        return UpsertResponse.from_dict(r.json())

    def get_vector(self, collection: str, id: str) -> VectorRecord:
        """Get a single vector by ID."""
        r = self._request("GET", f"/collections/{collection}/vectors/{id}")
        return VectorRecord.from_dict(r.json())

    def delete_vectors(self, collection: str, ids: list[str]) -> dict:
        """Delete vectors by IDs."""
        body = {"ids": ids}
        r = self._request("DELETE", f"/collections/{collection}/vectors", json=body)
        return r.json()

    # ── Search ────────────────────────────────────────────────────────────────

    def search_dense(self, collection: str, vector: list[float], k: int = 10) -> SearchResponse:
        """Dense (HNSW) vector search."""
        body = {"vector": vector, "k": k}
        r = self._request("POST", f"/collections/{collection}/search/dense", json=body)
        return SearchResponse.from_dict(r.json())

    def search_sparse(self, collection: str, query: str, k: int = 10) -> SearchResponse:
        """Sparse (BM25) text search."""
        body = {"query": query, "k": k}
        r = self._request("POST", f"/collections/{collection}/search/sparse", json=body)
        return SearchResponse.from_dict(r.json())

    def search_hybrid(
        self,
        collection: str,
        vector: list[float] | None = None,
        query: str | None = None,
        k: int = 10,
        alpha: float = 0.7,
    ) -> SearchResponse:
        """Hybrid (dense + sparse) search."""
        body: dict[str, Any] = {"k": k, "alpha": alpha}
        if vector is not None:
            body["vector"] = vector
        if query is not None:
            body["query"] = query
        r = self._request("POST", f"/collections/{collection}/search/hybrid", json=body)
        return SearchResponse.from_dict(r.json())

    def query_sql(self, sql: str) -> SearchResponse:
        """Execute a SQL query."""
        body = {"sql": sql}
        r = self._request("POST", "/query", json=body)
        return SearchResponse.from_dict(r.json())

    def health(self) -> dict:
        """Get server health status."""
        r = self._request("GET", "/health")
        return r.json()

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "VecDbClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()


class AsyncVecDbClient:
    """Asynchronous HTTP client for vecdb."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        headers: dict[str, str] = {}
        if api_key:
            headers["X-Api-Key"] = api_key
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout)

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, self._url(path), **kwargs)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ConnectionError(str(exc)) from exc
        raise_for_status(response)
        return response

    # ── Collections ──────────────────────────────────────────────────────────

    async def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
    ) -> CollectionInfo:
        """Create a new collection."""
        body = {"name": name, "dimension": dimension, "metric": metric, "index_type": index_type}
        r = await self._request("POST", "/collections", json=body)
        return CollectionInfo.from_dict(r.json())

    async def list_collections(self) -> list[CollectionInfo]:
        """List all collections."""
        r = await self._request("GET", "/collections")
        return [CollectionInfo.from_dict(c) for c in r.json()]

    async def get_collection(self, name: str) -> CollectionInfo:
        """Get a single collection by name."""
        r = await self._request("GET", f"/collections/{name}")
        return CollectionInfo.from_dict(r.json())

    async def delete_collection(self, name: str) -> bool:
        """Delete a collection. Returns True on success."""
        await self._request("DELETE", f"/collections/{name}")
        return True

    # ── Vectors ───────────────────────────────────────────────────────────────

    async def upsert(self, collection: str, records: list[VectorRecord]) -> UpsertResponse:
        """Upsert vectors into a collection."""
        body = {"records": [r.to_dict() for r in records]}
        r = await self._request("POST", f"/collections/{collection}/vectors", json=body)
        return UpsertResponse.from_dict(r.json())

    async def get_vector(self, collection: str, id: str) -> VectorRecord:
        """Get a single vector by ID."""
        r = await self._request("GET", f"/collections/{collection}/vectors/{id}")
        return VectorRecord.from_dict(r.json())

    async def delete_vectors(self, collection: str, ids: list[str]) -> dict:
        """Delete vectors by IDs."""
        body = {"ids": ids}
        r = await self._request("DELETE", f"/collections/{collection}/vectors", json=body)
        return r.json()

    # ── Search ────────────────────────────────────────────────────────────────

    async def search_dense(self, collection: str, vector: list[float], k: int = 10) -> SearchResponse:
        """Dense (HNSW) vector search."""
        body = {"vector": vector, "k": k}
        r = await self._request("POST", f"/collections/{collection}/search/dense", json=body)
        return SearchResponse.from_dict(r.json())

    async def search_sparse(self, collection: str, query: str, k: int = 10) -> SearchResponse:
        """Sparse (BM25) text search."""
        body = {"query": query, "k": k}
        r = await self._request("POST", f"/collections/{collection}/search/sparse", json=body)
        return SearchResponse.from_dict(r.json())

    async def search_hybrid(
        self,
        collection: str,
        vector: list[float] | None = None,
        query: str | None = None,
        k: int = 10,
        alpha: float = 0.7,
    ) -> SearchResponse:
        """Hybrid (dense + sparse) search."""
        body: dict[str, Any] = {"k": k, "alpha": alpha}
        if vector is not None:
            body["vector"] = vector
        if query is not None:
            body["query"] = query
        r = await self._request("POST", f"/collections/{collection}/search/hybrid", json=body)
        return SearchResponse.from_dict(r.json())

    async def query_sql(self, sql: str) -> SearchResponse:
        """Execute a SQL query."""
        body = {"sql": sql}
        r = await self._request("POST", "/query", json=body)
        return SearchResponse.from_dict(r.json())

    async def health(self) -> dict:
        """Get server health status."""
        r = await self._request("GET", "/health")
        return r.json()

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "AsyncVecDbClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()

    async def close(self) -> None:
        await self._client.aclose()
