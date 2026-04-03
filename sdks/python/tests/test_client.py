import httpx
import pytest
import respx

from vecdb import (
    AsyncVecDbClient,
    NotFoundError,
    UnauthorizedError,
    VecDbClient,
)
from vecdb.models import CollectionInfo, SearchResponse, SearchResult, UpsertResponse, VectorRecord


def test_create_collection_sync(base_url, sample_collection):
    with respx.mock() as router:
        router.post(f"{base_url}/collections").mock(
            return_value=httpx.Response(201, json=sample_collection)
        )
        client = VecDbClient(base_url=base_url)
        result = client.create_collection("test", dimension=3)
        assert isinstance(result, CollectionInfo)
        assert result.name == "test"
        assert result.dimension == 3


def test_list_collections_sync(base_url, sample_collection):
    second = {**sample_collection, "name": "test2"}
    with respx.mock() as router:
        router.get(f"{base_url}/collections").mock(
            return_value=httpx.Response(200, json=[sample_collection, second])
        )
        client = VecDbClient(base_url=base_url)
        results = client.list_collections()
        assert len(results) == 2
        assert all(isinstance(c, CollectionInfo) for c in results)


def test_upsert_vectors_sync(base_url):
    upsert_resp = {"inserted": 1, "updated": 0, "errors": [], "time_ms": 3}
    with respx.mock() as router:
        router.post(f"{base_url}/collections/test/vectors").mock(
            return_value=httpx.Response(200, json=upsert_resp)
        )
        client = VecDbClient(base_url=base_url)
        record = VectorRecord(id="vec-1", vector=[0.1, 0.2, 0.3], payload={"k": "v"})
        result = client.upsert("test", [record])
        assert isinstance(result, UpsertResponse)
        assert result.inserted == 1


def test_search_dense_sync(base_url, sample_search_response):
    with respx.mock() as router:
        router.post(f"{base_url}/collections/test/search/dense").mock(
            return_value=httpx.Response(200, json=sample_search_response)
        )
        client = VecDbClient(base_url=base_url)
        result = client.search_dense("test", vector=[0.1, 0.2, 0.3], k=3)
        assert isinstance(result, SearchResponse)
        assert result.count == 3
        assert isinstance(result.results[0], SearchResult)


async def test_search_hybrid_async(base_url, sample_search_response):
    with respx.mock() as router:
        router.post(f"{base_url}/collections/test/search/hybrid").mock(
            return_value=httpx.Response(200, json=sample_search_response)
        )
        client = AsyncVecDbClient(base_url=base_url)
        result = await client.search_hybrid("test", vector=[0.1, 0.2, 0.3], query="hello", k=3)
        assert isinstance(result, SearchResponse)
        assert result.count == 3
        await client.close()


def test_not_found_raises_NotFoundError(base_url):
    error_body = {"error": {"code": "not_found", "message": "collection 'ghost' not found"}}
    with respx.mock() as router:
        router.get(f"{base_url}/collections/ghost").mock(
            return_value=httpx.Response(404, json=error_body)
        )
        client = VecDbClient(base_url=base_url)
        with pytest.raises(NotFoundError) as exc_info:
            client.get_collection("ghost")
        assert exc_info.value.status_code == 404
        assert exc_info.value.error_code == "not_found"


def test_unauthorized_raises_UnauthorizedError(base_url):
    error_body = {"error": {"code": "unauthorized", "message": "invalid api key"}}
    with respx.mock() as router:
        router.get(f"{base_url}/collections").mock(
            return_value=httpx.Response(401, json=error_body)
        )
        client = VecDbClient(base_url=base_url)
        with pytest.raises(UnauthorizedError) as exc_info:
            client.list_collections()
        assert exc_info.value.status_code == 401


def test_context_manager_sync(base_url):
    health_resp = {"status": "ok", "version": "0.1.0", "vector_count": 0, "collections": 0}
    with respx.mock() as router:
        router.get(f"{base_url}/health").mock(
            return_value=httpx.Response(200, json=health_resp)
        )
        with VecDbClient(base_url=base_url) as client:
            result = client.health()
            assert result["status"] == "ok"
