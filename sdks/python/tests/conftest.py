import pytest
import respx


@pytest.fixture
def base_url() -> str:
    return "http://test.vecdb"


@pytest.fixture
def sample_collection() -> dict:
    return {
        "name": "test",
        "dimension": 3,
        "metric": "cosine",
        "index_type": "hnsw",
        "vector_count": 0,
        "created_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_vector_record() -> dict:
    return {
        "id": "vec-1",
        "vector": [0.1, 0.2, 0.3],
        "payload": {"title": "hello"},
        "text": "hello world",
    }


@pytest.fixture
def sample_search_response() -> dict:
    return {
        "results": [
            {
                "id": "vec-1",
                "score": 0.99,
                "dense_score": 0.99,
                "sparse_score": None,
                "payload": {"title": "hello"},
                "text": "hello world",
            },
            {
                "id": "vec-2",
                "score": 0.85,
                "dense_score": 0.85,
                "sparse_score": None,
                "payload": {"title": "world"},
                "text": None,
            },
            {
                "id": "vec-3",
                "score": 0.70,
                "dense_score": 0.70,
                "sparse_score": None,
                "payload": None,
                "text": None,
            },
        ],
        "count": 3,
        "time_ms": 5,
    }


@pytest.fixture
def mock_router():
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def async_mock_router():
    with respx.mock(assert_all_called=False) as router:
        yield router
