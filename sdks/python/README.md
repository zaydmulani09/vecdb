# vecdb-client

Python client SDK for the [vecdb](https://github.com/vecdb/vecdb) vector database.

Pure Python, no native extensions. Uses `httpx` for both sync and async HTTP.

## Install

```bash
pip install vecdb-client
```

## Quick Start

```python
from vecdb import VecDbClient, VectorRecord

client = VecDbClient(base_url="http://localhost:8080", api_key="secret")

client.create_collection("docs", dimension=1536)

client.upsert("docs", [
    VectorRecord(id="1", vector=[0.1] * 1536, text="hello world"),
])

results = client.search_dense("docs", vector=[0.1] * 1536, k=5)
for r in results.results:
    print(r.id, r.score)
```

## Async

```python
from vecdb import AsyncVecDbClient

async with AsyncVecDbClient(base_url="http://localhost:8080") as client:
    results = await client.search_hybrid("docs", vector=[0.1]*1536, query="hello")
```

## Dev

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
