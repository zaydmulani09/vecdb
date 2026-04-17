# vecdb

[![CI](https://github.com/zaydmulani09/vecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/zaydmulani09/vecdb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Rust](https://img.shields.io/badge/language-Rust-orange.svg)](https://www.rust-lang.org/)

**Open source, production-grade vector database written in Rust.**

---

## What is vecdb?

vecdb is a self-hosted vector database that stores high-dimensional float32 vectors alongside JSON metadata and supports hybrid semantic search — combining dense nearest-neighbour retrieval (HNSW/IVF) with sparse BM25 keyword search. Results from both indexes are fused using a configurable alpha weight so you get the precision of keyword matching and the recall of embedding search in a single query.

vecdb differs from managed alternatives in one key way: it runs entirely on your own hardware with no external services, no API keys, no egress fees, and no cloud dependency. It ships as a single statically-linked binary (~10 MB). A fresh instance is running in under a second. The entire codebase is MIT licensed and written in pure Rust — memory safe, async I/O, and fast enough to serve thousands of queries per second on a laptop.

---

## Why vecdb?

Most vector databases are cloud services: you send your data to their servers, pay per query, and accept their privacy policies. vecdb is different — it runs on your own machine, never phones home, and costs nothing beyond the hardware you already own. It combines dense HNSW search (semantic similarity) with sparse BM25 search (exact keyword matching) in a single query, which consistently outperforms either approach alone on real retrieval tasks. If you want hybrid semantic search without a cloud dependency, vecdb is for you.

---

## Features

- **HNSW dense search** — approximate nearest-neighbour via instant-distance; cosine, euclidean, and dot-product metrics
- **BM25 sparse search** — custom inverted index with tokenizer, stopwords, and IDF scoring; no external search dependencies
- **Hybrid fusion** — weighted sum (`alpha * dense + (1-alpha) * sparse`) or reciprocal rank fusion; configurable per query
- **SQL-like query language** — `SELECT … FROM col WHERE VECTOR_SIM(vec, [0.1, 0.2]) > 0.8 AND payload->>'genre' = 'sci-fi' LIMIT 10`
- **IVF backend** — pure-Rust k-means IVF index for larger collections; automatic selection via cost-based query planner
- **Column projection** — `SELECT id, title, score` pushes projection all the way to the metadata layer
- **Filter pushdown** — all comparison operators (`=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE`) and nested JSON paths
- **Multi-collection** — create any number of named collections; fully concurrent access
- **Connection pooling** — r2d2 pool over SQLite; configurable pool size
- **Graceful shutdown** — SIGINT/SIGTERM flushes WAL and saves indexes before exit
- **Prometheus metrics** — `/metrics` endpoint; request latency histograms, QPS counters, error rates
- **Python SDK** — pure Python, `httpx`, sync + async, zero native extensions
- **TypeScript SDK** — native `fetch`, ESM, Node 18+, zero runtime dependencies
- **Docker** — single-command deploy; multi-stage musl build → alpine image
- **CLI** — `vecdb ping`, `vecdb collection create`, `vecdb ingest`, `vecdb search`

---

## Quickstart

### Docker

```bash
docker run -p 6333:6333 vecdb:latest
curl http://localhost:6333/health
```

Or with docker compose (persistent volume, restart policy):

```bash
docker compose up -d
```

### Build from Source

```bash
# Requires Rust stable (1.77+)
cargo build --release -p vecdb-api
./target/release/vecdb-api
# Server starts on http://127.0.0.1:6333 by default
```

### First Requests

```bash
# Create a 3-dimensional collection
curl -X POST http://localhost:6333/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimension":3,"metric":"cosine"}'

# Upsert vectors
curl -X POST http://localhost:6333/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"records":[
    {"id":"doc1","vector":[0.1,0.2,0.9],"text":"machine learning","payload":{"topic":"AI"}},
    {"id":"doc2","vector":[0.8,0.1,0.1],"text":"database systems","payload":{"topic":"DB"}}
  ]}'

# Dense vector search
curl -X POST http://localhost:6333/collections/docs/search/dense \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.9],"k":5}'

# Hybrid search (dense + sparse BM25)
curl -X POST http://localhost:6333/collections/docs/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.9],"query":"machine learning","k":5,"alpha":0.7}'

# SQL query with VECTOR_SIM and payload filter
curl -X POST http://localhost:6333/query \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM docs WHERE VECTOR_SIM(vec, [0.1,0.2,0.9]) > 0.5 LIMIT 5"}'
```

---

## Python SDK

```bash
pip install ./sdks/python
```

```python
from vecdb import VecDbClient, VectorRecord

client = VecDbClient(base_url="http://localhost:8080")

# Create collection
client.create_collection("docs", dimension=768)

# Upsert vectors
client.upsert("docs", [
    VectorRecord(id="doc1", vector=[...], text="machine learning basics"),
    VectorRecord(id="doc2", vector=[...], text="vector databases explained"),
])

# Hybrid search
results = client.search_hybrid(
    "docs",
    vector=[...],
    query="machine learning",
    k=10,
    alpha=0.7,
)

for r in results.results:
    print(r.id, r.score)

# Async client
from vecdb import AsyncVecDbClient
import asyncio

async def main():
    async with AsyncVecDbClient(base_url="http://localhost:8080") as client:
        results = await client.search_dense("docs", vector=[...], k=10)

asyncio.run(main())
```

---

## TypeScript SDK

```bash
npm install ./sdks/typescript
```

```typescript
import { VecDbClient } from "vecdb-client";

const client = new VecDbClient({ baseUrl: "http://localhost:8080" });

// Create collection
await client.createCollection({ name: "docs", dimension: 768 });

// Upsert vectors
await client.upsert("docs", [
  { id: "doc1", vector: [...], text: "machine learning basics" },
  { id: "doc2", vector: [...], text: "vector databases explained" },
]);

// Hybrid search
const results = await client.searchHybrid("docs", {
  vector: [...],
  query: "machine learning",
  k: 10,
  alpha: 0.7,
});

for (const r of results.results) {
  console.log(r.id, r.score);
}
```

---

## CLI

```bash
# Build
cargo build --release -p vecdb-cli

# Ping server
vecdb ping

# Collection management
vecdb collection create my-docs --dimension 768
vecdb collection list
vecdb collection get my-docs
vecdb collection delete my-docs

# Ingest from JSONL file (each line: {"id":"...","vector":[...],"text":"...","payload":{...}})
vecdb ingest --file corpus.jsonl --collection my-docs --batch-size 500

# Search
vecdb search dense  --collection my-docs --vector "[0.1,0.2,...]" --k 10
vecdb search sparse --collection my-docs --query "semantic search" --k 10
vecdb search hybrid --collection my-docs --query "semantic search" --k 10 --alpha 0.7
vecdb search sql    --collection my-docs --sql "SELECT id, score FROM my-docs LIMIT 5"
```

---

## SQL Query Language

vecdb understands a subset of SQL extended with the `VECTOR_SIM` function:

```sql
-- Basic vector similarity search
SELECT * FROM my_collection
WHERE VECTOR_SIM(vec, [0.1, 0.2, 0.9]) > 0.5
LIMIT 10

-- With payload filter
SELECT id, score FROM my_collection
WHERE VECTOR_SIM(vec, [0.1, 0.2, 0.9]) > 0.5
  AND payload->>'genre' = 'sci-fi'
LIMIT 10

-- Numeric comparison
SELECT id, title, score FROM my_collection
WHERE VECTOR_SIM(vec, [0.1, 0.2, 0.9]) > 0.7
  AND payload->>'year' > 2020
ORDER BY score DESC
LIMIT 5

-- LIKE filter
SELECT * FROM my_collection
WHERE VECTOR_SIM(vec, [0.1, 0.2, 0.9]) > 0.6
  AND payload->>'title' LIKE '%machine%'
LIMIT 20
```

Supported operators: `=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE` (with `%` wildcards).
JSON path navigation: `payload->>'field'`, `payload->'nested'->>'key'`.

---

## Architecture

```
Client → HTTP (Axum) → Auth Middleware → Route Handler
                                              ↓
                                        CollectionManager
                                              ↓
                                          Storage
                                    ┌────────┼────────┐
                                  MMAP      WAL    SQLite
                                    └────────┼────────┘
                                             ↓
                                    ┌────────┴────────┐
                                 HnswIndex         SparseIndex
                                    └────────┬────────┘
                                         HybridEngine
                                             ↓
                                        QueryPlanner
                                             ↓
                                         Results
```

See [docs/architecture.md](docs/architecture.md) for the full component breakdown.

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `VECDB__PORT` | `6333` | Listen port |
| `VECDB__HOST` | `127.0.0.1` | Bind address |
| `VECDB__DATA_DIR` | `./data` | Collection storage directory |
| `VECDB__API_KEY` | _(none)_ | If set, all requests require `X-Api-Key` header |
| `VECDB__LOG_LEVEL` | `info` | Tracing level (`trace`/`debug`/`info`/`warn`/`error`) |
| `VECDB__QUERY_TIMEOUT_MS` | `5000` | Per-request timeout in milliseconds |

> Note: env vars use double-underscore (`VECDB__PORT`) as separator — this is the `config` crate convention.

See [docs/configuration.md](docs/configuration.md) for the full reference including TOML config file.

---

## Benchmarks

| Metric | HNSW | IVF |
|---|---|---|
| Recall@10 (10k vectors, dim=128) | TBD | TBD |
| p50 latency | TBD | TBD |
| p95 latency | TBD | TBD |
| QPS | TBD | TBD |

See [benchmarks/README.md](benchmarks/README.md) to run your own benchmarks.

---

## License

MIT © 2026 vecdb contributors
