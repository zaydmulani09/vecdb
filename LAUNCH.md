# Launch Posts

## Show HN

**Title:** Show HN: vecdb – local-only hybrid vector database in Rust (MIT)

**Body:**
vecdb is a self-hosted vector database that combines HNSW dense search with BM25 sparse search and fuses both scores in a single query — no cloud, no API keys, no egress fees, just a single statically-linked binary you run on your own hardware. It has an HTTP API, a SQL-like query language with a VECTOR_SIM predicate, Python and TypeScript SDKs, and a Docker image. The entire codebase is pure Rust and MIT licensed.

Repo: https://github.com/zaydmulani09/vecdb

---

## r/rust

**Title:** I built a hybrid vector database in pure Rust — HNSW + BM25 + RRF fusion, SQL query language, HTTP API

**Body:**
After seeing a lot of vector DBs either rely on Python internals or phone home to a cloud service, I built vecdb: a self-hosted vector database written entirely in Rust.

Technical highlights:
- **HNSW dense index** via instant-distance 0.6 (cosine, euclidean, dot-product)
- **BM25 sparse inverted index** — hand-written tokenizer, no external search deps
- **Hybrid fusion** — weighted sum (`alpha * dense + (1-alpha) * sparse`) or reciprocal rank fusion (RRF)
- **IVF backend** — pure-Rust k-means as an alternative index for large collections
- **SQL-like query language** — hand-written recursive-descent lexer + parser; `VECTOR_SIM(vec, [...]) > 0.8 AND payload->>'genre' = 'sci-fi'`
- **Axum 0.7 HTTP API** — 13 endpoints, Prometheus metrics, X-Api-Key auth, rate limiting, graceful shutdown
- **Memory-mapped vector store** — 64-byte header, grow-by-doubling, madvise hints on Linux
- **WAL** — serde_json framing with xxh3 checksum for crash recovery
- 171 Rust tests, Python + TypeScript SDKs, Docker multi-stage musl build

MIT licensed. Single binary, no cloud dependency.

Repo: https://github.com/zaydmulani09/vecdb

---

## r/selfhosted

**Title:** vecdb – run a vector database on your own machine, no cloud, no API keys, single binary

**Body:**
If you've wanted semantic/hybrid search for your own documents without sending data to OpenAI, Pinecone, or Weaviate, vecdb might be what you're looking for.

It runs entirely on your hardware:
- Single statically-linked binary (~10 MB), starts in under a second
- Docker image available (`docker compose up -d`)
- Persistent storage: memory-mapped vectors + SQLite metadata + write-ahead log
- Hybrid search: combines dense vector similarity with BM25 keyword search in one query
- HTTP API with optional API key auth and rate limiting
- Python and TypeScript SDKs included

No account required, no telemetry, no egress fees. MIT licensed.

Repo: https://github.com/zaydmulani09/vecdb
