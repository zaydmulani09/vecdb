# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

## [0.1.0] — 2026-05-18

### Added
- HNSW dense vector index via instant-distance
- BM25 sparse inverted index with hand-written tokenizer
- Hybrid retrieval engine with RRF and weighted fusion strategies
- IVF (Inverted File Index) backend with pure-Rust k-means
- SQL-like query language with VECTOR_SIM predicate
- Column projection and filter pushdown in query planner
- HTTP API (Axum 0.7) with 13 endpoints
- Prometheus metrics exporter at /metrics
- X-Api-Key authentication
- Request rate limiting (1000 req/min per IP)
- Security headers middleware
- Request body size limit (32MB)
- Graceful shutdown with WAL checkpoint and index save
- Multi-collection support
- Write-ahead log (WAL) for crash recovery
- Memory-mapped vector store with madvise hints
- Connection pooling for SQLite metadata store (r2d2)
- SIMD-accelerated distance functions (auto-vectorized)
- CLI tool (vecdb-cli) with ping, collection, ingest, inspect, search commands
- Python SDK (vecdb-python) with sync and async clients
- TypeScript SDK (vecdb-client) with native fetch, ESM
- Docker multi-stage musl build + docker-compose
- GitHub Actions CI (build, test, clippy, fmt)
- Benchmark harness with synthetic workloads
- Jupyter demo notebook
- 171 Rust tests, 8 pytest, 8 vitest

### Known Limitations
- IVF not auto-selected; HNSW is always default unless explicitly configured
- CLI ingest has no streaming progress for large batches
- Real MS MARCO / BEIR benchmark runs require external dataset download
