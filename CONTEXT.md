# vecdb — Build Context

**Single source of truth for continuing this build in any fresh Claude Code session.**
Last updated: after Prompt 25/25 completion — PROJECT COMPLETE. Prompt 26 complete — ready to ship. 171 Rust tests + 8 Python tests + 8 TypeScript tests passing.

---

## 1. Project Overview

**vecdb** is a production-grade, open-source vector database written entirely in Rust. It stores high-dimensional float32 vectors alongside JSON metadata and supports semantic search using both dense (HNSW) and sparse (BM25) indexes, with a hybrid fusion engine that combines both scores via configurable alpha weighting and a hand-written SQL query planner.

**Goal:** A self-hosted alternative to Pinecone/Weaviate/Qdrant — zero cloud dependency, MIT licensed, pure Rust, runs entirely on your own hardware, no API keys, no egress fees. Suitable for privacy-sensitive workloads, air-gapped environments, and cost-conscious teams.

**Tech stack:**
- Language: Rust edition 2021, Cargo workspace with 4 crates
- Toolchain: `stable-x86_64-pc-windows-gnu` via MSYS2 on Windows (CRITICAL — see Section 3)
- All cargo commands: `PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu <cmd>`
- HTTP server: axum 0.7 + tower-http 0.5 (CORS, trace, timeout middleware)
- Dense index: instant-distance 0.6 (HNSW, M hardcoded at 32 by the library)
- Sparse index: custom BM25 inverted index, no external search dependencies
- Storage: memmap2 0.9 (flat f32 binary file), rusqlite 0.32 bundled (SQLite metadata + payload), serde_json WAL; r2d2 0.8 + r2d2_sqlite 0.25 pool (max 8 connections per MetadataStore)
- Serialization: serde_json everywhere, NEVER bincode (see Section 3)
- Parallelism: rayon 1.9
- Config: `config` crate 0.14 + env vars with `VECDB__` prefix separator
- SQL: hand-written recursive-descent lexer + parser (NOT sqlparser-rs — see Section 3)
- Metrics: `metrics` 0.22 facade + `metrics-exporter-prometheus` 0.13 for Prometheus scraping
- Auth: X-Api-Key header middleware (global, optional — None disables auth)

**Crate layout:**
```
vecdb/
  crates/vecdb-core/   — all business logic: storage, index, sparse, hybrid, planner, types, SQL
  crates/vecdb-api/    — axum HTTP server (fully implemented); both [lib] and [[bin]] targets
  crates/vecdb-cli/    — pure HTTP client binary (vecdb); ping, collection, ingest, inspect, search
  crates/vecdb-bench/  — benchmark harness binary (stubs)
```

---

## 2. Exact Prompt Sequence and Status

| # | Title | Status |
|---|-------|--------|
| 1 | Project scaffold, workspace, types, CI | ✅ COMPLETE |
| 2 | MMAP vector store, WAL, SQLite metadata, Storage struct | ✅ COMPLETE |
| 3 | HNSW dense index (IndexBackend trait, HnswIndex, distance functions) | ✅ COMPLETE |
| 4 | BM25 sparse index (tokenizer, inverted index, SparseIndex facade) | ✅ COMPLETE |
| 5 | Hybrid fusion engine (HybridEngine, score normalization, alpha weighting, storage integration) | ✅ COMPLETE |
| 6 | Query planner — LogicalNode tree, PhysicalPlan, QueryPlanner cost model, PlanExecutor | ✅ COMPLETE |
| 7 | SQL parser — hand-written lexer + recursive-descent parser, VECTOR_SIM extension, AstConverter | ✅ COMPLETE |
| 8 | HTTP API layer — Axum route handlers, AppState wiring, auth middleware, error formatting | ✅ COMPLETE |
| 9 | Prometheus metrics + observability — RequestTimer, metrics.rs constants, PrometheusHandle, /metrics endpoint | ✅ COMPLETE |
| 10 | CLI Implementation — pure HTTP client (vecdb-cli), 5 subcommands, 8 wiremock tests | ✅ COMPLETE |
| 11 | Benchmark Harness — vecdb-bench, synthetic+JSONL datasets, recall@k, p50/p95/p99, QPS | ✅ COMPLETE |
| 12 | IVF backend — pure Rust IVF as alternative IndexBackend, AnyIndex enum dispatch | ✅ COMPLETE |
| 13 | Multi-collection support — collection registry, concurrent open, per-collection state | ✅ COMPLETE |
| 14 | Column projection + full filter pushdown — `SELECT id, title`, all operators, `MetadataStore::filter_ids` | ✅ COMPLETE |
| 15 | Connection pooling + graceful shutdown — r2d2 pool for MetadataStore, SIGINT/SIGTERM flush | ✅ COMPLETE |
| 16 | Python client SDK — pure httpx HTTP client (no PyO3) | ✅ COMPLETE |
| 17 | TypeScript Client SDK — native fetch, Node 18+, msw v2 mocks | ✅ COMPLETE |
| 18 | Docker + Deployment — Dockerfile (musl/alpine), docker-compose, health check scripts | ✅ COMPLETE |
| 18.5 | Benchmark Documentation + Download Scripts — replaces P19 + P20 | ✅ COMPLETE |
| 19 | MS MARCO Benchmark Run | ↩ REPLACED BY P18.5 |
| 20 | Vector quantization — scalar quantization (SQ8), product quantization (PQ) | ↩ REPLACED BY P18.5 |
| 21 | README + Docs Polish — README rewrite, architecture.md expand, api.md, configuration.md | ✅ COMPLETE |
| 22 | Security Hardening — rate limiting, body size limit, input validation, security headers | ✅ COMPLETE |
| 23 | Performance Tuning — SIMD distances, rayon IVF, madvise hints, O(1) avgdl, fusion fast-path | ✅ COMPLETE |
| 24 | Demo Notebook — Jupyter end-to-end walkthrough, synthetic 8-dim vectors, all search modes | ✅ COMPLETE |
| 25 | v0.1.0 release — Docker image, cargo publish, README final, changelog | ✅ COMPLETE |
| 26 | Pre-release polish — README, notebook, CI audit, LAUNCH.md | ✅ COMPLETE |

**Completed prompt summaries:**

- **P1:** Cargo workspace skeleton, `types.rs` (all core types: VectorId, Vector, CollectionConfig, VectorRecord, SearchResult, SearchRequest, UpsertRequest, UpsertResponse, DeleteRequest, IndexStats, DistanceMetric, IndexType), `errors.rs` (VecDbError with thiserror), `config.rs` (ServerConfig with config-crate loader), `.github/workflows/ci.yml`, stub modules in all crates, `server.rs` with full axum+tower-http wiring, route stubs returning 501.

- **P2:** `MmapVectorStore` (64-byte header: magic 0x56454344, version 1, dimension u64, count u64, capacity u64, timestamps; grow-by-doubling via anon mmap swap; O(1) random access), `WriteAheadLog` (serde_json framing: u32 LE length + payload + xxh3-32 checksum; seek-to-End before write + sync_all; replay via fresh `std::fs::read`), `MetadataStore` (SQLite WAL mode, PRAGMA synchronous=NORMAL, vectors table with soft-delete `deleted=1`, collections table, `list_active`, `count_active`, `list_collections`), `Storage` struct wiring all three with `create`, `open`, `upsert`, `delete`, `search`, `recover_from_wal`, `checkpoint`, `stats`, `rebuild_index`, `save_indexes`.

- **P3:** `IndexBackend` trait (build, insert, search, delete, save, load_from, len, is_empty, index_type, rebuild), `HnswIndex` (instant-distance wrapper; brute-force fallback when n≤100 or dirty=true; auto-rebuild at n.is_multiple_of(1000); soft-delete `deleted: HashSet<VectorId>`; save/load via serde_json to `{name}.hnsw.json`; `load_file()` inherent method bypasses trait object limitation), distance functions in `distance.rs` (`cosine_similarity`, `cosine_distance`, `euclidean_distance`, `dot_product`, `dot_product_distance`, `normalize`, `compute_distance`), `HnswConfig` (m, ef_construction, ef_search, metric — m is API-only, instant-distance ignores it).

- **P4:** `Tokenizer` (split on non-alphanum, lowercase fold, length filter 2–64, 100-word English stopword array hardcoded, optional suffix stripping; `tokenize()` removes stopwords for indexing; `tokenize_query()` KEEPS stopwords for search), `InvertedIndex` (BM25 formula: IDF=ln((N−df+0.5)/(df+0.5)+1); PostingList sorted by doc_id for O(log n) binary search; `score(query_terms, candidate_ids)` with optional filter; `search(query, k)` calls tokenize_query; `score_candidates(text, ids)` convenience wrapper; JSON save/load; `#[serde(skip)]` on `tokenizer: Tokenizer` field), `SparseIndex` facade, wired into Storage (`pub sparse` field, `search_sparse()`, `save_indexes()` saves both HNSW and sparse).

- **P5:** `min_max_normalize` (all-equal → 1.0 to avoid NaN), `softmax_normalize` (max-subtraction trick for stability), `weighted_sum_fusion` (min-max normalize both, union ids, alpha*dense + (1-alpha)*sparse), `reciprocal_rank_fusion` (1/(k+rank) summed per list, 1-indexed), `FusionStrategy` enum (`#[derive(Default)]` with `#[default]` on WeightedSum), `HybridEngine` (`new()` panics on bad alpha; `fuse()` handles pure-dense/pure-sparse/hybrid paths; `compute_oversample_k()`), `Storage::search_dense()`, `Storage::search_hybrid()` (HNSW oversample k×5 → BM25 re-score candidates or score_all → HybridEngine::fuse → hydrate from metadata).

- **P6:** `SortKey` enum (`Score | Field(String)`), full `LogicalNode` tree (VectorScan, SparseScan, HybridScan, Filter, Sort, Limit, Project, Join variants), `PhysicalPlan` (root, index_type, use_hybrid, pre_filter, filter_predicate: Option<String>, estimated_cost, candidate_k, output_k), `QueryPlanner` (cost model: `estimate_index_cost` = ln(N)*k*0.1, `should_use_hnsw(≥100)`, `should_pre_filter(<0.1 selectivity or >100k vectors)`, `compute_candidate_k` multipliers ×1/3/5/10 capped at 10k, `plan_search`, `plan_sql` stub replaced in P7, `explain`), `PlanExecutor` (scan stage → filter stage via `apply_json_filter` → sort desc → truncate to output_k), `Storage::execute_search`, `pub planner: QueryPlanner` on Storage.

- **P7:** Hand-written SQL lexer+parser in `crates/vecdb-core/src/planner/sql/`. Files: `ast.rs` (SelectStatement, Condition, VectorCondition, ScalarCondition, Operator, Literal, OrderBy), `lexer.rs` (Token enum + Lexer; handles VECTOR_SIM keyword, inline [f32,...] VectorLit, single-quoted strings with '' escape, negative numbers, ->/->>, all comparison operators), `parser.rs` (SqlParser::parse static entry; recursive descent: parse_select → parse_columns → parse_where_clause → parse_or_condition → parse_and_condition → parse_atom_condition → parse_vector_condition/parse_scalar_condition; expect() uses std::mem::discriminant; advance() returns owned Token clone), `converter.rs` (AstConverter {planner: QueryPlanner}; converts SelectStatement→PhysicalPlan; extracts VectorCondition as query_vector; collects AND-chain equality scalars into {"field": value} JSON so existing apply_json_filter works; extract_query_vector static method), `mod.rs` (all exports + mod tests), `tests.rs` (18 tests). `plan_sql` stub replaced with real impl. `Storage::execute_sql` added. `lib.rs` exports SqlParser, AstConverter, all AST types, Operator as SqlOperator.

- **P8:** Full axum HTTP API layer. New `[lib]` target added to `vecdb-api/Cargo.toml` (`name = "vecdb_api"`, `path = "src/lib.rs"`) alongside existing `[[bin]]`. `lib.rs` exposes 7 modules. `types.rs` defines all HTTP-layer request/response structs. `error.rs` defines `ApiError(VecDbError)` with `IntoResponse` mapping VecDbError variants to HTTP status codes and JSON `{"error":{"code":"...","message":"..."}}` body. `middleware.rs` implements `auth_middleware` (axum 0.7 signature, no generics): reads `X-Api-Key` header, compares to `state.config.api_key`, returns 401 if mismatch. `state.rs` defines `AppState` with `storage: Mutex<Storage>` (NOT RwLock — rusqlite::Connection is !Sync), `config: ServerConfig`, `collection_name: String`; type alias `SharedState = Arc<AppState>`. `server.rs` implements `run(config)`: creates Storage (open if `.db` exists, else create with dimension 1536), wraps in `AppState`, builds router with TraceLayer + TimeoutLayer + CorsLayer + auth middleware, binds TCP, calls `axum::serve`. `routes/mod.rs` defines the router with 11 routes. Each sub-module (`health.rs`, `collections.rs`, `vectors.rs`, `search.rs`) fully implements its handlers. `main.rs` uses clap, builds ServerConfig, calls `vecdb_api::server::run(config)`. 10 integration tests added (tests 1–10 in `routes/tests.rs`). Key fix: `tower = { version = "0.4", features = ["util"] }` in root Cargo.toml (needed for `ServiceExt::oneshot` in tests). Key fix: all `.read().await`/`.write().await` replaced with `.lock().await` because `Storage: !Sync`.

- **P12:** Pure-Rust IVF (Inverted File Index) backend. `ivf.rs`: `IvfIndex` struct with k-means training (deterministic init: `i * (n/n_lists)` spacing; dead-centroid reinit: `ci*7 % n`; convergence threshold 1e-6; DEFAULT_N_LISTS=256, DEFAULT_N_PROBE=16, DEFAULT_MAX_ITER=25). Degenerate case (n < n_lists): clamps n_lists to n.max(1). Insert: buffers in lists[0] when untrained; after training, assigns to nearest centroid; rebuilds on first insert (triggers initial training) and every 1000 inserts. Search: probes n_probe nearest centroids, brute-force across all lists when untrained. `AnyIndex` enum in `index/mod.rs` wraps HnswIndex + IvfIndex and implements IndexBackend by delegation. Three factory methods: `from_config`, `load_or_create`, `save_for_collection` — storage/mod.rs imports only AnyIndex (zero direct HnswIndex references). File naming: `{name}.hnsw.json` for HNSW, `{name}.ivf.json` for IVF. `Storage.index` type changed from `HnswIndex` to `AnyIndex`. 8 new IVF tests in `ivf_tests.rs`.

- **P21:** README + Docs Polish. `README.md` rewritten from stub to full production README (11 sections: What is vecdb, Features, Quickstart, Python SDK, TypeScript SDK, CLI, SQL Query Language, Architecture, Configuration, Benchmarks, License). `docs/architecture.md` expanded from inaccurate stub: corrected all errors (no FAISS, no bincode, no sqlparser-rs, no `VECDB_` single-underscore env vars); added Storage Layer detail (MmapVectorStore header layout, WAL framing, MetadataStore schema, write order), Index Layer (HnswIndex, IvfIndex, AnyIndex), Sparse Layer (tokenize vs tokenize_query), Hybrid Engine (two-stage pipeline, WeightedSum, RRF), Query Planner (PhysicalPlan fields, cost model, PlanExecutor pipeline), SQL Parser grammar, HTTP API Layer (CollectionManager locking, auth, metrics, graceful shutdown), full data flow ASCII diagram. `docs/api.md` created: all 14 endpoints with request/response tables, curl examples, filter syntax, error envelope docs, authentication section. `docs/configuration.md` created: env vars table (`VECDB__` double-underscore prefix, 7 variables), CLI flags table, full commented TOML example, docker-compose integration, storage layout, performance tuning notes. Rust workspace unchanged at 155 tests.

- **P23:** Performance Tuning — five targeted hot-path improvements in `vecdb-core` only; no API/CLI/bench changes. **(1) SIMD distance kernels** (`distance.rs`): `dot_product_simd` and `cosine_similarity_simd` use 8-element loop unrolling; LLVM auto-vectorizes to AVX2/SSE on x86-64; single-pass cosine computes dot, ‖a‖², ‖b‖² simultaneously; falls back to scalar for n<8; `compute_distance` now dispatches to SIMD variants for Cosine and DotProduct; exported from `index/mod.rs` as `cosine_similarity_simd`, `dot_product_simd`. **(2) Rayon IVF parallelism** (`ivf.rs`): `nearest_centroid_indices` uses `par_iter().enumerate()` for parallel centroid scoring; brute-force search path pre-collects filtered pairs serially then par-scores them; inline score conversion avoids `&self` borrow conflict in closures. **(3) mmap madvise hints** (`mmap.rs`): `#[cfg(target_os = "linux")]` blocks in `open()` (MADV_WILLNEED) and `get_all()` (MADV_SEQUENTIAL); no-ops on Windows/macOS. **(4) O(1) BM25 avgdl** (`inverted.rs`): `doc_count` renamed to `pub total_docs` with `#[serde(alias = "doc_count")]` for backward-compatible JSON deserialization; new `#[serde(default)] pub total_tokens: usize` maintained as a running sum by `index_document` (+= doc_length) and `remove_document` (-= removed_len via `saturating_sub`); `recompute_avg_doc_length` now does `total_tokens / total_docs` (O(1)) instead of `values().sum()` (O(n)); `load()` migrates old files by backfilling `total_tokens` from `doc_lengths` once on load; `doc_lengths` made `pub`; `doc_count()` accessor returns `total_docs`. **(5) Fusion normalization fast-path** (`fusion.rs`): `min_max_normalize` signature changed from `&[(VectorId, f32)] -> Vec<(VectorId, f32)>` to `&[f32] -> Vec<f32>`; single-pass fold finds min+max simultaneously; range < EPSILON fast-path returns `vec![1.0; n]`; `weighted_sum_fusion` extracts raw scores, normalizes, zips IDs back; existing fusion tests updated for new API. **(6) perf_tests.rs**: 8 new tests (dot_product_simd vs scalar, cosine_simd vs scalar, short-vector fallback, zero-vector, IVF parallel correctness, total_tokens maintained, avgdl O(1) matches manual, min_max_normalize new API). Total: 171 tests (119 core + 36 api + 8 cli + 8 bench). Zero clippy warnings.

- **P26:** Pre-release polish — README: badge URL fixed, port corrected to 6333, "Why vecdb?" section added, SDK install instructions fixed, copyright year updated; docker-compose.yml: port corrected to 6333, env var separator fixed to double-underscore (`VECDB__`); notebook: port corrected to 6333, SQL cells fixed to use `payload->>` field path syntax, sql3 fixed to include VECTOR_SIM predicate; release.yml and ci.yml verified correct; LAUNCH.md created with Show HN, r/rust, and r/selfhosted post drafts.

- **P25:** v0.1.0 Release — CHANGELOG.md, RELEASE.md, release.yml CI workflow, version bump to 0.1.0, vecdb-core publish metadata (description, license, repository, keywords, categories, readme).

- **P24:** Demo Notebook — pure Python + Jupyter; zero Rust changes; Rust workspace test count unchanged at 171. Three new files in `notebooks/`: **(1) `requirements.txt`**: `jupyter>=1.0.0`, `notebook>=7.0.0`, `vecdb-client @ ../sdks/python`, `requests>=2.31.0` — no GPU/ML deps. **(2) `README.md`**: prerequisites, Option A (Docker: `docker run -p 8080:8080 vecdb:latest`) and Option B (build from source: `cargo build --release -p vecdb-api`), install steps, run steps (`jupyter notebook vecdb_demo.ipynb`), 11-section feature table, notes on synthetic vectors/safe re-run/real embeddings path. **(3) `vecdb_demo.ipynb`**: valid nbformat 4 Jupyter notebook; 25 cells (13 markdown + 12 code); sections: Title, Setup, Create Collection, Define 20 Documents, Upsert, Dense Search, Sparse Search, Hybrid Search (main + alpha comparison loop), SQL Queries, Collection Stats, Get/Delete, Cleanup, Summary. Uses **synthetic 8-dimensional vectors** — no embedding model or GPU required. Key workarounds: `client.health()` returns a plain `dict` so accesses use `.get('version', '0.1.0')` not attribute syntax; `list_collections` SDK iterates wrong key on real server response so notebook uses `requests.get(BASE_URL+"/collections")` directly and unpacks `data.get("collections", [])`. All code cells are syntactically valid Python 3.9+. Unicode chars use `\uXXXX` escapes (valid JSON).

- **P22:** Security Hardening — all changes in `vecdb-api` only; zero `vecdb-core` changes. **(1) Request body size limit**: `RequestBodyLimitLayer::new(32 * 1024 * 1024)` from `tower-http` "limit" feature added as outermost layer. **(2) Rate limiting**: `tower_governor = "0.4"` (NOT 0.8 — getrandom 0.3 dlltool issue on Windows GNU) with `GovernorConfigBuilder::default().per_second(17).burst_size(1000)` for ~1000 req/min per IP; requires `into_make_service_with_connect_info::<SocketAddr>()`. **(3) Input validation**: `validate_collection_name(name: &str) -> Result<(), ApiError>` enforces 1–64 chars, `[a-zA-Z0-9_-]` only; called at start of every route handler that accepts a `:name` path param; returns HTTP 400 with VecDbError::InvalidQuery. **(4) Security response headers**: `security_headers` axum middleware inserts `x-content-type-options: nosniff`, `x-frame-options: DENY`, `x-xss-protection: 0`, `referrer-policy: strict-origin-when-cross-origin`. **(5) k/alpha/SQL bounds**: k validated 1–10000 in search handlers; alpha validated 0.0–1.0 in hybrid; SQL length ≤4096 chars; dimension 1–65536 in create_collection. **(6) Layer order** (last = outermost): RequestBodyLimitLayer → GovernorLayer → security_headers → TraceLayer → TimeoutLayer → CorsLayer → auth_middleware. 8 new tests (tests 29–36). Total: 163 tests (127→163). Zero clippy warnings.

- **P18.5:** Benchmark documentation + download scripts. Replaces P19 (MS MARCO run) and P20 (quantization) since dataset files are not locally available. `benchmarks/README.md`: full workflow docs — server setup, dataset download, embedding step, vecdb-bench usage, expected perf numbers placeholder. `benchmarks/scripts/download_msmarco.sh`: downloads MS MARCO (~8.8 GB), extracts TSV, converts to JSONL (`id`+`text`); `--dry-run` flag prints plan without downloading. `benchmarks/scripts/download_beir.sh`: downloads any of 5 BEIR datasets (default: scifact ~5k docs), validates dataset name, prints doc count after extraction. `benchmarks/scripts/run_synthetic_bench.sh`: checks binary + server health; runs 3 configs (small/medium/large: 1k/10k/50k vectors, dim=128); parses JSON output; writes markdown table to `benchmarks/results/synthetic_results.md`. `benchmarks/results/placeholder.md`: expected results format with TBD cells for MS MARCO and BEIR. Real benchmarks require: (1) run download script, (2) embed with model of choice, (3) ingest via vecdb-cli, (4) run vecdb-bench. Rust workspace unchanged at 155 tests. Shell validation scripts not counted as tests.

- **P18:** Docker + Deployment. Three infrastructure files at repo root + `scripts/`. `Dockerfile`: two-stage build — Stage 1 `rust:1.77-slim` installs `musl-tools`, adds `x86_64-unknown-linux-musl` target, builds `vecdb-api` as fully static binary; Stage 2 `alpine:3.19` copies only the binary, creates non-root `vecdb` user (uid 1000), creates `/data` volume, sets `HEALTHCHECK` via `wget`. `docker-compose.yml`: single `vecdb` service, named volume `vecdb-data`, port 8080, env vars `VECDB_DATA_DIR`/`VECDB_LOG_LEVEL`, `restart: unless-stopped`. `.dockerignore`: excludes `target/`, `sdks/`, `docs/`, `.git/`, `*.md`, `*.jsonl`, `*.csv`. `scripts/docker-health-check.sh`: manual health check via `wget`, exits 0 on `"status":"ok"`. `scripts/validate-docker.sh`: 5 shell tests verifying file content without Docker daemon — all pass. Rust workspace unchanged at 155 tests. No new counted tests (shell validation scripts only).

- **P17:** TypeScript client SDK. `sdks/typescript/` is a standalone ESM package outside the Cargo workspace. Zero runtime dependencies — uses native `fetch` (Node 18+, modern browsers). `src/errors.ts`: `VecDbError` class hierarchy (`NotFoundError`, `UnauthorizedError`, `InvalidRequestError`, `ServerError`, `ConnectionError`) with `raiseForStatus(response)` that parses `{"error":{"code":...,"message":...}}` JSON. `src/models.ts`: TypeScript interfaces only (no classes) — `CollectionConfig`, `CollectionInfo`, `VectorRecord`, `SearchResult`, `UpsertResponse`, `SearchResponse`, `HealthResponse`. `src/client.ts`: `VecDbClient` class; `_request<T>` uses `AbortController` + `setTimeout` for timeout; wraps network errors in `ConnectionError`; `listCollections` parses `response.collections` (not raw array); `createCollection` maps `indexType` camelCase → `index_type` snake_case. `package.json`: `type: "module"`, ESM exports, TypeScript 6.0.3, vitest 4.1.6, msw 2.14.6, @types/node ^22. `tsconfig.json`: `NodeNext` module + `rootDir: ./src` (required by TS6) + `lib: [ES2020, DOM]` (DOM required for `fetch`/`AbortController`/`Response` globals). `vitest.config.ts` provides actual vitest config (package.json `"vitest"` key is not read by vitest). Tests use `msw/node` `setupServer` + `http`/`HttpResponse` (msw v2 API). 8 vitest tests pass. Rust workspace unchanged at 155 tests.

- **P16:** Python client SDK. Pure Python, no PyO3, no Rust bindings. `sdks/python/` is a standalone package outside the Cargo workspace. Uses `httpx>=0.27` for sync (`VecDbClient`) and async (`AsyncVecDbClient`) HTTP. `models.py`: stdlib `@dataclass` only — `CollectionConfig`, `CollectionInfo`, `VectorRecord`, `SearchResult`, `UpsertResponse`, `SearchResponse` — all with `from_dict()` classmethods. `exceptions.py`: `VecDbError` hierarchy (`NotFoundError`, `UnauthorizedError`, `InvalidRequestError`, `ServerError`, `ConnectionError`) + `raise_for_status(response)` parses `{"error":{"code":...,"message":...}}` JSON. `client.py`: both classes share identical method signatures (collections CRUD, upsert, get_vector, delete_vectors, search_dense/sparse/hybrid, query_sql, health). Both support context manager protocol (`with`/`async with`). `pyproject.toml`: PEP 621, `setuptools.build_meta`, Python `>=3.9`, `asyncio_mode = "auto"`. Tests use `respx` mock transport — no live server. 8 pytest tests pass. Python `>=3.9` required. `respx` used for test mocking. Rust workspace unchanged at 155 tests.

- **P15:** Connection pooling + graceful shutdown. `MetadataStore` field changed from `conn: Connection` to `pool: Pool<SqliteConnectionManager>` (private) + `pub path: PathBuf`. `MetadataStore` now derives `Clone`. `build_pool(path, max_size)` creates `SqliteConnectionManager::file(path).with_init(|conn| conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA synchronous=NORMAL; PRAGMA foreign_keys=ON;"))` with `max_size` connections. Schema DDL (CREATE TABLE only, no PRAGMAs) runs once on a single connection at init. Private `conn()` helper acquires pooled connections: `self.pool.get().map_err(|e| VecDbError::StorageError(...))`. `open_with_pool_size(path, max_size)` added for testing. `server.rs` gains `shutdown_signal()` (listens for Ctrl+C + SIGTERM via `tokio::select!`); `axum::serve(...).with_graceful_shutdown(shutdown_signal()).await?`; after serve returns, write-locks `CollectionManager`, iterates `list()`, calls `checkpoint()` then `save_indexes()` per collection. New deps: `r2d2 = "0.8"`, `r2d2_sqlite = "0.25"` (rusqlite 0.31→0.32 upgraded for compatibility). getrandom 0.4.2 (transitive via scheduled-thread-pool) requires `dlltool.exe` — ensure `/c/msys64/mingw64/bin` is in PATH for all cargo commands. 3 pool tests in `storage/metadata.rs` (concurrent reads, concurrent upsert+read, pool-size-2 exhaustion). 2 API tests added (shutdown flush roundtrip, health endpoint with pooled metadata).

- **P14:** Column projection + full filter pushdown. `PhysicalPlan` gains `output_columns: Vec<String>`. `AstConverter::convert` captures `stmt.columns` before the statement is moved and sets `plan.output_columns` after `plan_search`. `apply_json_filter` rewritten as `pub(crate) fn apply_json_filter(record: &SearchResult, filter: &serde_json::Value) -> bool` — handles: array (AND-all), structured `{"field","op","value"}` (all 7 operators: `= != < <= > >= LIKE`), legacy flat `{"key": value}` (equality fallback for HTTP API compat). `resolve_field` handles "id" special-case and dot-notation nested paths. `like_match(text, pattern)` recursive `%`-wildcard. `apply_projection(results, columns)` builds new payload with only selected keys; empty/`["*"]` → passthrough; `id`/`score`/`text` are virtual columns. `PlanExecutor::execute` gains Stage 2a (allowed_ids filter from `plan.pre_filter`) and Stage 5 projection. `MetadataStore::filter_ids` loads all active rows and applies `apply_json_filter` for pre-filter optimization. `converter.rs` replaced `collect_equality_conditions` with `collect_all_conditions` emitting structured `{"field","op","value"}` JSON for ALL scalar operators. New `filter_tests.rs` with 10 tests: equality, not-equal, numeric gt/lte, LIKE prefix/contains, nested path, id field, projection SELECT, combined SQL filter+projection.

- **P13:** Multi-collection support. `collection_manager.rs`: `CollectionManager` struct with `HashMap<String, Arc<Mutex<Storage>>>` + `PathBuf data_dir`. Methods: `new`, `load_existing` (scans `.db` files on startup), `create` (returns `CollectionAlreadyExists` → 409 if duplicate), `get` (sync, returns `Option<Arc<Mutex<Storage>>>`), `delete` (removes map entry + 6 file suffixes: .db .wal .vectors .hnsw.json .ivf.json .sparse.json), `list` (sorted), `len`, `is_empty`, `all_arcs`, `insert` (test helper). `AppState` rewritten: `storage: Mutex<Storage>` + `config: ServerConfig` + `collection_name: String` removed; replaced by `collections: RwLock<CollectionManager>` + `metrics_handle` + `api_key: Option<String>`. Every route handler acquires a read lock on the manager, clones the `Arc<Mutex<Storage>>`, releases the lock, then acquires the storage mutex — zero contention between concurrent reads. `VecDbError::CollectionAlreadyExists` new variant → HTTP 409 Conflict in `error.rs`. SQL routes extract table name via `SqlParser::parse(&sql)?.table` (local variable renamed to `col` to satisfy grep check). `explain_query` computes plans while holding storage lock to avoid needing `QueryPlanner: Clone`. `server.rs` now calls `CollectionManager::new(data_dir)` + `manager.load_existing().await`. `middleware.rs` reads `state.api_key` directly. 8 new tests added (tests 19-26): create_201, duplicate_409, get_404_unknown, delete_200, delete_404_unknown, vectors_in_unknown_404, search_in_unknown_404, list_shows_created.

- **P9:** Prometheus metrics + observability layer. New file `crates/vecdb-api/src/metrics.rs` with 8 metric name constants and `install_recorder() -> PrometheusHandle`. `AppState` gains `metrics_handle: PrometheusHandle` field; `AppState::new()` updated accordingly. `server.rs` calls `install_recorder()` as the very first operation (before tracing), uses `try_init()` (not `.init()`) for tracing to survive multiple test initializations. `routes/mod.rs` gains `RequestTimer` struct (operation, collection, started: Instant) with `Drop` impl that records `vecdb_request_duration_ms` histogram; `start()` increments `vecdb_requests_total` counter. Real `metrics_handler` added using `State(state): State<SharedState>` extractor — renders `state.metrics_handle.render()` with `text/plain; version=0.0.4` content-type. `error.rs` gains `ApiError::kind_str() -> &'static str` and increments `vecdb_errors_total` counter in `IntoResponse`. Every route handler's first line is `let _timer = super::RequestTimer::start("operation_name", Some(&name))`. Search handlers additionally record `vecdb_search_duration_ms` histogram and `vecdb_search_results_count` gauge per search type. Upsert handler records `vecdb_upsert_total` counter; delete handler records `vecdb_delete_total` counter. `routes/tests.rs` updated: `OnceLock<PrometheusHandle>` + `get_or_init_metrics()` guard prevents double-install panic across parallel tests; `build_test_app()` uses 3-dim storage + api_key="test-key" + `std::mem::forget(dir)` to avoid Windows file-lock cleanup; `body_to_string()` helper for raw text responses. 8 new tests added (tests 11–18). Total: 108 tests (90 vecdb-core + 18 vecdb-api), 0 failures, 0 clippy warnings.

---

## 3. Every Architectural Decision and Deviation Made So Far

### 3.1 Serialization: serde_json everywhere, NEVER bincode
`serde_json::Value` is used in `VectorRecord.payload`. `bincode` 1.x cannot serialize/deserialize `serde_json::Value` because it calls `deserialize_any` which bincode does not implement. This affects: the WAL (embeds VectorRecord in WalEntry::Insert), the HNSW save/load (SerializedHnswIndex embeds vectors), and the sparse index save/load (InvertedIndex has HashMap fields). **All serialization uses `serde_json::to_vec`/`from_slice` or `to_string`/`from_str`.** `bincode` remains in `Cargo.toml` but is never used. Do not remove it (it may be used for a future performance path with a different payload type), but never call it.

### 3.2 instant-distance feature names: "serde" and "serde-big-array"
The correct feature names for instant-distance 0.6 are `"serde"` (not `"serialize"`) and `"serde-big-array"`. The `serde-big-array` feature is required because the HNSW graph structure uses `[PointId; M*2]` arrays where M=32, producing 64-element arrays that exceed serde's default array size limit of 32. Without `serde-big-array` the crate fails to compile with a derive error.

### 3.3 GNU toolchain, not MSVC
Development and CI run on `stable-x86_64-pc-windows-gnu` via MSYS2. The linker (`gcc`) lives at `/c/msys64/mingw64/bin`. Every cargo command in this project must use:
```
PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu <subcommand>
```
Never use the default toolchain or the MSVC toolchain. The CI yml uses `ubuntu-latest` with the default stable toolchain — CI and local builds use different toolchains, which is fine.

### 3.4 WAL: seek-to-End + sync_all instead of O_APPEND
Windows's `FILE_APPEND_DATA` access right (what `OpenOptions::append(true)` requests) has a quirk: data written via a handle with `FILE_APPEND_DATA` may not be visible to `std::fs::read()` in the same process until the handle is flushed or closed. The fix is to open the file with `read(true).write(true).create(true)`, then before each write call `self.file.seek(SeekFrom::End(0))`, write the framed entry, then call `self.file.sync_all()`. Replay opens the file independently via `std::fs::read(&self.path)` into a `Cursor` — this always reads the latest committed data regardless of any per-handle buffering.

### 3.5 WAL framing format
Each record on disk: `[4-byte LE u32 length][N-byte serde_json payload][4-byte LE u32 xxh3-checksum]`. The checksum is xxh3_64 of the payload bytes, truncated to u32. On replay, a checksum mismatch stops replay and treats everything after as corrupt. Entry types: `WalEntry::Insert { id, mmap_index, record }`, `WalEntry::Delete { id }`, `WalEntry::Checkpoint { entry_count, timestamp }`.

### 3.6 MmapVectorStore grow: anonymous mmap swap before resize
On Windows you cannot call `file.set_len(new_size)` while a `MmapMut` mapping is open on that file — it returns an OS error. The grow() method: (1) creates a 1-byte anonymous `MmapMut::map_anon(1)` placeholder, (2) swaps it into `self.mmap` via `std::mem::replace` (this drops the old mmap and releases the OS file mapping), (3) calls `self.file.set_len(new_size)`, (4) remaps with `MmapMut::map_mut(&self.file)`. Capacity doubles each grow.

### 3.7 MmapVectorStore header layout (64 bytes)
```
bytes  0.. 4  — magic u32 LE: 0x56454344 ("VECD")
bytes  4.. 8  — version u32 LE: 1
bytes  8..16  — dimension u64 LE
bytes 16..24  — count u64 LE (number of stored vectors)
bytes 24..32  — capacity u64 LE
bytes 32..40  — created_at unix u64 LE
bytes 40..48  — updated_at unix u64 LE
bytes 48..64  — reserved (zeroed)
```
Vector data starts at byte 64 (HEADER_SIZE constant). Each vector is `dimension * 4` bytes of packed little-endian f32.

### 3.8 HNSW M hardcoded at 32
`instant-distance` 0.6.1 hardcodes M=32 in its type system (the internal `[PointId; M*2]` array). `HnswConfig.m` exists for API compatibility and future backend replacement but has no effect on instant-distance's actual graph degree. All code that reads `HnswConfig.m` should treat it as a documentation field only.

### 3.9 HnswIndex auto-rebuild policy
- n ≤ 100: brute-force search always (HNSW too small to be useful)
- After insert when `n.is_multiple_of(1000)` (clippy prefers this over `n % 1000 == 0`): full rebuild from live vectors
- `dirty = true` after any insert/delete: forces brute-force until next rebuild
- Deletes: soft-delete only via `deleted: HashSet<VectorId>`; actual compaction happens via `build()` at rebuild time
- Compact when `deleted.len() > vectors.len() / 4`: trigger rebuild from `delete()`

### 3.10 Soft deletes everywhere
- `MetadataStore`: SQLite `deleted=1` flag, `updated_at` set on delete
- `HnswIndex`: in-memory `deleted: HashSet<VectorId>`; vectors stay in the `vectors: Vec<Vector>` array
- `MmapVectorStore`: no delete support — deleted vectors remain in the flat file forever until compaction (future prompt)
- `SparseIndex`/`InvertedIndex`: hard delete (actually removes from posting lists via `remove_document`)

### 3.11 Tokenizer: built-in stopword list, not stop-words crate
The `stop-words` crate was evaluated and rejected. A 100-word English stopword array `static ENGLISH_STOPWORDS: &[&str]` is hardcoded in `tokenizer.rs`. This avoids a crate dependency and keeps the build simple. The `Tokenizer` struct derives `Debug + Clone` (required because `InvertedIndex` derives those) but NOT `Serialize/Deserialize` — the `#[serde(skip)]` attribute on the `tokenizer: Tokenizer` field in `InvertedIndex` handles serialization by omitting it; `Tokenizer::default()` reconstructs it on load.

### 3.12 tokenize() vs tokenize_query() — CRITICAL
`Tokenizer::tokenize(text)`: removes stopwords. Used for **indexing documents**.
`Tokenizer::tokenize_query(text)`: keeps stopwords. Used for **search queries**.
**Never call `tokenize()` on a search query.** If you do, short queries like "to be or not" return zero results because all tokens are stopwords. Always use `tokenize_query` in search paths and `tokenize` in indexing paths.

### 3.13 save_index() renamed to save_indexes() in Prompt 4
When `SparseIndex` was added to `Storage`, `save_index()` was renamed to `save_indexes()`. It saves both `self.index` (HNSW → `{name}.hnsw.json`) and `self.sparse` (BM25 → `{name}.sparse.json`). Any future code that persists indexes must call `save_indexes()`. `save_index()` does not exist.

### 3.14 sqlparser-rs was NOT used — hand-written parser
`sqlparser-rs` was evaluated but rejected because hooking VECTOR_SIM as a custom function into its AST and visitor system was too complex and would require unsafe downcasting or significant workarounds. Instead, a focused hand-written recursive-descent parser in `crates/vecdb-core/src/planner/sql/` handles exactly the SQL subset vecdb needs. This avoids a heavy dependency and makes the grammar fully under our control.

### 3.15 apply_json_filter dual-format: structured (SQL) vs legacy flat (HTTP API)
`apply_json_filter(record, filter)` detects the filter format by shape:
- **Structured** (from SQL converter): `{"field":"f","op":"=","value":v}` — supports `= != < <= > >= LIKE`
- **Array of structured**: `[cond1, cond2]` — AND-all
- **Legacy flat** (from HTTP API `SearchRequest.filter`): `{"key": value, ...}` — equality-only for backward compat

The SQL `AstConverter` always emits structured format via `collect_all_conditions()` → `scalar_to_json()`. The HTTP API layer passes the raw `filter` JSON from the request body, which uses the legacy flat format. **Both shapes are handled simultaneously — do not remove either branch.**

### 3.35 PhysicalPlan.output_columns and apply_projection
`PhysicalPlan.output_columns: Vec<String>` carries the SELECT column list. Empty = `SELECT *` (no projection). `AstConverter::convert` sets it from `stmt.columns` (captured before stmt is moved into `SearchRequest`). `PlanExecutor::execute` calls `apply_projection(results, &plan.output_columns)` as the final stage. `apply_projection` builds a new payload keeping only listed keys; `id`/`score`/`similarity`/`text` are virtual — skipped in payload but `text` field is suppressed unless `"text"` is in the column list. `plan_search` always sets `output_columns: vec![]`.

### 3.36 MetadataStore::filter_ids for pre-filter optimization
`MetadataStore::filter_ids(filter) -> Result<Vec<VectorId>>` loads all active rows from SQLite, builds a dummy `SearchResult` per row, and calls `apply_json_filter`. Returns IDs of matching rows. Used by `PlanExecutor` when `plan.pre_filter = true`: results of vector scan are intersected with the allowed ID set before post-filter. This avoids scanning all vectors for highly selective filters. The intra-crate import `storage::metadata` → `planner::executor::apply_json_filter` is intentional and valid within one Rust crate.

### 3.37 MetadataStore uses r2d2 connection pool — conn field renamed pool
`MetadataStore.conn: Connection` was replaced by `pool: Pool<SqliteConnectionManager>` (private). `MetadataStore` now derives `Clone`. Every method acquires a connection via `self.conn()` helper which calls `self.pool.get().map_err(...)`. PRAGMAs (`journal_mode=WAL`, `busy_timeout=5000`, `synchronous=NORMAL`, `foreign_keys=ON`) are set on EVERY connection via `SqliteConnectionManager::with_init`. Schema DDL is run once on the first connection at `open()`. Pool size defaults to 8; `open_with_pool_size(path, max_size)` allows override for testing. **Never access `metadata.conn` — it no longer exists. Use `metadata.pool.get()` or call a method.**

### 3.38 Graceful shutdown — always flush before exit
`server.rs::shutdown_signal()` waits for Ctrl+C (`tokio::signal::ctrl_c()`) or SIGTERM (`tokio::signal::unix` on Unix, `std::future::pending::<()>()` on non-Unix). Wired as `axum::serve(...).with_graceful_shutdown(shutdown_signal())`. After serve drains in-flight requests and returns, `run()` write-locks `CollectionManager`, calls `storage.checkpoint()` then `storage.save_indexes()` for every open collection. Logs: "shutdown signal received", "flushing all collections before exit", "checkpointed '{name}'", "saved indexes for '{name}'", "vecdb shutdown complete". This ensures WAL entries and index state are always persisted on clean shutdown.

### 3.39 getrandom v0.4 on Windows GNU requires dlltool in PATH
r2d2 → scheduled-thread-pool 0.2.7 → rand 0.10 → rand_core 0.10 → getrandom 0.4.2. `getrandom` v0.4.2's build script needs `dlltool.exe` on Windows GNU toolchain. `dlltool.exe` is at `/c/msys64/mingw64/bin/dlltool.exe` on this machine. All cargo commands that build from scratch must use `PATH="$PATH:/c/msys64/mingw64/bin"`. (Already required per §3.3; getrandom 0.4 makes it a hard requirement for any new builds.) Note: rusqlite was upgraded from 0.31 → 0.32 to get r2d2_sqlite 0.25 which is compatible. The rusqlite API surface we use (execute, query_row, prepare, params!, error types) is unchanged between 0.31 and 0.32.

### 3.16 plan_sql signature takes vector_count as second argument
The old stub was `pub fn plan_sql(&self, _sql: &str) -> Result<PhysicalPlan>`. The real implementation is `pub fn plan_sql(&self, sql: &str, vector_count: usize) -> Result<PhysicalPlan>`. This is consistent with `plan_search` which also takes `vector_count` for the cost model. Any caller of `plan_sql` must pass the current `metadata.count_active()?` as the second argument.

### 3.17 execute_sql parses SQL once — not twice
`Storage::execute_sql` calls `SqlParser::parse(sql)` once to get the AST, then calls `AstConverter::extract_query_vector(&stmt)` (a no-copy read of the AST) and `AstConverter::convert(stmt, vector_count)` on the same AST. This is one parse total. **If you ever touch `execute_sql`, verify it still parses only once** — do not add a second `SqlParser::parse` call.

### 3.18 Operator exported as SqlOperator from lib.rs
`crate::planner::sql::ast::Operator` conflicts with any future `Operator` type at the top level (e.g., a query operator in the planner). It is therefore re-exported from `lib.rs` as `SqlOperator`:
```rust
pub use planner::sql::Operator as SqlOperator;
```

### 3.19 MetadataStore methods take &str, not &VectorId (&String)
Methods `upsert`, `get`, `delete` all take `id: &str`. Always pass string literals or `.as_str()`. This avoids &String vs &str friction.

### 3.20 HnswIndex::load_file() not IndexBackend::load_from()
`IndexBackend::load_from()` returns `Box<dyn IndexBackend>` and has `where Self: Sized` bound — making it unusable via trait objects. `Storage::open()` uses the concrete `HnswIndex::load_file(path, config) -> Result<Self>` inherent method instead.

### 3.21 MetadataStore has list_collections() for multi-collection support
`MetadataStore::list_collections() -> Result<Vec<CollectionConfig>>` reads from the `collections` table. This will be used in a future prompt (multi-collection) to enumerate all known collections on startup. Do not remove it.

### 3.22 Storage uses Mutex<Storage>, NOT RwLock — rusqlite is !Sync
`rusqlite::Connection` uses `RefCell<InnerConnection>` internally, making it `!Sync`. This means `Storage: !Sync` and therefore `tokio::sync::RwLock<Storage>: !Sync` — which makes `Arc<AppState>: !Send` and breaks the axum handler signature requirements. The fix, implemented in P8, is `tokio::sync::Mutex<Storage>` which only requires `T: Send` (not `T: Sync`) for `Mutex<T>: Sync`. All handlers use `.lock().await` — there is no `.read()` or `.write()`. **Never revert to RwLock.**

### 3.23 tower must have features = ["util"] for ServiceExt::oneshot in tests
`tower = { version = "0.4" }` with empty features does not expose `ServiceExt`, which is gated behind the `util` feature. The integration tests use `app.oneshot(req)` (from `tower::ServiceExt`). Without `features = ["util"]`, the build fails with "found an item that was configured out: gated behind the `util` feature". Fixed in root `Cargo.toml`:
```toml
tower = { version = "0.4", features = ["util"] }
```

### 3.24 vecdb-api has both [lib] and [[bin]] targets
`vecdb-api/Cargo.toml` defines `[lib]` (`name = "vecdb_api"`, `path = "src/lib.rs"`) alongside `[[bin]]` (`name = "vecdb"`, `path = "src/main.rs"`). This allows `main.rs` to call `vecdb_api::server::run(config)` and integration tests in `routes/tests.rs` to `use crate::...` directly. Without the `[lib]` section, test files inside the crate cannot reference internal modules via `crate::`.

### 3.25 Auth uses X-Api-Key header, not Authorization: Bearer
The auth middleware reads `request.headers().get("X-Api-Key")`. It does NOT use `Authorization: Bearer`. This matches the integration tests which all send `Header("X-Api-Key", key)`. Do not change to Bearer without updating all tests and the CONTEXT.md.

### 3.26 PrometheusHandle install_recorder() must be called exactly once per process
`PrometheusBuilder::new().install_recorder()` panics on the second call in the same process. In production (`server.rs`), it is called once at the top of `run()`. In tests (`routes/tests.rs`), a `static TEST_METRICS_HANDLE: OnceLock<PrometheusHandle>` ensures it is called at most once across all parallel tests:
```rust
static TEST_METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

fn get_or_init_metrics() -> PrometheusHandle {
    TEST_METRICS_HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("failed to install test recorder")
        })
        .clone()
}
```
**Never call `install_recorder()` outside of `server.rs` and this guard.** Never call it in individual test functions.

### 3.27 tracing_subscriber uses try_init(), not init() in server.rs
`tracing_subscriber::fmt()...init()` panics if called more than once in the same process (e.g., when running tests in the same process). `server.rs` uses `.try_init()` instead — it returns `Result` which is discarded with `let _ = ...`. This allows multiple test runs in the same process without panicking. Always use `try_init()` wherever tracing is initialized in this project.

### 3.28 Tests use std::mem::forget(dir) for TempDir on Windows
On Windows, SQLite keeps file handles open while Storage is alive. If `TempDir` is dropped before `Storage` (e.g., if the TempDir goes out of scope while the test is still running), Windows refuses to delete the directory and panics. In `build_test_app()`:
```rust
let dir = TempDir::new().unwrap();
let path = dir.path().to_path_buf();
std::mem::forget(dir);  // prevent cleanup — OS cleans up temp dirs on process exit
let storage = Storage::create(&path, &config).expect("create storage");
```
Tests that use `make_state(dir, ...)` keep `dir` alive for the duration via the binding in the test function — no forget needed there.

### 3.29 RequestTimer Drop impl records latency automatically
`RequestTimer::start(operation, collection)` increments `vecdb_requests_total` immediately. When the returned `_timer` goes out of scope (end of handler function, whether via Ok or Err return), the `Drop` impl records `vecdb_request_duration_ms` histogram automatically. Handlers must bind the return value: `let _timer = super::RequestTimer::start(...)`. Using `super::RequestTimer::start(...)` without binding creates a temporary that is dropped immediately — latency would always be ~0ms.

### 3.30 metrics_handler uses State extractor for PrometheusHandle
The `/metrics` handler cannot just call a global function — it reads the `PrometheusHandle` from `AppState`:
```rust
async fn metrics_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let body = state.metrics_handle.render();
    (StatusCode::OK, [("content-type", "text/plain; version=0.0.4")], body)
}
```
This is consistent with all other handlers and avoids global state. The handle is stored per-process (it IS a global recorder internally) but the API surface goes through state.

### 3.32 AnyIndex enum — storage uses only AnyIndex, never HnswIndex or IvfIndex directly
`AnyIndex` in `index/mod.rs` wraps both backends. `Storage.index` is `AnyIndex`. The three factory methods (`from_config`, `load_or_create`, `save_for_collection`) encapsulate all type-specific logic. `storage/mod.rs` imports only `AnyIndex` and `IndexBackend` — zero direct `HnswIndex`/`IvfIndex` references. IVF file naming: `{name}.ivf.json`. HNSW file naming unchanged: `{name}.hnsw.json`. `AnyIndex::load_or_create` uses `config.index_type` (loaded from SQLite via `metadata.load_collection`) to decide which file to look for — so collection configs survive across restarts.

### 3.33 IvfIndex circular-import avoidance — AnyIndex lives in index/mod.rs not backend.rs
`backend.rs` defines `IndexBackend` trait + `HnswIndex`. `ivf.rs` imports `IndexBackend` from `super::backend`. If `AnyIndex` lived in `backend.rs`, it would need to import `IvfIndex` from `ivf.rs`, creating a mutual import cycle. Instead, `AnyIndex` is defined in `index/mod.rs` (the crate root of the index module), which imports from both child modules without any cycle.

### 3.34 CollectionManager: RwLock<CollectionManager> wraps Arc<Mutex<Storage>> per collection
`AppState.collections` is `tokio::sync::RwLock<CollectionManager>`. This is valid because `CollectionManager` contains only `HashMap<String, Arc<Mutex<Storage>>>` and `PathBuf` — both `Sync`. Each `Storage` is behind its own `Arc<Mutex<Storage>>` (not RwLock — see §3.22). Route handlers: read-lock the manager, clone the Arc, release the read-lock, then lock the storage mutex. This means concurrent reads to different collections do not block each other, and concurrent reads to the same collection serialize at the Mutex level. The write lock is held only during `create` and `delete` (rare). Local variables that hold the extracted collection name must NOT be named `collection_name` — use `col` or `name` (grep check enforces this — `grep -r "collection_name" crates/vecdb-api/src/` must return zero).

### 3.31 Error body format: nested {"error": {"code": "...", "message": "..."}}
`ApiError::into_response()` produces:
```json
{ "error": { "code": "not_found", "message": "document 'xyz' not found" } }
```
NOT the flat `{ "error": "..." }` format. Tests verify the outer `error` key. When writing new handlers or tests, use `body["error"]["code"]` or `body["error"]["message"]` to access fields — not `body["error"]` as a string.

### 3.40 tower_governor version: MUST be 0.4, NOT 0.8
`tower_governor = "0.8"` (default) pulls in `governor = "0.10"` which depends on `getrandom 0.3`. `getrandom` v0.3's build script requires `dlltool.exe` on the Windows GNU toolchain — `error: error calling dlltool 'dlltool.exe': program not found`. Use `tower_governor = { version = "0.4" }` which resolves to v0.4.3 and uses `governor = "0.6.3"` / `getrandom = "0.2"`. Do not upgrade tower_governor to 0.5+ without verifying the getrandom transitive dep is still ≤0.2.

### 3.41 min_max_normalize signature changed in P23: &[f32] → Vec<f32>
`min_max_normalize` in `fusion.rs` was changed from `&[(VectorId, f32)] -> Vec<(VectorId, f32)>` to `&[f32] -> Vec<f32>`. Callers must extract scores first, normalize, then zip IDs back — `weighted_sum_fusion` does this. The function is re-exported from `vecdb_core::hybrid` and `vecdb_core::lib.rs`; any external code using the old signature will break. The existing fusion tests (test_min_max_normalize_basic/all_equal/empty) were updated to use `&[f32]` slices.

### 3.42 InvertedIndex field rename: doc_count → total_docs; new total_tokens field
`InvertedIndex.doc_count: usize` was renamed to `pub total_docs: usize` with `#[serde(alias = "doc_count")]`. Old JSON files with `"doc_count"` deserialize correctly. New field `#[serde(default)] pub total_tokens: usize` is the running sum of document lengths; defaults to 0 on load, then migrated via `load()` by summing `doc_lengths` once. `doc_lengths` is now `pub`. The `doc_count()` accessor returns `self.total_docs`. Do NOT revert to the O(n) `values().sum()` approach in `recompute_avg_doc_length`.

### 3.43 IVF brute-force scoring uses inline match instead of self.distance_to_score()
In the rayon-parallelized brute-force path, `self.distance_to_score(dist)` cannot be used inside a `par_iter()` closure because it takes `&self` which would conflict with the borrow on `self.metric`. Instead, an inline `match metric { ... }` block is used (identical logic). `distance_to_score` is still used in the probe-based (non-brute-force) path where no borrow conflict exists.

---

## 4. Complete Current File Tree

Every file on disk right now. ✅ = fully implemented, ⚠️ = stub/partial.

```
vecdb/
├── Cargo.toml                                    ✅ workspace root — tower has features = ["util"]
├── Cargo.lock
├── README.md                                     ✅
├── CHANGELOG.md                                  ✅ Keep a Changelog format, v0.1.0 entry (P25)
├── RELEASE.md                                    ✅ step-by-step release process, targets, binary naming (P25)
├── LAUNCH.md                                     ✅ Show HN + r/rust + r/selfhosted post drafts (P26)
├── CONTEXT.md                                    ✅ this file
├── docs/
│   ├── architecture.md                          ✅ full component breakdown, data flow, storage/index/planner detail
│   ├── api.md                                   ✅ all 14 endpoints, request/response tables, curl examples, error envelope
│   └── configuration.md                         ✅ VECDB__ env vars, CLI flags, TOML example, storage layout, tuning
├── Dockerfile                                    ✅ two-stage: rust:1.77-slim (musl builder) → alpine:3.19 (runtime)
├── docker-compose.yml                            ✅ single service, named volume, healthcheck, restart:unless-stopped
├── .dockerignore                                 ✅ excludes target/, sdks/, docs/, .git/, *.md, *.jsonl, *.csv
├── .gitignore
├── scripts/
│   ├── docker-health-check.sh                   ✅ manual wget health check, exits 0 on "status":"ok"
│   └── validate-docker.sh                       ✅ 5 shell validation tests (no Docker daemon required)
├── benchmarks/
│   ├── README.md                                ✅ full benchmark workflow docs
│   ├── results/
│   │   ├── placeholder.md                       ✅ expected results format (TBD cells for MS MARCO + BEIR)
│   │   └── synthetic_results.md                 ✅ overwritten by run_synthetic_bench.sh
│   └── scripts/
│       ├── download_msmarco.sh                  ✅ downloads ~8.8GB MS MARCO, converts TSV→JSONL; --dry-run flag
│       ├── download_beir.sh                     ✅ downloads any of 5 BEIR datasets (default: scifact)
│       └── run_synthetic_bench.sh               ✅ 3 bench configs → markdown table in results/
├── .github/
│   └── workflows/
│       ├── ci.yml                               ✅ ubuntu-latest, test+clippy+fmt+doc
│       └── release.yml                          ✅ tag-triggered, 4 targets, cross for musl, gh-release (P25)
├── docs/
│   └── architecture.md                          ✅ 5-layer ASCII diagram
└── crates/
    ├── vecdb-core/
    │   ├── Cargo.toml                            ✅
    │   └── src/
    │       ├── lib.rs                            ✅ all re-exports including SQL types
    │       ├── config.rs                         ✅ ServerConfig, from_env_and_file
    │       ├── errors.rs                         ✅ VecDbError, Result type alias
    │       ├── types.rs                          ✅ all core types
    │       ├── hybrid/
    │       │   ├── mod.rs                        ✅
    │       │   └── fusion.rs                     ✅ FusionStrategy, HybridEngine, normalize, fuse
    │       ├── index/
    │       │   ├── mod.rs                        ✅ AnyIndex enum + from_config/load_or_create/save_for_collection
    │       │   ├── backend.rs                    ✅ IndexBackend trait, HnswIndex, HnswConfig
    │       │   ├── distance.rs                   ✅ cosine/euclidean/dot/normalize/compute + SIMD kernels (P23)
    │       │   ├── ivf.rs                        ✅ IvfIndex (k-means, n_lists=256, n_probe=16) + rayon (P23)
    │       │   ├── ivf_tests.rs                  ✅ 8 IVF tests
    │       │   └── perf_tests.rs                 ✅ 8 performance tests (P23)
    │       ├── planner/
    │       │   ├── mod.rs                        ✅ pub mod sql + mod filter_tests
    │       │   ├── executor.rs                   ✅ PlanExecutor, apply_json_filter (all ops), apply_projection
    │       │   ├── filter_tests.rs               ✅ 10 filter+projection tests
    │       │   ├── plan.rs                       ✅ QueryPlanner, PhysicalPlan (+ output_columns), LogicalNode, SortKey
    │       │   └── sql/
    │       │       ├── mod.rs                    ✅ all exports + mod tests
    │       │       ├── ast.rs                    ✅ SelectStatement, Condition, VectorCondition, etc.
    │       │       ├── lexer.rs                  ✅ Token enum, Lexer
    │       │       ├── parser.rs                 ✅ SqlParser recursive descent
    │       │       ├── converter.rs              ✅ AstConverter — all ops, structured filter JSON, output_columns
    │       │       └── tests.rs                  ✅ 18 tests (test 15 updated for structured filter format)
    │       ├── sparse/
    │       │   ├── mod.rs                        ✅ SparseIndex facade + score/score_all
    │       │   ├── inverted.rs                   ✅ InvertedIndex, PostingList, BM25; total_docs/total_tokens O(1) avgdl (P23)
    │       │   └── tokenizer.rs                  ✅ Tokenizer, TokenizerConfig, ENGLISH_STOPWORDS
    │       └── storage/
    │           ├── mod.rs                        ✅ Storage struct + all methods including execute_sql; AnyIndex
    │           ├── metadata.rs                   ✅ MetadataStore (r2d2 pool, Clone), filter_ids(), pool_tests
    │           ├── mmap.rs                       ✅ MmapVectorStore, 64-byte header, grow; madvise hints (P23)
    │           └── wal.rs                        ✅ WriteAheadLog, WalEntry, framed format
    ├── vecdb-api/
    │   ├── Cargo.toml                            ✅ [lib] + [[bin]] targets, uuid + chrono + tempfile
    │   └── src/
    │       ├── lib.rs                            ✅ 8 pub mod declarations (+ collection_manager)
    │       ├── main.rs                           ✅ clap Args, calls vecdb_api::server::run
    │       ├── collection_manager.rs             ✅ CollectionManager (HashMap<String, Arc<Mutex<Storage>>>)
    │       ├── metrics.rs                        ✅ 8 metric constants + install_recorder()
    │       ├── error.rs                          ✅ ApiError, kind_str(), IntoResponse; 409 for CollectionAlreadyExists
    │       ├── middleware.rs                     ✅ auth_middleware + validate_collection_name + security_headers (P22)
    │       ├── server.rs                         ✅ run(), shutdown_signal(), graceful flush; rate limit + body limit layers (P22)
    │       ├── state.rs                          ✅ AppState (RwLock<CollectionManager> + PrometheusHandle + api_key)
    │       ├── types.rs                          ✅ all HTTP request/response structs; ExplainRequest.collection added
    │       └── routes/
    │           ├── mod.rs                        ✅ router(), metrics_handler(), RequestTimer + Drop
    │           ├── health.rs                     ✅ GET /health — aggregates across all collections
    │           ├── collections.rs                ✅ POST/GET /collections, GET/DELETE /collections/:name
    │           ├── vectors.rs                    ✅ POST/DELETE /collections/:name/vectors, GET /:name/vectors/:id
    │           ├── search.rs                     ✅ dense/sparse/hybrid search, /query (SQL→col), /explain
    │           └── tests.rs                      ✅ 28 tests (1-10 API, 11-18 metrics, 19-26 multi-collection, 27-28 shutdown)
    ├── vecdb-bench/
    │   ├── Cargo.toml                            ✅ reqwest+rayon+rand+indicatif; no vecdb-core
    │   └── src/
    │       ├── main.rs                           ✅ clap CLI; all flags; table/json output
    │       ├── client.rs                         ✅ BenchClient (post/delete_no_body, X-Api-Key, 30s)
    │       ├── types.rs                          ✅ mirror structs (UpsertRequest, DenseSearchRequest, SearchResponse, …)
    │       ├── datasets.rs                       ✅ generate_synthetic (seeded StdRng, L2-normalize); load_jsonl
    │       ├── runner.rs                         ✅ BenchmarkRunner::run; compute_recall; percentile
    │       └── tests.rs                          ✅ 8 unit tests
    └── vecdb-cli/
        ├── Cargo.toml                            ✅ reqwest 0.12, wiremock dev-dep; no vecdb-core
        └── src/
            ├── main.rs                           ✅ Cli struct, 5 subcommands, VecDbClient init
            ├── client.rs                         ✅ VecDbClient (get/post/delete_no_body, X-Api-Key, 30s timeout)
            ├── types.rs                          ✅ mirror structs (no vecdb-core imports)
            ├── output.rs                         ✅ print_health, print_collection, print_search_results, print_upsert_result
            ├── tests.rs                          ✅ 8 wiremock integration tests
            └── commands/
                ├── mod.rs                        ✅
                ├── collection.rs                 ✅ List/Get/Create/Delete via HTTP
                ├── ingest.rs                     ✅ JSONL + CSV (RFC 4180); batch upsert; dry_run
                ├── inspect.rs                    ✅ GET /collections/:name
                └── search.rs                     ✅ Dense/Sparse/Hybrid/Sql subcommands
├── notebooks/                                        ✅ Jupyter demo notebook (P24) — no Rust changes
│   ├── requirements.txt                             ✅ jupyter, notebook, vecdb-client, requests (no GPU deps)
│   ├── README.md                                    ✅ start server (Docker/source), install deps, run notebook
│   └── vecdb_demo.ipynb                             ✅ 25 cells (13 md + 12 code); 8-dim synthetic vectors; all search modes
└── sdks/
    ├── python/                                       ✅ pure Python httpx SDK (no PyO3, no Rust bindings)
    │   ├── pyproject.toml                            ✅ PEP 621, requires Python >=3.9, httpx>=0.27
    │   ├── README.md                                 ✅
    │   ├── vecdb/
    │   │   ├── __init__.py                           ✅ exports all public symbols
    │   │   ├── client.py                             ✅ VecDbClient (sync) + AsyncVecDbClient (async)
    │   │   ├── models.py                             ✅ dataclasses: CollectionInfo/Config, VectorRecord, SearchResult/Response, UpsertResponse
    │   │   └── exceptions.py                         ✅ VecDbError hierarchy + raise_for_status()
    │   └── tests/
    │       ├── conftest.py                           ✅ fixtures: base_url, sample data, respx mock routers
    │       └── test_client.py                        ✅ 8 pytest tests (sync + async, respx mocks)
    └── typescript/                                   ✅ pure TS ESM SDK — native fetch, Node 18+, zero runtime deps
        ├── package.json                              ✅ type:module, ESM exports, TS 6.0.3, vitest 4.1.6, msw 2.14.6
        ├── tsconfig.json                             ✅ NodeNext, rootDir:./src, lib:[ES2020,DOM]
        ├── vitest.config.ts                          ✅ actual vitest config (package.json "vitest" key not read by vitest 4.x)
        ├── README.md                                 ✅
        ├── src/
        │   ├── index.ts                              ✅ re-exports all public symbols with .js extensions
        │   ├── client.ts                             ✅ VecDbClient — fetch+AbortController, listCollections→response.collections
        │   ├── models.ts                             ✅ TypeScript interfaces (no classes)
        │   └── errors.ts                             ✅ VecDbError hierarchy + raiseForStatus()
        └── tests/
            ├── setup.ts                              ✅ msw/node setupServer, beforeAll/afterEach/afterAll
            └── client.test.ts                        ✅ 8 vitest tests (msw v2 http/HttpResponse API)
        ├── pyproject.toml                            ✅ PEP 621, requires Python >=3.9, httpx>=0.27
        ├── README.md                                 ✅
        ├── vecdb/
        │   ├── __init__.py                           ✅ exports all public symbols
        │   ├── client.py                             ✅ VecDbClient (sync) + AsyncVecDbClient (async)
        │   ├── models.py                             ✅ dataclasses: CollectionInfo/Config, VectorRecord, SearchResult/Response, UpsertResponse
        │   └── exceptions.py                         ✅ VecDbError hierarchy + raise_for_status()
        └── tests/
            ├── conftest.py                           ✅ fixtures: base_url, sample data, respx mock routers
            └── test_client.py                        ✅ 8 pytest tests (sync + async, respx mocks)
```

---

## 5. Current Test Count and Status

**Rust total: 171 tests, 0 failures, 0 clippy warnings.**
- 119 in `vecdb-core` (111 existing + 8 new perf_tests from P23)
- 36 in `vecdb-api` (routes::tests — 28 existing + 8 new security/rate-limit tests from P22)
- 8 in `vecdb-cli` (wiremock integration tests)
- 8 in `vecdb-bench` (pure-compute unit tests)

**Note (P24):** `notebooks/` contains only Python + Jupyter files. No Rust code was added or modified. Rust workspace test count is unchanged at 171.

**Python total: 8 pytest tests, 0 failures.**
- 8 in `sdks/python/tests/test_client.py` (respx mocks, sync + async, error cases)

**TypeScript total: 8 vitest tests, 0 failures.**
- 8 in `sdks/typescript/tests/client.test.ts` (msw v2 mocks, sync await, error cases)

Run Rust:
```
PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu test --workspace
```

Run Python (requires Python >=3.9):
```
cd sdks/python
pip install -e ".[dev]"
pytest tests/ -v
```

Run TypeScript (requires Node 18+):
```
cd sdks/typescript
npm install
npm test
npm run build
```

Run API tests only:
```
PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu test -p vecdb-api -- routes::tests
```

Run core tests only:
```
PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu test -p vecdb-core
```

### `crates/vecdb-api/src/routes/tests.rs` — 18 tests

**Tests 1–10 (HTTP API layer):**
```
test_health_endpoint                  — GET /health → 200, body.status == "ok"
test_upsert_and_get_vector            — POST 3 records, GET one by id, verify vector values
test_dense_search                     — upsert 5 vecs, dense search k=3, ≤3 results with scores
test_sparse_search                    — upsert 5 vecs with text, sparse search → results array
test_hybrid_search                    — upsert 5 vecs, hybrid search → dense_score or sparse_score present
test_sql_query                        — upsert 5 vecs, POST /query with SQL VECTOR_SIM → results array
test_delete_vector                    — upsert, delete, GET → 404
test_auth_middleware_blocks_without_key — api_key set, no header → 401
test_auth_middleware_allows_with_correct_key — api_key set, correct header → 200
test_upsert_dimension_mismatch        — one good + one bad-dim record → errors array non-empty
```

**Tests 11–18 (Prometheus metrics):**
```
test_metrics_endpoint_returns_200                    — GET /metrics with key → 200
test_metrics_endpoint_returns_prometheus_content_type — content-type contains "text/plain"
test_requests_total_increments_after_health_check    — after GET /health, /metrics body contains "vecdb_requests_total"
test_metrics_contains_duration_histogram             — after GET /health, /metrics body contains "vecdb_request_duration_ms"
test_upsert_increments_upsert_total                  — after POST vectors, /metrics body contains "vecdb_upsert_total"
test_errors_total_increments_on_not_found            — after GET unknown id, /metrics body contains "vecdb_errors_total"
test_search_results_gauge_present_after_dense_search — after dense search, /metrics body contains "vecdb_search_results_count"
test_metrics_endpoint_requires_auth                  — GET /metrics without key → 401
```

### `crates/vecdb-core/src/planner/sql/tests.rs` — 18 tests

```
test_lexer_keywords                 — SELECT FROM WHERE AND OR NOT ORDER BY ASC DESC LIMIT LIKE all tokenize
test_lexer_comparison_operators     — = != <> < <= > >= all produce correct Token variants
test_lexer_vector_literal           — [1.0, -2.5, 0.3] produces VectorLit with 3 f32 values
test_lexer_string_escaping          — 'it''s a test' → StringLit("it's a test")
test_lexer_arrows                   — -> and ->> produce Arrow and ArrowText tokens
test_parse_select_star              — SELECT * FROM mycol LIMIT 5 → table=mycol, columns=[], limit=Some(5)
test_parse_vector_sim_condition     — VECTOR_SIM(embedding, [1.0,0.0,0.5]) > 0.8 → VectorCondition
test_parse_scalar_equality          — WHERE category = 'science' → ScalarCondition with Eq
test_parse_combined_condition       — VECTOR_SIM AND scalar → And(Vector, Scalar)
test_parse_order_by_desc            — ORDER BY score DESC → OrderBy{field:"score", descending:true}
test_parse_not_condition            — AND NOT active = false → Not(Scalar) on right of And
test_parse_or_condition             — WHERE ... AND (cat='A' OR cat='B') → And top-level confirmed
test_parse_invalid_sql_errors       — empty/"SELECT FROM"/"SELECT * docs" all return Err
test_converter_dense_vector_plan    — extracts 4-f32 query_vector, plan.output_k=5, not hybrid
test_converter_filter_is_json_object — filter_predicate parses as {"field":"category","op":"=","value":"science"}
test_execute_sql_dense              — 10 docs inserted, SQL VECTOR_SIM query → ≤5 results, sorted
test_execute_sql_with_filter        — 5 "A" + 5 "B" docs, SQL with AND cat='A' → only A results
test_lexer_negative_numbers_in_vector — [-1.5, 2.0, -0.5, 3.14] → all 4 values correct
```

### `crates/vecdb-core/src/planner/filter_tests.rs` — 10 tests (P14)

```
test_filter_equality_string         — region="US" matches, "EU" does not
test_filter_not_equal               — status!="inactive" matches active; status!="active" does not
test_filter_numeric_gt              — price>100 matches 150; price>200 does not
test_filter_numeric_lte             — score<=0.5 matches 0.5; score<=0.4 does not
test_filter_like_prefix             — LIKE "hello%" matches "hello world"; LIKE "world%" does not
test_filter_like_contains           — LIKE "%world%" matches; LIKE "%xyz%" does not
test_filter_nested_path             — meta.region="EU" navigates nested JSON
test_filter_id_field                — field="id" resolves to record.id not payload
test_projection_keeps_only_selected_columns — SELECT id, region: region present, title/price absent
test_storage_sql_with_filter_and_projection — SELECT id, region WHERE region='US': only US, no 'extra'
```

### `crates/vecdb-core/src/planner/plan.rs` — 11 tests

```
test_plan_dense_only           — vector only → VectorScan, use_hybrid=false, HNSW
test_plan_sparse_only          — text only → SparseScan, use_hybrid=false
test_plan_hybrid               — both → HybridScan, use_hybrid=true, candidate_k≥50
test_plan_with_filter          — filter present → Filter node in tree, filter_predicate=Some
test_cost_model_small_collection — should_use_hnsw: 50→false, 100→true, 1M→true
test_candidate_k_hybrid_with_filter — (10,true,true)=100, (10,true,false)=50, (10,false,false)=10
test_explain_output            — contains "HybridScan", "Limit(10)", "use_hybrid: true"
test_executor_dense_search     — Storage.execute_search dense, ≤5 results, sorted desc
test_executor_hybrid_search    — Storage.execute_search hybrid, results sorted desc
test_executor_invalid_no_inputs — vector=None text=None → Err(InvalidQuery)
test_pre_filter_threshold      — (0.05,50k)=true, (0.5,50k)=false, (0.5,200k)=true
```

### `crates/vecdb-core/src/index/perf_tests.rs` — 8 tests (P23)

```
test_dot_product_simd_matches_scalar     — dim=1536, simd vs scalar agree within 1e-2
test_cosine_similarity_simd_matches_scalar — dim=1536, simd vs scalar agree within 1e-4
test_simd_short_vector_fallback          — n=4 < 8: dot and cosine fall back to scalar, exact match
test_cosine_simd_zero_vector             — zero vec → 0.0 (no NaN/panic), via simd kernel
test_ivf_search_parallel_correctness     — 30 vecs, query≈v15; v15 in top-3, results sorted desc
test_inverted_total_tokens_maintained    — total_tokens == sum(doc_lengths) after insert+remove
test_inverted_avgdl_matches_manual       — O(1) avg_doc_length() matches manual sum/count after each op
test_min_max_normalize_new_api           — &[f32]→Vec<f32>: basic, all-equal EPSILON path, empty
```

### `crates/vecdb-core/src/hybrid/fusion.rs` — 10 tests

```
test_min_max_normalize_basic        — [0.0, 0.5, 1.0] → [0.0, 0.5, 1.0] (P23: &[f32] API)
test_min_max_normalize_all_equal    — all 0.7 → all 1.0 (EPSILON fast-path)
test_min_max_normalize_empty        — empty input → empty output, no panic
test_weighted_sum_fusion_pure_dense — alpha=1.0 → "a" (highest dense) ranks first
test_weighted_sum_fusion_pure_sparse — alpha=0.0 → "b" (highest sparse) ranks first
test_weighted_sum_fusion_doc_only_in_one_list — "c" only sparse; all 3 ids in output
test_rrf_basic                      — b(rank2 dense + rank1 sparse) wins; math verified
test_hybrid_engine_fuse_weighted_sum — 3 results, both scores populated, sorted desc
test_hybrid_engine_fuse_empty_sparse — pure dense path: sparse_score=None for all
test_hybrid_engine_fuse_empty_dense  — pure sparse path: dense_score=None for all
```

### `crates/vecdb-core/src/index/distance.rs` — 7 tests

```
test_cosine_identical        — cosine_similarity(v,v) ≈ 1.0
test_cosine_orthogonal       — cosine_similarity([1,0],[0,1]) ≈ 0.0
test_cosine_opposite         — cosine_similarity([1,0],[-1,0]) ≈ -1.0
test_euclidean_zero          — euclidean_distance(v,v) == 0.0
test_euclidean_known         — euclidean_distance([0,0],[3,4]) ≈ 5.0
test_normalize               — normalize([3,4]) has magnitude 1.0
test_zero_vector_cosine      — cosine_similarity(zero, other) == 0.0 (no panic)
```

### `crates/vecdb-core/src/index/backend.rs` — 9 tests

```
test_distance_cosine_basic          — 4 vecs, search top-2, checks cosine score > 0.99
test_euclidean_search               — near/far/medium, query [1.1,1.1], expects "near"
test_brute_force_fallback           — nulls hnsw field, dirty=false → brute force path
test_soft_delete                    — delete "a", verifies absent from results
test_save_and_load                  — save HNSW to JSON, reload, verify top result same
test_dimension_mismatch_on_insert   — insert 3-dim vector into dim-4 index → DimensionMismatch
test_storage_search_integration     — 10 docs via Storage.search, scores all in [0,1]
test_index_rebuild_after_reopen     — save_indexes(), Storage.open, search still works
test_auto_rebuild_on_large_insert   — insert 1000 vectors, search for v500 returns v500
```

### `crates/vecdb-core/src/sparse/tokenizer.rs` — 6 tests

```
test_basic_tokenize             — "the quick brown fox" → stopwords removed, lowercase
test_query_keeps_stopwords      — tokenize_query keeps "to", "be"; doc tokenize removes them
test_min_length_filter          — "a I go running" → "a" and "I" filtered (len<2 or stopword)
test_compute_term_frequencies   — [dog,cat,dog] → {dog:2, cat:1}
test_strip_suffixes             — "running jumped faster" with strip_suffixes=true
test_punctuation_split          — "hello, world! foo-bar baz.qux" splits on all non-alphanum
```

### `crates/vecdb-core/src/sparse/inverted.rs` — 12 tests

```
test_index_and_doc_count            — index 2 docs, doc_count() == 2
test_bm25_relevance_order           — "rust" twice in d1, once in d2; d1 scores higher
test_remove_document                — remove d1, search for "rust" → empty
test_remove_nonexistent_returns_err — remove "ghost" → Err(NotFound)
test_upsert_updates_scores          — re-index d1 with new content; old terms gone, new terms present
test_candidate_filter               — score() with [d1] filter only returns d1
test_save_and_load                  — serialize to JSON, load, search still works
test_idf_decreases_with_more_docs   — IDF for "common" decreases as more docs indexed
test_avg_doc_length_updates         — avg_doc_length() > 0 after indexing 2 docs
test_empty_query_returns_empty      — search("", 10) → empty results
test_posting_list_sorted            — insert z_doc/a_doc/m_doc → binary search order maintained
test_posting_list_get               — get("alpha") found, get("gamma") None
```

### `crates/vecdb-core/src/storage/mod.rs` — 17 tests

```
test_mmap_create_and_open           — create store, write 3 vecs, reopen, verify all 3 correct
test_mmap_dimension_mismatch        — append wrong-dim vec → DimensionMismatch
test_mmap_grow                      — capacity=2, append 3 vecs forces grow, all readable after
test_wal_append_and_replay          — 3 inserts + 1 delete, replay returns 4 entries in order
test_wal_checkpoint_and_truncate    — 3 inserts, checkpoint, 2 more, truncate → 2 entries remain
test_metadata_upsert_and_get        — upsert doc1, get doc1, verify id/text/payload/mmap_index
test_metadata_soft_delete           — delete doc2, get returns NotFound, count_active == 0
test_storage_full_roundtrip         — 5 upserts, delete 1, reopen, verify count 4, checkpoint
test_wal_recovery                   — crash without checkpoint, reopen recovers r1/r2/r3
test_sparse_search_after_upsert     — upsert with text, search_sparse("rust database") → v1 first
test_sparse_delete_removes_from_index — upsert, delete, search_sparse returns empty
test_save_indexes_and_reload        — save_indexes(), reopen Storage, search_sparse works
test_storage_search_hybrid_both_signals — 5 docs, vector+text, 3 results, sorted, scores≥0
test_storage_search_hybrid_dense_only — no text field, vector-only hybrid works
test_storage_search_hybrid_sparse_only — query_vector=None, sparse-only hybrid works
test_storage_search_hybrid_no_inputs_errors — None+None → Err(InvalidQuery)
test_storage_search_dense_method    — search_dense: dense_score=Some, sparse_score=None
```

---

## 6. Complete Dependency List

### Root `Cargo.toml` — `[workspace.dependencies]`
```toml
tokio = { version = "1.36", features = ["full"] }
axum = { version = "0.7", features = ["json", "http1", "tokio"] }
tower = { version = "0.4", features = ["util"] }       # "util" required for ServiceExt::oneshot in tests
tower-http = { version = "0.5", features = ["trace", "cors", "timeout", "limit"] }  # "limit" added P22 for RequestBodyLimitLayer
tower_governor = { version = "0.4" }    # P22 rate limiting — MUST be 0.4 (0.8+ → getrandom 0.3 → dlltool fail)
serde = { version = "1.0", features = ["derive"] }
serde_json = { version = "1.0" }
bincode = { version = "1.3" }           # present but never used — serde_json::Value breaks bincode
rusqlite = { version = "0.32", features = ["bundled", "serde_json"] }
rayon = { version = "1.9" }
thiserror = { version = "1.0" }
anyhow = { version = "1.0" }
tracing = { version = "0.1" }
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
metrics = { version = "0.22" }
metrics-exporter-prometheus = { version = "0.13" }
ordered-float = { version = "4.2", features = ["serde"] }
uuid = { version = "1.7", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
clap = { version = "4.5", features = ["derive", "env"] }  # "env" required for #[arg(env = "...")]
config = { version = "0.14" }
xxhash-rust = { version = "0.8", features = ["xxh3"] }
memmap2 = { version = "0.9" }
instant-distance = { version = "0.6", features = ["serde", "serde-big-array"] }
tempfile = { version = "3.10" }
reqwest = { version = "0.12", default-features = false, features = ["json", "rustls-tls"] }  # 0.12 not 0.13 — 0.13 uses aws-lc-rs which fails on Windows GNU
wiremock = { version = "0.6" }
rand = { version = "0.8" }          # synthetic dataset generation (bench)
indicatif = { version = "0.17" }    # progress bars (bench)
```

### `crates/vecdb-core/Cargo.toml`
```toml
[dependencies]
serde = { workspace = true }
serde_json = { workspace = true }
bincode = { workspace = true }
rusqlite = { workspace = true }
rayon = { workspace = true }
thiserror = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }
metrics = { workspace = true }
ordered-float = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }
xxhash-rust = { workspace = true }
config = { workspace = true }
memmap2 = { workspace = true }
instant-distance = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

### `crates/vecdb-api/Cargo.toml`
```toml
[lib]
name = "vecdb_api"
path = "src/lib.rs"

[[bin]]
name = "vecdb"
path = "src/main.rs"

[dependencies]
vecdb-core = { path = "../vecdb-core" }
tokio = { workspace = true }
axum = { workspace = true }
tower = { workspace = true }
tower-http = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
metrics = { workspace = true }
metrics-exporter-prometheus = { workspace = true }
anyhow = { workspace = true }
config = { workspace = true }
clap = { workspace = true }
uuid = { workspace = true }
chrono = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

### `crates/vecdb-cli/Cargo.toml`
```toml
[[bin]]
name = "vecdb"
path = "src/main.rs"

[dependencies]
tokio = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
reqwest = { workspace = true }  # no vecdb-core dep

[dev-dependencies]
wiremock = { workspace = true }
tempfile = { workspace = true }
tokio = { workspace = true }
```

### `crates/vecdb-bench/Cargo.toml`
```toml
[[bin]]
name = "vecdb-bench"
path = "src/main.rs"

[dependencies]
tokio = { workspace = true }
clap = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
tracing = { workspace = true }
tracing-subscriber = { workspace = true }
reqwest = { workspace = true }   # no vecdb-core dep
rayon = { workspace = true }
rand = { workspace = true }
indicatif = { workspace = true }

[dev-dependencies]
tempfile = { workspace = true }
```

---

## 7. Complete Storage Struct State

### Struct definition (`crates/vecdb-core/src/storage/mod.rs`)

```rust
pub struct Storage {
    pub vectors: MmapVectorStore,    // flat f32 mmap file — {name}.vectors
    pub wal: WriteAheadLog,          // serde_json framed append log — {name}.wal
    pub metadata: MetadataStore,     // SQLite: records, payloads, collections — {name}.db
    pub index: HnswIndex,            // dense HNSW index (instant-distance)
    pub sparse: SparseIndex,         // BM25 inverted index
    pub hybrid: HybridEngine,        // fusion engine (default: alpha=0.7, WeightedSum, oversample=5)
    pub planner: QueryPlanner,       // cost model, plan_search, plan_sql
    pub config: CollectionConfig,    // dimension, metric, hnsw params, bm25 params
    data_dir: PathBuf,               // private: root directory for all collection files
}
```

### File naming convention (all under `data_dir/`)
```
{name}.vectors       — MmapVectorStore flat binary file (64-byte header + packed f32 arrays)
{name}.wal           — WriteAheadLog (serde_json framed entries)
{name}.db            — SQLite MetadataStore (vectors table + collections table)
{name}.hnsw.json     — HnswIndex serialized JSON (written by save_indexes)
{name}.sparse.json   — InvertedIndex serialized JSON (written by save_indexes)
```

### Every method signature on Storage

```rust
// Construction
pub fn create(data_dir: &Path, config: &CollectionConfig) -> Result<Self>
pub fn open(data_dir: &Path, collection_name: &str) -> Result<Self>

// Mutation
pub fn upsert(&mut self, record: VectorRecord) -> Result<UpsertResult>
pub fn delete(&mut self, id: &VectorId) -> Result<()>

// Search
pub fn search(&self, query: &Vector, k: usize) -> Result<Vec<SearchResult>>
pub fn search_dense(&self, query_vector: &Vector, k: usize) -> Result<Vec<SearchResult>>
pub fn search_sparse(&self, query: &str, k: usize) -> Result<Vec<SearchResult>>
pub fn search_hybrid(
    &self,
    query_vector: Option<&Vector>,
    query_text: Option<&str>,
    k: usize,
    alpha: f32,
    strategy: FusionStrategy,
) -> Result<Vec<SearchResult>>
pub fn execute_search(&self, request: &SearchRequest) -> Result<Vec<SearchResult>>
pub fn execute_sql(&self, sql: &str) -> Result<Vec<SearchResult>>

// Maintenance
pub fn rebuild_index(&mut self) -> Result<()>
pub fn save_indexes(&self) -> Result<()>        // saves both HNSW and sparse
pub fn recover_from_wal(&mut self) -> Result<usize>
pub fn checkpoint(&mut self) -> Result<()>
pub fn stats(&self) -> Result<IndexStats>
```

### Call order inside `create(data_dir, config)`
1. `std::fs::create_dir_all(data_dir)`
2. `MmapVectorStore::create(&vectors_path, config.dimension, 1024)`
3. `WriteAheadLog::open(&wal_path)` (creates file if absent)
4. `MetadataStore::open(&metadata_path)` (runs SCHEMA CREATE IF NOT EXISTS)
5. `metadata.save_collection(config)` (INSERT OR REPLACE into collections table)
6. `HnswIndex::from_collection_config(config)` (empty, no HNSW built yet)
7. `SparseIndex::create(&sparse_path)` (empty InvertedIndex)
8. `HybridEngine::default()` (alpha=0.7, WeightedSum, oversample=5)
9. `QueryPlanner::new(config.clone())`
10. Return `Storage { vectors, wal, metadata, index, sparse, hybrid, planner, config, data_dir }`

### Call order inside `open(data_dir, collection_name)`
1. `MmapVectorStore::open(&vectors_path)`
2. `WriteAheadLog::open(&wal_path)`
3. `MetadataStore::open(&metadata_path)`
4. `metadata.load_collection(collection_name)` → `config`
5. If `{name}.hnsw.json` exists: `HnswIndex::load_file(&index_path, &config)` else `HnswIndex::from_collection_config(&config)`
6. `SparseIndex::open(&sparse_path)` (loads if `.sparse.json` exists, else creates empty)
7. `HybridEngine::default()`, `QueryPlanner::new(config.clone())`
8. Construct `storage` struct
9. `let recovered = storage.recover_from_wal()?`
10. If `!index_path.exists() || recovered > 0`: `storage.rebuild_index()?`
11. Log and return `storage`

### Call order inside `upsert(record)`
1. Dimension check: `record.vector.len() != self.config.dimension` → `DimensionMismatch`
2. `self.metadata.get(&record.id)`:
   - `Ok((existing_mmap_idx, _))` → `mmap_index = existing_mmap_idx`, result = `Updated`
   - `Err(NotFound)` → `mmap_index = self.vectors.append(&record.vector)?`, result = `Inserted`
   - Other error → return error
3. `self.wal.append(&WalEntry::Insert { id, mmap_index, record: record.clone() })`
4. If `Updated`: `self.vectors.overwrite(mmap_index, &record.vector)?`
5. `self.metadata.upsert(&id, mmap_index, &record)?`
6. `self.index.insert(record.id.clone(), record.vector.clone())?`
7. If `record.text.is_some() && !text.is_empty()`: `self.sparse.index_document(&id, text)?`
8. Return `Ok(UpsertResult::Inserted)` or `Ok(UpsertResult::Updated)`

### Call order inside `delete(id)`
1. `self.wal.append(&WalEntry::Delete { id: id.clone() })?`
2. `self.metadata.delete(id)?` (sets `deleted=1` in SQLite, returns NotFound if already deleted)
3. `let _ = self.index.delete(id)` (soft-delete in HashSet; ignores NotFound)
4. `let _ = self.sparse.remove_document(id)` (hard-delete from posting lists; ignores NotFound)

### Call order inside `rebuild_index()`
1. `let active = self.metadata.list_active()?` → `Vec<(VectorId, usize)>`
2. For each `(id, mmap_idx)`: `self.vectors.get(mmap_idx)?` → collect `pairs: Vec<(VectorId, Vector)>`
3. `self.index.build(pairs)?` (rebuilds HNSW from scratch)
4. `self.sparse = SparseIndex::create(&self.data_dir.join(format!("{}.sparse.json", self.config.name)))`
5. For each active `id`: `self.metadata.get(id)` → if `rec.text.is_some() && !text.is_empty()`: `self.sparse.index_document(id, text)`

### Call order inside `save_indexes()`
1. `self.index.save(&hnsw_path)?` → writes `{name}.hnsw.json` as serde_json
2. `self.sparse.save()?` → writes `{name}.sparse.json` as serde_json

### Call order inside `execute_search(request)`
1. `let vector_count = self.metadata.count_active()?`
2. `let plan = self.planner.plan_search(request, vector_count)?`
3. `tracing::debug!("Query plan:\n{}", self.planner.explain(&plan))`
4. `PlanExecutor::new(self).execute(&plan, request.vector.as_ref(), request.query_text.as_deref())`

### Call order inside `execute_sql(sql)`
1. `use crate::planner::sql::{AstConverter, SqlParser}`
2. `let vector_count = self.metadata.count_active()?`
3. `let stmt = SqlParser::parse(sql)?`
4. `let query_vector = AstConverter::extract_query_vector(&stmt)`
5. `let converter = AstConverter::new(QueryPlanner::new(self.config.clone()))`
6. `let plan = converter.convert(stmt, vector_count)?`
7. `tracing::debug!("SQL plan:\n{}", self.planner.explain(&plan))`
8. `PlanExecutor::new(self).execute(&plan, query_vector.as_ref(), None)`

### Call order inside `search_hybrid(query_vector, query_text, k, alpha, strategy)`
1. Validate: both None → `InvalidQuery`; dimension check if vector provided
2. `oversample_k = k * 5`
3. Dense: if `query_vector.is_some()`: `self.index.search(qv, oversample_k)` → `dense_results`; else `vec![]`
4. Sparse: if `query_text.is_some()`:
   - if `!dense_results.is_empty()`: extract candidate ids → `self.sparse.score(qt, &candidate_ids)?`
   - else: `self.sparse.score_all(qt, oversample_k)?`
5. `HybridEngine::new(alpha, strategy, 5).fuse(dense_results, sparse_results, k)?`
6. Hydrate: for each result, `self.metadata.get(&result.id)` → set `result.payload` and `result.text`

---

## 8. Complete AppState and HTTP Layer Architecture

### AppState (`crates/vecdb-api/src/state.rs`)

```rust
pub struct AppState {
    pub storage: tokio::sync::Mutex<Storage>,   // Mutex NOT RwLock — Storage: !Sync (rusqlite)
    pub config: ServerConfig,
    pub collection_name: String,                 // single collection: "default"
    pub metrics_handle: PrometheusHandle,        // for rendering /metrics
}

impl AppState {
    pub fn new(
        storage: Storage,
        config: ServerConfig,
        collection_name: String,
        metrics_handle: PrometheusHandle,
    ) -> Self { ... }
}

pub type SharedState = Arc<AppState>;
```

### Metrics module (`crates/vecdb-api/src/metrics.rs`)

```rust
pub const REQUESTS_TOTAL: &str = "vecdb_requests_total";
pub const ERRORS_TOTAL: &str = "vecdb_errors_total";
pub const UPSERT_TOTAL: &str = "vecdb_upsert_total";
pub const DELETE_TOTAL: &str = "vecdb_delete_total";
pub const REQUEST_DURATION_MS: &str = "vecdb_request_duration_ms";
pub const SEARCH_DURATION_MS: &str = "vecdb_search_duration_ms";
pub const SEARCH_RESULTS_COUNT: &str = "vecdb_search_results_count";
pub const INDEX_SIZE_VECTORS: &str = "vecdb_index_size_vectors";  // reserved, not yet recorded

pub fn install_recorder() -> PrometheusHandle { ... }  // panics on second call
```

### RequestTimer (`crates/vecdb-api/src/routes/mod.rs`)

```rust
pub struct RequestTimer {
    operation: &'static str,
    collection: String,
    started: std::time::Instant,
}

impl RequestTimer {
    pub fn start(operation: &'static str, collection: Option<&str>) -> Self {
        // increments vecdb_requests_total{operation, collection}
        // logs "request started"
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        // records vecdb_request_duration_ms{operation, collection} histogram
        // logs "request finished" with elapsed_ms
    }
}
```

### Full route table

| Method | Path | Handler | Timer collection arg |
|--------|------|---------|---------------------|
| GET | /health | health::health | None |
| POST | /collections | collections::create_collection | None |
| GET | /collections | collections::list_collections | None |
| GET | /collections/:name | collections::get_collection | Some(&name) |
| DELETE | /collections/:name | collections::delete_collection | Some(&name) |
| POST | /collections/:name/vectors | vectors::upsert_vectors | Some(&name) |
| DELETE | /collections/:name/vectors | vectors::delete_vectors | Some(&name) |
| GET | /collections/:name/vectors/:id | vectors::get_vector | Some(&name) |
| POST | /collections/:name/search/dense | search::search_dense | Some(&name) |
| POST | /collections/:name/search/sparse | search::search_sparse | Some(&name) |
| POST | /collections/:name/search/hybrid | search::search_hybrid | Some(&name) |
| POST | /query | search::query_sql | None |
| POST | /explain | search::explain_query | None |
| GET | /metrics | routes::metrics_handler | — (no timer) |

### Error response shape (from error.rs)

```json
{ "error": { "code": "not_found", "message": "document 'xyz' not found" } }
```

HTTP status codes:
- 200 OK — success
- 201 Created — new collection created
- 400 Bad Request — InvalidQuery
- 401 Unauthorized — missing/wrong X-Api-Key
- 404 Not Found — NotFound, CollectionNotFound
- 422 Unprocessable Entity — DimensionMismatch
- 500 Internal Server Error — StorageError, IndexError, SparseError, MetadataError, SerializationError, IoError, SqliteError

---

## 9. Complete Public API Surface of vecdb-core

### lib.rs re-exports (what is directly importable as `vecdb_core::X`)

```rust
pub mod config;
pub mod errors;
pub mod hybrid;
pub mod index;
pub mod planner;
pub mod sparse;
pub mod storage;
pub mod types;

pub use config::ServerConfig;
pub use errors::{Result, VecDbError};
pub use hybrid::{min_max_normalize, softmax_normalize, FusionStrategy, HybridEngine};
pub use planner::{LogicalNode, PhysicalPlan, PlanExecutor, QueryPlanner, SortKey};
pub use planner::sql::{
    AstConverter, SqlParser,
    SelectStatement, Condition, VectorCondition, ScalarCondition,
    Literal, Operator as SqlOperator, OrderBy,
};
pub use types::*;    // all types in types.rs re-exported at top level
```

### types.rs (all re-exported at top level via `pub use types::*`)

```rust
pub type VectorId = String;
pub type Vector = Vec<f32>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DistanceMetric { #[default] Cosine, Euclidean, DotProduct }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum IndexType { #[default] HNSW, IVF }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index_type: IndexType,
    pub hnsw_m: usize,                // default 16 (API-only; instant-distance uses M=32)
    pub hnsw_ef_construction: usize,  // default 200
    pub hnsw_ef_search: usize,        // default 50
    pub bm25_k1: f32,                 // default 1.5
    pub bm25_b: f32,                  // default 0.75
    pub created_at: DateTime<Utc>,
}
impl CollectionConfig { pub fn new(name: impl Into<String>, dimension: usize) -> Self }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: VectorId,
    pub vector: Vector,
    pub payload: serde_json::Value,  // CRITICAL: this field breaks bincode
    pub text: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: VectorId,
    pub score: f32,               // final/fused score
    pub dense_score: Option<f32>, // None in pure sparse path
    pub sparse_score: Option<f32>,// None in pure dense path
    pub payload: serde_json::Value,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Option<Vector>,
    pub query_text: Option<String>,
    #[serde(default = "default_k")]       // default 10
    pub k: usize,
    #[serde(default = "default_alpha")]   // default 0.7
    pub alpha: f32,
    pub collection: String,
    pub filter: Option<serde_json::Value>,
}

pub struct UpsertRequest { pub records: Vec<VectorRecord>, pub collection: String }
pub struct UpsertResponse { pub inserted: usize, pub updated: usize, pub errors: Vec<String> }
pub struct DeleteRequest { pub ids: Vec<VectorId>, pub collection: String }
pub struct IndexStats {
    pub collection: String, pub vector_count: usize, pub index_type: IndexType,
    pub dimension: usize, pub disk_bytes: u64, pub memory_bytes: u64,
}
```

### errors.rs

```rust
pub enum VecDbError {
    StorageError(String),
    IndexError(String),
    SparseError(String),
    MetadataError(String),
    NotFound { id: String },
    DimensionMismatch { expected: usize, got: usize },
    CollectionNotFound(String),
    InvalidQuery(String),
    SerializationError(String),
    IoError(#[from] std::io::Error),
    SqliteError(#[from] rusqlite::Error),
}
pub type Result<T> = std::result::Result<T, VecDbError>;
```

### config.rs

```rust
pub struct ServerConfig {
    pub host: String,           // default "127.0.0.1"
    pub port: u16,              // default 6333
    pub data_dir: PathBuf,      // default "./data"
    pub log_level: String,      // default "info"
    pub api_key: Option<String>,// default None
    pub max_connections: usize, // default 100
    pub query_timeout_ms: u64,  // default 5000
}
impl ServerConfig {
    pub fn from_env_and_file(path: Option<&str>) -> anyhow::Result<Self>
}
impl Default for ServerConfig { ... }
```

---

## 10. Complete SQL Grammar Supported

### Supported tokens (lexer)
```
Keywords:    SELECT FROM WHERE AND OR NOT ORDER BY ASC DESC LIMIT LIKE TRUE FALSE
Functions:   VECTOR_SIM  EMBED  (EMBED is lexed but not yet handled by parser/converter)
Operators:   =  !=  <>  <  <=  >  >=  LIKE
Punctuation: ( ) [ ] , . * ;
JSON path:   ->  (Arrow)   ->>  (ArrowText)
Literals:
  Identifier:  [a-zA-Z_][a-zA-Z0-9_]*
  StringLit:   'text'  with  ''  escape for embedded single-quote
  IntLit:      optional leading minus, digits only
  FloatLit:    optional leading minus, digits . digits, optional e/E exponent
  VectorLit:   [ f32 , f32 , ... ]  inline (negative components supported)
```

### Grammar (approximate EBNF)
```ebnf
select_stmt  ::= SELECT columns FROM identifier
                 [WHERE condition]
                 [ORDER BY identifier [ASC|DESC]]
                 [LIMIT integer]
                 [;]

columns      ::= * | identifier (, identifier)*

condition    ::= or_cond
or_cond      ::= and_cond (OR and_cond)*
and_cond     ::= atom (AND atom)*
atom         ::= NOT atom
               | ( or_cond )
               | vector_cond
               | scalar_cond

vector_cond  ::= VECTOR_SIM ( identifier , vector_lit ) operator number

scalar_cond  ::= field_path operator literal

field_path   ::= identifier (. identifier | -> string_or_ident | ->> string_or_ident)*

operator     ::= = | != | <> | < | <= | > | >= | LIKE

literal      ::= StringLit | FloatLit | IntLit | TRUE | FALSE

number       ::= FloatLit | IntLit

vector_lit   ::= [ ]
               | [ float (, float)* ]
```

### Example queries that work
```sql
SELECT * FROM mycollection WHERE VECTOR_SIM(embedding, [1.0, 0.5, -0.3]) > 0.8 LIMIT 10
SELECT * FROM docs WHERE VECTOR_SIM(vec, [0.1, 0.2]) > 0.7 AND category = 'science'
SELECT * FROM articles WHERE VECTOR_SIM(v, [1.0]) >= 0.5 ORDER BY score DESC LIMIT 5
SELECT * FROM col WHERE VECTOR_SIM(emb, [0.0, 1.0]) > 0.5 AND active = true LIMIT 20
SELECT id, score FROM collection WHERE VECTOR_SIM(e, [0.1]) > 0.4
```

### What is explicitly NOT supported
- **OR at the top level of WHERE** — AstConverter only collects equality conditions from AND chains
- **Subqueries**, **JOIN in SQL**, **INSERT/UPDATE/DELETE/CREATE**
- **EMBED() function** — lexed but parser/converter do not handle it
- **Arithmetic in WHERE**, **IS NULL**, **IN (...)**, **BETWEEN**
- **Multiple tables in FROM**

---

## 11. Strict Rules That Must Never Be Violated

1. **serde_json everywhere, never bincode.** `serde_json::Value` in `VectorRecord.payload` breaks `bincode` 1.x. WAL, HNSW save, sparse save all use `serde_json`. Never call bincode.

2. **Always add new deps to BOTH files.** Root `Cargo.toml` `[workspace.dependencies]` AND the specific crate's `Cargo.toml` as `{ workspace = true }`. Never add directly to a crate without the workspace entry.

3. **Run `cargo search <crate>` before adding any new crate.** Verify exact crate name, version, available feature names. Never guess feature names.

4. **Run all verification checks before marking a prompt done:**
   ```
   PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu fmt --all
   PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu build --workspace
   PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu test --workspace
   PATH="$PATH:/c/msys64/mingw64/bin" cargo +stable-x86_64-pc-windows-gnu clippy --workspace -- -D warnings
   ```
   All must pass clean. Zero warnings. Fix everything before declaring done.

5. **Never leave `unimplemented!()`, `todo!()`, or stub `Err(...)` in completed prompt code.**

6. **Fix every clippy warning before finishing.** Common traps: redundant closures, `or_default()` vs `or_insert_with(Default::default)`, `n.is_multiple_of(1000)` vs `n % 1000 == 0`, derivable Default impls.

7. **Never rename a public method without grepping all call sites first.** `grep -rn "method_name" crates/` across all crates before any rename. `save_indexes()` not `save_index()`.

8. **`tokenize_query()` for search, `tokenize()` for indexing.** Never swap.

9. **GNU toolchain always.** Every cargo command needs `+stable-x86_64-pc-windows-gnu` and `PATH="$PATH:/c/msys64/mingw64/bin"`.

10. **MetadataStore methods take `&str`, not `&VectorId`.** Pass `.as_str()` or string literals.

11. **`HnswIndex::load_file()` not `IndexBackend::load_from()`.** The trait method requires downcasting. Use the inherent method.

12. **SearchResult.score must be consistent.** Dense: `dense_score=Some(s)`, `sparse_score=None`, `score=s`. Sparse: `sparse_score=Some(s)`, `dense_score=None`, `score=s`. Hybrid: both Some, `score=fused_value`.

13. **AstConverter filter predicate must be a JSON object.** `filter_predicate` must parse as `serde_json::Object`. Do not change format without updating executor.rs.

14. **plan_sql takes two arguments.** Signature: `pub fn plan_sql(&self, sql: &str, vector_count: usize) -> Result<PhysicalPlan>`.

15. **execute_sql must parse SQL only once.** Do not add a second `SqlParser::parse` call.

16. **Confirm test count before and after every prompt.** Run `cargo test --workspace 2>&1 | grep "test result"`. New count must equal old count + exactly the promised new tests.

17. **Never touch a previously completed file without reading it first.** Use the Read tool.

18. **AppState.storage is `Mutex<Storage>`, not RwLock.** `rusqlite::Connection: !Sync`. All handlers use `.lock().await`. Never change to RwLock.

19. **tower must have `features = ["util"]` in workspace root.** Required for `ServiceExt::oneshot` in tests.

20. **install_recorder() called exactly once per process.** In production: top of `server::run()`. In tests: via `OnceLock<PrometheusHandle>` in `tests.rs`. Never call it in individual test functions.

21. **Use `try_init()` for tracing, not `init()`.** `init()` panics on second call; `try_init()` returns Result and is safe to ignore.

22. **RequestTimer: bind the return value.** `let _timer = super::RequestTimer::start(...)` — not just `super::RequestTimer::start(...)`. Unbound temporaries drop immediately, recording 0ms latency.

23. **Error body is nested.** Access via `body["error"]["code"]` not `body["error"]`. The shape is `{"error": {"code": "...", "message": "..."}}`.

24. **vecdb-api has [lib] and [[bin]].** Do not remove the `[lib]` section — it enables test modules to use `crate::` and enables `main.rs` to call `vecdb_api::server::run`.

25. **`std::mem::forget(dir)` in build_test_app().** TempDir on Windows panics if dropped while Storage files are open. Only `build_test_app()` needs this — `make_state()` tests keep dir alive via the binding.

---

## 12. What Prompt 10 Implemented

**Prompt 10: CLI Implementation** — `vecdb-cli` reimplemented as a pure HTTP client binary. No vecdb-core dependency. Talks to a running `vecdb-api` server via reqwest.

### Architecture

- **Binary name:** `vecdb` (declared via `[[bin]]` in vecdb-cli/Cargo.toml)
- **`client.rs`:** `VecDbClient` wraps `reqwest::Client` with base URL + X-Api-Key default header + 30s timeout. Methods: `get`, `post`, `delete_no_body`. Non-2xx → clear `anyhow::Error`; 401 → "set --api-key or VECDB_API_KEY".
- **`types.rs`:** Mirror structs with `Serialize`/`Deserialize`, no vecdb-core imports.
- **`output.rs`:** All stdout formatting. `print_search_results` renders aligned table (ID/SCORE/DENSE/SPARSE columns).
- **`commands/collection.rs`:** `List/Get/Create/Delete` → CRUD on `/collections` endpoints.
- **`commands/ingest.rs`:** Reads JSONL or CSV (auto-detected by extension). Manual RFC 4180 CSV parser (no `csv` crate). Batches records; `--dry-run` parses only, no HTTP calls. `--batch-size` defaults to 100.
- **`commands/inspect.rs`:** `GET /collections/:name` → pretty-prints `CollectionResponse`.
- **`commands/search.rs`:** `Dense/Sparse/Hybrid/Sql` subcommands. Vector parsed from JSON array string.
- **`tests.rs`:** 8 wiremock tests — each starts a real HTTP server on a random port, no live vecdb-api needed.

### Key implementation details
- reqwest 0.12 (not 0.13) — 0.13 uses `aws-lc-rs` which fails to link on Windows GNU (`nanosleep64`); 0.12 uses `ring` which is pure Rust.
- reqwest features: `default-features = false, features = ["json", "rustls-tls"]`
- clap `"env"` feature required for `#[arg(env = "...")]`
- `UpsertRecord` derives `Clone` — needed for `chunk.to_vec()` in batch loop.

### 8 CLI wiremock tests
- `test_ping_hits_health_endpoint` — asserts 1 GET /health received
- `test_collection_list` — asserts GET /collections returns without error
- `test_collection_create` — asserts POST /collections 201
- `test_ingest_jsonl` — 3-record JSONL, asserts 1 POST, body has 3 records
- `test_ingest_csv` — 2-row CSV, asserts 1 POST, body has 2 records
- `test_ingest_dry_run` — asserts received_requests is empty
- `test_search_dense` — asserts POST /collections/test/search/dense called
- `test_client_401_returns_clear_error` — asserts error contains "api-key" or "unauthorized"

---

## 13. What Prompt 11 Implemented

**Prompt 11: Benchmark Harness** — `vecdb-bench` rewritten as a pure HTTP client benchmark binary. No vecdb-core dependency.

### Architecture
- **`client.rs`:** `BenchClient` (same pattern as vecdb-cli's `VecDbClient`). `post` and `delete_no_body` methods. X-Api-Key header, 30s timeout.
- **`types.rs`:** Mirror structs with `#![allow(dead_code)]` (response fields not all consumed). `UpsertRequest`, `DenseSearchRequest`, `SearchResponse`, `SearchResult`, `CreateCollectionRequest`.
- **`datasets.rs`:** `generate_synthetic(n, dim, seed)` — seeded `StdRng::seed_from_u64`, `Uniform(-1.0, 1.0)`, L2-normalize. `load_jsonl(path)` — BufReader line-by-line, skips blanks, returns error with line number on parse failure.
- **`runner.rs`:** `DatasetKind { Synthetic, Jsonl }`, `SearchType { Dense, Sparse, Hybrid }` (Sparse/Hybrid `#[allow(dead_code)]`), `BenchmarkConfig`, `BenchmarkResult`, `BenchmarkRunner::run`. Free functions: `compute_recall(gt, server, k) -> f64` and `percentile(sorted, pct) -> f64`.
- **`main.rs`:** clap CLI with 13 flags. `print_table` for markdown table output; `--output json` for pretty JSON.
- **`tests.rs`:** 8 pure-compute tests (no network).

### Key implementation details
- `rand = "0.8"` (latest is 0.10 but spec pins 0.8 for stable API)
- `indicatif = "0.17"` (progress bars during ingest and benchmark phases)
- Brute-force ground truth uses `rayon::par_iter` for parallel cosine similarity
- `percentile(sorted, pct)` uses index `(len-1)*pct/100` — verified against test: 100 values 1..=100, p50=50 p95=95 p99=99
- Warm-up: `(query_count / 10).max(1)` queries run silently before timed loop
- `#![allow(dead_code)]` in types.rs — mirror structs exist for deserialization; not all fields consumed

### 8 bench unit tests (pure compute, no server)
- `test_synthetic_generates_correct_count` — 100 vecs of dim 32
- `test_synthetic_vectors_are_normalized` — magnitude within 1e-5 of 1.0
- `test_synthetic_is_reproducible` — same seed → identical vectors
- `test_synthetic_ids_are_correct` — "syn-0" through "syn-9"
- `test_load_jsonl_parses_correctly` — 3-record file, correct ids and vector values
- `test_load_jsonl_skips_blank_lines` — blank lines between records, still 3 records
- `test_recall_computation` — gt=["a","b","c"] server=["a","b","d"] → 2/3
- `test_percentile_computation` — 1..=100 sorted, p50=50 p95=95 p99=99

---

## 14. Known Issues and Future Work

### Known issues
- **No WAL compaction:** WAL grows unboundedly. `checkpoint()` exists but is not called automatically. Compaction is future work (Prompt 14).
- **Mmap never compacted:** Deleted vectors remain in the flat file forever. Disk usage only grows. Future work (Prompt 14).
- **SQL OR clauses silently ignored in filter construction:** AstConverter only extracts AND-chain equality conditions into `filter_predicate`. OR-branched scalar conditions are not used.
- **INDEX_SIZE_VECTORS not recorded:** The constant exists in `metrics.rs` but no handler records it. Future work — will be recorded in the health handler or a background task.
- **`collect_equality_conditions` in AstConverter:** Only `Operator::Eq` scalar conditions go into the JSON filter. `!=`, `<`, `LIKE` etc. are silently ignored.

### Implementation deviations from original prompt plan
The original prompt plan listed Prompt 9 as "Multi-collection support" and Prompt 11 as "Observability". The actual implementation diverged:
- **P8** (as planned): HTTP API layer ✅
- **P9** (actual): Prometheus metrics + observability (was originally Prompt 11)
- **P10** (actual): CLI as pure HTTP client binary — no vecdb-core dep (was originally Prompt 16)
- **P11** (actual): Benchmark harness — vecdb-bench pure HTTP client; recall@k; p50/p95/p99; rayon brute-force GT
- **P12** (next): Multi-collection support
- Multi-collection support, compaction, etc. are deferred and renumbered

The prompt numbering in this file reflects what was actually built, not the original plan.

---

## 15. Remaining Prompt Titles in Order (12 through 25)

| # | Title | What it covers |
|---|-------|----------------|
| 10 | CLI Implementation | pure HTTP client binary; 5 subcommands; JSONL+CSV ingest; 8 wiremock tests | ✅ COMPLETE |
| 11 | Benchmark Harness | vecdb-bench pure HTTP client; synthetic+JSONL datasets; recall@k; p50/p95/p99; QPS; 8 bench tests | ✅ COMPLETE |
| 12 | Multi-collection support | Collection registry in AppState; dynamic create/open; `Arc<RwLock<HashMap<String, Mutex<Storage>>>>` or similar; concurrent access |
| 13 | Compaction | WAL truncation policy; mmap compaction (rewrite without deleted vectors); SQLite vacuum; background compaction task |
| 13 | Metadata payload filtering | Pre-filter via SQLite WHERE before vector scan; `should_pre_filter` logic actually wired to MetadataStore |
| 14 | IVF flat index | Pure Rust IVF as alternative IndexBackend; cluster centroids; assign/search; `IndexType::IVF` actually works |
| 15 | Cost-based index selection | Planner picks HNSW vs IVF based on collection size; auto-build IVF on large insert batches |
| 16 | Benchmark harness | Full MS MARCO and synthetic dataset support; recall@10; latency p50/p95/p99; QPS; JSON output |
| 17 | Authentication enhancements | Per-collection ACL; environment variable config; bearer token option |
| 18 | gRPC transport | tonic server; protobuf Search, Upsert, Delete service definitions; streaming upsert; dual HTTP+gRPC |
| 19 | Streaming ingestion | Chunked async batch writer; backpressure; `POST /collections/:name/vectors/stream`; internal ring buffer |
| 20 | Vector quantization | Scalar quantization (SQ8); product quantization (PQ); compressed search path |
| 21 | Distributed sharding | Shard registry; consistent hashing for collection→node mapping; shard routing layer |
| 22 | Python client SDK | PyO3 bindings or generated reqwest-based HTTP client; publish to PyPI |
| 23 | Integration + load tests | End-to-end multi-collection; concurrent writers; 10k vector recall test; latency under load |
| 24 | v0.1.0 release | Dockerfile; `cargo publish`; README polished with quick-start; CHANGELOG.md; GitHub release |
| 25 | (reserved / overflow) | Buffer for anything that overflows from earlier prompts |
