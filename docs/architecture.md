# vecdb Architecture

## Overview

vecdb is a self-hosted hybrid vector database. Every search request goes through two indexes in parallel — a dense HNSW or IVF index for semantic similarity and a BM25 inverted index for keyword relevance — and the results are fused by the hybrid engine before being returned to the client.

The data path is: HTTP request → auth middleware → route handler → query planner → storage (executes plan) → index + sparse → hybrid engine → response. Mutations follow write-ahead logging: every insert or delete is written to the WAL before it touches SQLite or the memory-mapped vector file, guaranteeing crash recovery.

---

## Component Map

### `vecdb-core`

All business logic. Nothing in this crate knows about HTTP.

| Module | Responsibility |
|---|---|
| `storage/` | `Storage` struct wiring MmapVectorStore + WAL + MetadataStore + indexes |
| `index/` | `IndexBackend` trait, `HnswIndex`, `IvfIndex`, `AnyIndex` enum dispatch |
| `sparse/` | `Tokenizer`, `InvertedIndex` (BM25), `SparseIndex` facade |
| `hybrid/` | `HybridEngine`, `FusionStrategy`, score normalization |
| `planner/` | `QueryPlanner`, `LogicalNode`, `PhysicalPlan`, `PlanExecutor` |
| `planner/sql/` | Lexer, recursive-descent parser, AST, `AstConverter` |
| `types.rs` | Core types: `VectorRecord`, `SearchResult`, `CollectionConfig`, etc. |
| `errors.rs` | `VecDbError` enum with `thiserror` |
| `config.rs` | `ServerConfig` with `config`-crate loader and env var support |

### `vecdb-api`

The axum HTTP server. Has both a `[lib]` target (for integration tests) and a `[[bin]]` target.

| Module | Responsibility |
|---|---|
| `server.rs` | `run(config)`, TCP bind, middleware stack, graceful shutdown |
| `routes/` | One file per resource group: health, collections, vectors, search |
| `collection_manager.rs` | `CollectionManager` — registry of open collections |
| `state.rs` | `AppState` (`RwLock<CollectionManager>` + metrics handle + api_key) |
| `middleware.rs` | `auth_middleware` — reads `X-Api-Key`, returns 401 on mismatch |
| `error.rs` | `ApiError(VecDbError)` → HTTP status + JSON error body |
| `types.rs` | HTTP request/response structs (separate from core types) |
| `metrics.rs` | `install_recorder()`, metric name constants |

### `vecdb-cli`

A pure HTTP client binary that calls the vecdb API. Has no dependency on `vecdb-core` — uses its own mirror types so the CLI binary stays small.

Subcommands: `ping`, `collection` (list/get/create/delete), `ingest` (JSONL + CSV), `inspect`, `search` (dense/sparse/hybrid/sql).

### `vecdb-bench`

Benchmark harness binary. Generates synthetic float32 datasets or loads JSONL files, ingests vectors, runs timed search queries, computes recall@k against brute-force ground truth, and reports p50/p95/p99 latency percentiles and QPS.

---

## Storage Layer

### MmapVectorStore

Vectors are stored in a flat binary file as packed little-endian `f32` arrays. A 64-byte header precedes the data:

```
bytes  0.. 4  magic u32 LE: 0x56454344 ("VECD")
bytes  4.. 8  version u32 LE: 1
bytes  8..16  dimension u64 LE
bytes 16..24  count u64 LE
bytes 24..32  capacity u64 LE
bytes 32..40  created_at unix timestamp u64 LE
bytes 40..48  updated_at unix timestamp u64 LE
bytes 48..64  reserved (zeroed)
```

Data starts at byte 64. Each vector is `dimension * 4` bytes. Random access is O(1) via `memmap2`. When the file is full, capacity doubles: the existing `MmapMut` is swapped for a 1-byte anonymous map, the file is resized, then remapped — this is required on Windows where a mapped file cannot be resized while the mapping is open.

### WriteAheadLog

Every mutation is framed and appended to the WAL before touching any other store:

```
[4-byte LE u32 length][N-byte serde_json payload][4-byte LE u32 xxh3-32 checksum]
```

Entry types: `Insert { id, mmap_index, record }`, `Delete { id }`, `Checkpoint { entry_count, timestamp }`.

The file is opened with `read + write + create` (not `O_APPEND`). Before each write, `seek(End(0))` positions the cursor; after the write, `sync_all()` flushes to disk. Replay reads the entire file via `std::fs::read` into a `Cursor` for checksum verification. A corrupt checksum stops replay at that point and treats later entries as lost.

### MetadataStore

SQLite database managed through an r2d2 connection pool (max 8 connections, configurable). Opened in WAL journal mode with `busy_timeout=5000` and `synchronous=NORMAL`.

Schema:
```sql
CREATE TABLE vectors (
    id TEXT PRIMARY KEY,
    mmap_index INTEGER,
    payload TEXT,       -- JSON
    text TEXT,
    created_at INTEGER,
    updated_at INTEGER,
    deleted INTEGER DEFAULT 0
);
CREATE TABLE collections (
    name TEXT PRIMARY KEY,
    dimension INTEGER,
    metric TEXT,
    index_type TEXT,
    created_at INTEGER
);
```

Soft-delete: `deleted=1` flag. Active records use `WHERE deleted = 0`. `filter_ids(filter)` loads all active IDs and applies `apply_json_filter` in Rust — used for pre-filter optimization when the filter is highly selective.

### Write Order

```
Client upsert
    → WAL.append(Insert { id, ... })   ← crash-safe point
    → MetadataStore.upsert(id, ...)
    → MmapVectorStore.append(vector)
    → HnswIndex.insert(id, vector)
    → SparseIndex.insert(id, text)
```

On crash after WAL write but before the rest: replay re-runs the insert on startup.

---

## Index Layer

### HnswIndex

Wraps `instant-distance` 0.6. The graph degree M is hardcoded at 32 by the library (the `HnswConfig.m` field is API surface only). Search uses brute force when `n ≤ 100` or `dirty = true`; rebuilds automatically every 1000 inserts.

Deletes are soft: vectors are removed from `deleted: HashSet<VectorId>` at query time and compacted on the next rebuild. When `deleted.len() > n/4`, rebuild is triggered immediately from `delete()`.

Saved to `{collection}.hnsw.json` via serde_json.

### IvfIndex

Pure-Rust k-means IVF. Parameters: `n_lists=256`, `n_probe=16`, `max_iter=25`. Deterministic initialization spaces initial centroids at `i * (n / n_lists)`. Dead centroids (empty lists after training) are reinitialized to `vectors[ci * 7 % n]`. Convergence threshold: 1e-6.

When `n < n_lists`, clamps `n_lists` to `n.max(1)` to avoid division by zero. New inserts go to `lists[0]` until trained; first insert triggers training; every 1000 subsequent inserts trigger a rebuild.

Saved to `{collection}.ivf.json` via serde_json.

### AnyIndex

`AnyIndex` is an enum in `index/mod.rs` that wraps both backends and implements `IndexBackend` by delegation:

```rust
pub enum AnyIndex {
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
}
```

`Storage.index` is always `AnyIndex`. The backend is chosen at collection creation time from `CollectionConfig.index_type` and persisted in SQLite.

---

## Sparse Index

### Tokenizer

Splits on non-alphanumeric characters, lowercases, filters length 2–64, removes 100 hardcoded English stopwords. Two methods with different stopword behavior:

- `tokenize(text)` — removes stopwords. Used when **indexing documents**.
- `tokenize_query(text)` — keeps stopwords. Used when **searching**. If you call `tokenize` on a query like "to be or not", all tokens are stopwords and the query returns zero results.

### InvertedIndex

BM25 scoring with `k1=1.5`, `b=0.75`:

```
IDF = ln((N - df + 0.5) / (df + 0.5) + 1)
score(d,q) = Σ IDF(t) * (tf * (k1+1)) / (tf + k1*(1 - b + b*(|d|/avgdl)))
```

Posting lists are sorted by `doc_id` for O(log n) binary search. Hard delete removes entries from posting lists (unlike the HNSW soft-delete approach). Serialized to `{collection}.sparse.json` via serde_json (NOT bincode — `serde_json::Value` in payloads is incompatible with bincode).

---

## Hybrid Engine

### Two-Stage Pipeline

```
Dense HNSW search (k × 5 candidates)
        ↓
Sparse BM25 re-score of those same candidates
        ↓
HybridEngine::fuse()
        ↓
Top-k results with dense_score, sparse_score, score
```

The oversample factor of 5× ensures the sparse re-scorer sees enough candidates that the final fusion is meaningful. When only dense or only sparse is requested, the fuse path is skipped and the score is passed through directly.

### Fusion Strategies

**WeightedSum** (default):
```
normalized_dense  = min_max(dense_scores)
normalized_sparse = min_max(sparse_scores)
score = alpha * normalized_dense + (1 - alpha) * normalized_sparse
```

All-equal scores normalize to 1.0 (not NaN) to avoid degenerate outputs.

**ReciprocalRankFusion**:
```
score(d) = Σ 1 / (k + rank(d, list_i))   where k=60, ranks are 1-indexed
```

---

## Query Planner

### PhysicalPlan Fields

| Field | Type | Meaning |
|---|---|---|
| `index_type` | `IndexType` | HNSW or IVF |
| `use_hybrid` | `bool` | Activate sparse re-score |
| `pre_filter` | `bool` | Filter in SQLite before index scan |
| `filter_predicate` | `Option<String>` | JSON filter expression |
| `output_columns` | `Vec<String>` | SELECT column list (empty = `*`) |
| `candidate_k` | `usize` | Oversample factor for ANN |
| `output_k` | `usize` | Final result count |
| `estimated_cost` | `f32` | Cost model output |

### Cost Model

```
index_cost = ln(N) * k * 0.1
```

Pre-filter is activated when selectivity < 0.1 or N > 100,000. `candidate_k` multipliers: 1× (no filter), 3× (post-filter), 5× (hybrid), 10× (both), capped at 10,000.

### PlanExecutor Pipeline

1. Scan — call `search_dense` or `search_hybrid` for candidate set
2. Pre-filter — intersect candidates with `MetadataStore::filter_ids` result (when `pre_filter=true`)
3. Post-filter — apply `apply_json_filter` in Rust (all operators including LIKE with `%` wildcard)
4. Sort — descending by score
5. Truncate — to `output_k`
6. Project — apply `apply_projection` to keep only `output_columns` in payload

---

## SQL Parser

Hand-written lexer + recursive-descent parser in `crates/vecdb-core/src/planner/sql/`. No external parser crates — `sqlparser-rs` was rejected because VECTOR_SIM required too many workarounds.

**Grammar (informal):**
```
SELECT <columns>
FROM <table>
[WHERE <condition>]
[ORDER BY score DESC]
[LIMIT <n>]

condition := vector_condition [AND scalar_condition]*

vector_condition := VECTOR_SIM(field, [f32, ...]) <op> <threshold>

scalar_condition := <field_path> <op> <literal>
field_path := identifier | identifier->>'key' | identifier->'nested'->>'key'
```

`AstConverter` walks the parsed `SelectStatement` and emits a `PhysicalPlan`. Scalar conditions are converted to structured filter JSON `{"field":"...","op":"...","value":...}` that `apply_json_filter` understands.

---

## HTTP API Layer

### CollectionManager

`CollectionManager` (in `vecdb-api`) holds `HashMap<String, Arc<Mutex<Storage>>>`. Route handlers:

1. Read-lock the manager → clone `Arc<Mutex<Storage>>` → release read-lock
2. Acquire the storage mutex
3. Call business logic
4. Release storage mutex

This means concurrent requests to **different** collections never block each other. Concurrent requests to the **same** collection serialize at the `Mutex` level (required because `rusqlite::Connection: !Sync`).

### Auth Middleware

`auth_middleware` reads the `X-Api-Key` request header. If `state.api_key` is `Some(key)` and the header is missing or mismatched, it returns HTTP 401 before the handler runs. If `api_key` is `None`, all requests pass through.

### Metrics

`RequestTimer::start(op, collection)` increments `vecdb_requests_total` counter and starts a timer. When the returned `_timer` is dropped (end of handler scope), the `Drop` impl records `vecdb_request_duration_ms` histogram. Search handlers additionally record `vecdb_search_duration_ms` and `vecdb_search_results_count`.

### Graceful Shutdown

`shutdown_signal()` waits for Ctrl+C or SIGTERM (via `tokio::signal`). After `axum::serve` drains in-flight requests:

1. Write-lock `CollectionManager`
2. For each open collection: `storage.checkpoint()` then `storage.save_indexes()`
3. Log "vecdb shutdown complete"

---

## Data Flow: Hybrid Search Request

```
POST /collections/my-col/search/hybrid
  { "vector": [...], "query": "machine learning", "k": 10, "alpha": 0.7 }
          │
          ▼
  auth_middleware (check X-Api-Key)
          │
          ▼
  search_hybrid handler
    ├─ CollectionManager::get("my-col") → Arc<Mutex<Storage>>
    └─ Storage::search_hybrid(request)
            │
            ▼
       QueryPlanner::plan_search(request, n)
         → PhysicalPlan { use_hybrid: true, candidate_k: 50, output_k: 10 }
            │
            ▼
       PlanExecutor::execute(plan, storage)
         ├─ HnswIndex::search(vector, k=50) → Vec<(id, dense_score)>
         ├─ SparseIndex::score_candidates(query, ids) → Vec<(id, sparse_score)>
         └─ HybridEngine::fuse(dense, sparse, alpha=0.7)
              ├─ min_max_normalize(dense_scores)
              ├─ min_max_normalize(sparse_scores)
              └─ 0.7 * dense + 0.3 * sparse per candidate
            │
            ▼
       MetadataStore::get_batch(top_ids) → payloads + text
            │
            ▼
  SearchResponse { results: [...], count: 10, time_ms: 4 }
          │
          ▼
  JSON response to client
```
