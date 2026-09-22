# CURRENT_STATE.md

Baseline audit of `vecdb` as it exists today, before the v2 "embeddable / SQLite of vector search" pivot. Written by reading the source, not the README. Living document — updated as v2 progresses.

_Audited: 2026-09-15. Git HEAD: `0511c4a` (fix: use MSYS2 perl for vendored OpenSSL on Windows)._

> **Progress log**
> - **Phase 0 — Scope & baseline:** ✅ gate PASS. This document.
> - **Phase 3 — Filtered search:** ✅ gate PASS (2026-09-15). Real predicate pushdown: `Collection::query_filtered` / `Storage::search_dense_filtered` generate candidates **from the filter** (exact full-precision scan of the matching set when selective) instead of ANN-then-discard. The metadata predicate is pushed into SQLite via `json_extract` (shared eval logic → identical semantics, superset SQL + in-process re-eval), and a declared **payload index** (`CollectionConfig::with_indexed_fields`, SQLite expression index) turns it into an index lookup. Executor's SQL dense pre-filter path rerouted through it. SIFT10K, 95%-eliminating filter (`examples/filter_bench.rs`): pushdown **recall@10 1.00** vs naive post-filter **0.26 at any budget** (ANN only returns its bounded neighborhood); with the index, filtered latency is **1.75× unfiltered** (same ballpark) vs **6.8× slower** without it. Honest writeup in `benchmarks/results/filtered_search.md`. Tests: 134 lib + embedded/quant/filtered integration (filtered: 3), clippy `--all-targets` clean.
> - **Phase 2 — Quantization:** ✅ gate PASS (2026-09-15). Added opt-in int8 scalar quantization per collection (`Quantization` on `CollectionConfig`; `Db::create_collection_quantized`). New `ScalarQuantizedIndex` (`src/index/scalar.rs`) — per-dim asymmetric int8, flat scan via a precomputed-norm `f32·i8` dot kernel (exact-score parity with the float path, unit-tested). Wired as a third `AnyIndex` variant with its own `.sq.json` persistence; default float32 path unchanged. Measured on **SIFT10K** (real, `examples/quant_bench.rs`): **3.88× smaller in-memory index, 1.0% recall@10 loss**; latency is **~1.8× slower** on this cache-resident set (compute-bound; int8's bandwidth win is a larger-N effect, deferred to Phase 4) — reported honestly in `benchmarks/results/quantization.md`. **Binary quantization (Prompt 6) also done:** `BinaryQuantizedIndex` (`src/index/binary.rs`), per-dim mean-threshold 1-bit + popcount Hamming, `.bq.json` persistence, `Db::create_collection_binary`; SIFT10K = **31.9× smaller, recall@10 0.33 raw → 0.76 with top-100 f32 rerank**. Tests: 134 lib + 2 embedded + 3 quantized-mode + 1 doctest, clippy `--all-targets` clean.
> - **Phase 1 — Embedded core:** ✅ gate PASS (2026-09-15). Added ergonomic `vecdb_core::{Db, Collection}` (`src/db.rs`) wrapping `Storage`; dropped the top-level `ServerConfig` re-export from the embedded surface (moved to `vecdb_core::config::ServerConfig`, api repointed). Integration test `tests/embedded_roundtrip.rs` proves insert→query→persist→reopen (recall@1 perfect, delete persists). A fresh external `cargo` project using `vecdb-core` as a path dep runs the full loop in 17 LOC with no server (verified). Windows note: build with the **msvc** toolchain — the gnu default lacks `dlltool`; no toolchain file pinned to avoid breaking Linux/musl CI. Server crate (`vecdb-api`) vendored-OpenSSL build needs perl+make locally (CI has them; secondary to embedded story).

---

## 1. Headline finding

**The embedded core already exists.** `vecdb-core` is a pure library crate with **zero network/server dependencies**. The full insert → search → persist → reopen loop runs in-process via `vecdb_core::storage::Storage`. The axum HTTP server (`vecdb-api`) is already a thin wrapper on top of it (`CollectionManager` holds `Arc<Mutex<Storage>>` per collection).

So v2 is **mostly a repositioning + ergonomics + measurement effort, not a rewrite.** The "no server required" claim is already technically true; it is just not the story the code presents itself as, and there is no clean top-level embedded API surface.

---

## 2. Architecture (as built)

Cargo workspace, 4 crates:

| Crate | Role | Net deps? |
|-------|------|-----------|
| `vecdb-core` | storage, index (HNSW+IVF), sparse BM25, hybrid fusion, SQL planner, types | **none** |
| `vecdb-api` | axum 0.7 HTTP server, routes, middleware, metrics, collection manager | tokio, axum, tower |
| `vecdb-cli` | HTTP client CLI (talks to the server over reqwest) | reqwest |
| `vecdb-bench` | benchmark harness (drives the server over HTTP) | reqwest |

Core internal layers: `storage/` (mmap + WAL + SQLite metadata) → `index/` (HNSW via `instant-distance`, pure-Rust IVF, `IndexBackend` trait) → `sparse/` (custom inverted index + tokenizer, BM25) → `hybrid/` (weighted-sum / RRF fusion) → `planner/` (custom SQL lexer/parser/AST → logical → physical plan, cost model).

**Coupling note:** `vecdb-cli` and `vecdb-bench` are HTTP clients — they require a running server. There is no in-process CLI or in-process benchmark path today. That is a v2 gap (embedded CLI, embedded bench).

---

## 3. API surface

### Embedded (core) — already usable, but raw
`vecdb_core::storage::Storage`:
- `Storage::create(data_dir: &Path, config: &CollectionConfig) -> Result<Storage>`
- `Storage::open(data_dir: &Path, collection_name: &str) -> Result<Storage>`
- `upsert(record: VectorRecord) -> Result<UpsertResult>`
- `delete(id: &VectorId) -> Result<()>`
- `search(query, k)`, `search_dense(query, k)`, `search_sparse(text, k)`, `search_hybrid(vec?, text?, k, alpha, strategy)`
- `execute_search(&SearchRequest)`, `execute_sql(&str)`
- `save_indexes()`, `checkpoint()`, `recover_from_wal()`, `rebuild_index()`, `stats()`

**Ergonomics gaps for the "SQLite of vector search" pitch:**
- No single top-level handle. Caller works with `Storage` (module-path type), one instance per collection, addressed by `(data_dir, name)`. There is no `Db` / `Collection` two-level API and no `.vecdb` file/dir concept formalized.
- `Storage` has many `pub` fields (`vectors`, `wal`, `metadata`, `index`, …) — internals are exposed, not encapsulated.
- `lib.rs` re-exports `config::ServerConfig` at the top level — server-flavored naming leaking into the embedded core.
- `VectorRecord` requires caller to set `created_at`/`updated_at` manually (no builder / defaulting).

### Server (HTTP)
REST routes under `vecdb-api`: health, collections CRUD, vectors upsert/delete, search (dense/sparse/hybrid/sql), Prometheus `/metrics`. Optional `X-Api-Key`. This stays as an optional wrapper in v2.

---

## 4. On-disk format

Per-collection, multiple files in a directory (NOT a single `.vecdb` file yet), named `<collection>.*`:

| File | Content |
|------|---------|
| `<c>.vectors` | mmap float32 vector store |
| `<c>.wal` | write-ahead log |
| `<c>.db` | SQLite (rusqlite, bundled) — metadata, payload JSON, id→mmap_index map |
| `<c>.sparse.json` | BM25 inverted index (JSON) |
| `<c>.hnsw.json` / `<c>.ivf.json` | persisted dense index |

**Format details (verified in source):**
- Vector store: 64-byte header, `MAGIC = 0x56454344` ("VECD"), `VERSION = 1`, then dimension/count/capacity/timestamps; float32 payload little-endian. **Versioned header already present.**
- WAL: length-prefixed entries, each with an xxh3 checksum. Entry variants `Insert{id,mmap_index,record}`, `Delete{id}`, `Checkpoint{count,ts}`. Replay on `open`; `checkpoint()` truncates.
- Crash-safety: WAL-first on upsert, replay-on-open reconciles metadata, `test_wal_recovery` proves drop-without-checkpoint survives. This is **real WAL, not aspirational.**

**v2 format gaps:**
- Not a single file — "point vecdb at a file" is currently "point it at a directory of 5+ files per collection."
- Metadata layer is SQLite. Fine for embedded (bundled, no server), but adds `rusqlite` + `r2d2` weight and is a design decision to consciously keep or drop.
- Vectors are float32 only — **no quantization** (Phase 2 target).

---

## 5. Filtered search — the Phase 3 gap is real

`execute()` in `planner/executor.rs` has a `pre_filter` path, but it is **not** true predicate pushdown:

1. Stage 0 builds an allowed-id `HashSet` from SQLite (`metadata.filter_ids`).
2. Stage 1 runs `search_dense(qv, candidate_k)` — a plain HNSW top-k that is **filter-blind**.
3. Stage 2a does `results.retain(|r| ids.contains(&r.id))` — **drops non-matching hits after the fact.**

This was exactly the "search then throw away results that don't match" anti-pattern the v2 plan calls out. At high filter selectivity (95%+ eliminated), HNSW returns `candidate_k` hits and most get discarded → recall collapses.

**Resolved in Phase 3.** `Storage::search_dense_filtered` now generates candidates from the filter (exact scan of the matching set when selective), with the predicate pushed into SQLite (`json_extract` + optional payload index). Measured: pushdown recall@10 1.00 vs naive post-filter 0.26 at any budget. See `benchmarks/results/filtered_search.md`.

(README claims "filter pushdown" — technically it pushes the *predicate evaluation* to the metadata layer for the allowed-id set, but does not push it into the ANN traversal. The claim is misleading for the case that matters.)

---

## 6. Test coverage

**171 test functions total** across the workspace (real, not stubs):
- `vecdb-core`: ~119 (storage 17+3, sparse 18, index 32, planner/SQL 49) — strong.
- `vecdb-api`: 36 route tests.
- `vecdb-cli`: 8, `vecdb-bench`: 8.

Core storage round-trip, WAL recovery, hybrid paths, SQL parsing all covered. No integration test yet that proves the *20-lines-from-a-fresh-crate* embedded story (Phase 1 Prompt 4 target). No quantization tests (feature absent). Filtered-search tests exist (`filter_tests.rs`, 10) but assert correctness, not that pushdown avoids the latency cost.

---

## 7. README claims vs actual behavior

README is **server-first** and oversells:

| README says | Reality |
|-------------|---------|
| "production-grade vector database", "self-hosted" | Embedded-capable library + optional server. Positioning is backwards vs v2. |
| Quickstart is `docker run` first | Embedded (add crate, open dir) is not shown at all. |
| "single statically-linked binary (~10 MB)" | Unverified here — server binary size not measured this audit. |
| "fast enough to serve thousands of queries per second on a laptop" | **No reproducible benchmark backs this.** `benchmarks/results/` contains `synthetic_results.md` + `placeholder.md`. Must not be repeated in v2 assets until measured. |
| "Filter pushdown — all comparison operators" | Predicate eval pushed to metadata; NOT pushed into ANN traversal (see §5). |
| Prometheus, connection pooling, graceful shutdown, SDKs | Present (server-side features). Real, but secondary to the embedded pitch. |

**crates.io:** not published. No `publish = false` flags, no crates.io badge; `vecdb-core` version `0.1.0`. Publishing is Phase 5 work.

---

## 8. v2 gap summary (what each phase actually has to do)

- **Phase 1 (embedded core):** Core already net-free ✅. Real work = clean `Db`/`Collection` ergonomic API, `.vecdb` single-file-or-dir concept, encapsulate `Storage` internals, drop `ServerConfig` from the embedded surface, add the 20-line integration test. Make `vecdb-cli`/`vecdb-bench` able to run in-process (not HTTP-only).
- **Phase 2 (quantization):** Net-new. int8 scalar quantization (opt-in per collection), real recall/memory/latency numbers. Binary quant stretch.
- **Phase 3 (filtered search):** Net-new. Fuse the allowed-id set into HNSW traversal, prove latency stays near unfiltered at high selectivity.
- **Phase 4 (benchmarks):** Harness exists but is HTTP-driven. Need real public dataset (SIFT1M subset), vecdb embedded vs qdrant/chroma/pgvector, honest "where vecdb loses". Delete synthetic/placeholder results.
- **Phase 5 (packaging):** Publish `vecdb-core` to crates.io, minimal Docker for server mode, README rewrite embedded-first, repo metadata.
- **Phase 6 (launch):** DEV.to writeup, Show HN / r/rust posts — every claim traceable to this file, BENCHMARK.md, or code.

---

## Phase 0 quality gate

**PASS** — this document reflects behavior confirmed by reading the actual source (`storage/mod.rs`, `storage/mmap.rs`, `storage/wal.rs`, `planner/executor.rs`, `collection_manager.rs`, both Cargo manifests), not the README. Two README claims were found to be unbacked/misleading (QPS number, filter pushdown) and are flagged above. No code changed.
