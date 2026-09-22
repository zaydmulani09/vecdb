# vecdb

[![CI](https://github.com/zaydmulani09/vecdb/actions/workflows/ci.yml/badge.svg)](https://github.com/zaydmulani09/vecdb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![crates.io](https://img.shields.io/crates/v/vecdb-core.svg)](https://crates.io/crates/vecdb-core)

**An embeddable vector database in Rust. No server, no daemon, no docker — `cargo add`, point it at a file, query in-process.**

Think SQLite, for vector search. vecdb is a library you link into your app, not a service you run beside it. It stores float32 vectors + JSON metadata on disk and does dense (HNSW), sparse (BM25), hybrid, and filtered search — all embedded.

---

## Quickstart

```sh
cargo add vecdb-core serde_json
```

```rust
use vecdb_core::Db;
use serde_json::json;

fn main() -> Result<(), vecdb_core::VecDbError> {
    let db = Db::open("data.vecdb")?;              // a directory; created if absent
    let mut docs = db.open_or_create("docs", 4)?; // name, dimension

    docs.insert("a", vec![0.1, 0.2, 0.3, 0.4], json!({ "title": "hello" }))?;
    docs.insert("b", vec![0.9, 0.8, 0.7, 0.6], json!({ "title": "world" }))?;

    let hits = docs.query(&[0.1, 0.2, 0.3, 0.4], 5)?; // top-5 nearest
    println!("nearest: {}", hits[0].id);

    docs.flush()?; // durable via WAL already; flush persists the index for fast reopen
    Ok(())
}
```

That's the whole thing — no process to start. `cargo run` and you have vector search.

More: `query_text` (BM25), `query_hybrid` (dense+sparse), `query_filtered` (predicate pushdown on metadata), `create_collection_quantized` (int8, ~4× smaller index). See the [docs](https://docs.rs/vecdb-core).

---

## Scale expectations — read this before you load a million vectors

vecdb is built around HNSW, whose **index build is the slow part**. Query is fast; building the graph is not, and it runs single-node on your machine:

| Dataset size | Build time (default `ef_construction=200`) | Feel |
|---|---|---|
| up to ~100k | seconds to a few minutes | instant, use it freely |
| ~1M | **~2.5 hours** on a laptop | budget for it; build once, reuse |

If you're at 1M+ and don't need maximum recall, **set `ef_construction=100`** — ~31% faster build (~1.7h at 1M) for a negligible recall difference (0.9832 → 0.9803 on SIFT1M):

```rust
use vecdb_core::{Db, types::CollectionConfig};

let db = Db::open("data.vecdb")?;
let mut cfg = CollectionConfig::new("docs", 128);
cfg.hnsw_ef_construction = 100;
let docs = db.create_collection_with(cfg)?;
```

Query latency stays competitive with server databases at 1M (in-process, no HTTP hop). The honest full numbers vs qdrant / chroma / pgvector — including where vecdb loses — are in **[BENCHMARK.md](BENCHMARK.md)**. Bulk-loading? Use `Collection::insert_batch` — it builds the index once instead of on every insert.

---

## Features

- **Dense search** — HNSW (via instant-distance), cosine / euclidean / dot metrics, SIMD distance kernels
- **Sparse search** — built-in BM25 inverted index, no external search engine
- **Hybrid** — weighted-sum or reciprocal-rank fusion of dense + sparse
- **Filtered search** — predicate pushdown on JSON metadata; declare `indexed_payload_fields` for fast selective filters
- **Quantization** — opt-in int8 (~4× smaller index) or 1-bit binary (~32×) per collection
- **Durable** — write-ahead log, crash recovery, reopen persists
- **Embedded** — zero network dependencies in `vecdb-core`; SQLite (bundled) for metadata

---

## Server mode (optional)

Prefer a REST service? The `vecdb-api` crate wraps the embedded core in an axum HTTP server (Prometheus metrics, API key auth, Python/TypeScript SDKs). It's secondary to the embedded story — most users want the library. See [`crates/vecdb-api`](crates/vecdb-api).

---

## License

MIT — see [LICENSE](LICENSE).
