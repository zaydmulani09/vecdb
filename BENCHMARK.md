# vecdb Benchmark — vs qdrant, chroma, pgvector

vecdb runs **embedded, in-process, with no container**. The three comparison
systems run as real servers. This document reports how they compare on the same
dataset, the same queries, and the same recall math.

**Every number here comes from an actual run of the committed harness
(`crates/vecdb-compare`) and is reproducible with the commands at the bottom.
Nothing is estimated, and the rows where vecdb loses are left exactly as
measured.**

_Status: methodology finalized; 100k sanity + 1M publish numbers being filled
from the live runs._

## Systems & exact versions

| System | Version | How it runs here |
|--------|---------|------------------|
| **vecdb** | this repo @ branch `v2` | Embedded Rust library, in-process, no server |
| **qdrant** | v1.19.1 (native `qdrant-x86_64-pc-windows-msvc`) | Local server, REST :6333 |
| **chroma** | chromadb 0.5.20 (pip, Python 3.11) | Local server, REST :8000 |
| **pgvector** | pgvector on Neon managed Postgres (PostgreSQL 17) | Managed cloud, TLS |

_pgvector runs on managed cloud because there is no official pgvector Windows
binary and this machine has no local Postgres build toolchain. See the fairness
notes below for exactly how that is accounted for._

## Hardware & build

- **CPU:** Intel Core i7-1355U (12 logical cores)
- **RAM:** 11.7 GB
- **OS:** Windows 11
- **Toolchain:** rustc 1.95.0, `x86_64-pc-windows-msvc`, `--release`, default
  `target-cpu` (no AVX2 flag)

## Dataset

- **SIFT** (TEXMEX corpus), 128-dim, Euclidean (L2). Standard ANN benchmark set.
  - 100k pass: first 100,000 base vectors, 1,000 queries, exact ground truth
    recomputed on the subset.
  - 1M pass: full 1,000,000 base vectors, 1,000 queries, dataset ground truth.
  - Fetch: `curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`

## Metrics & method

- **recall@10** — mean over queries of |returned top-10 ∩ true top-10| / 10,
  against exact L2 ground truth. Identical computation for every system.
- **build s** — wall-clock to ingest all vectors and build the index, ready for
  indexed search.
- **QPS / p50 / p99** — single-client, one query at a time (not concurrent), k=10.
- **mem MB** — resident memory of the server process(es) after build
  (approximate; see notes).
- **disk MB** — on-disk size of the system's data directory after build.

### Index configuration (stated, not hidden)

| System | Build params | Query param |
|--------|--------------|-------------|
| vecdb | HNSW, ef_construction **200** (primary), M=32 (instant-distance) | ef_search 50 |
| qdrant | HNSW defaults (M=16, ef_construct 100), `indexing_threshold=1000` so every segment indexes | default |
| chroma | HNSW defaults (M=16, ef_construct 100) | `search_ef=100` |
| pgvector | HNSW defaults (m=16, ef_construction=64) | ef_search 40 |

### Fairness notes (read these before the tables)

- **pgvector is remote.** Its per-query latency is measured **server-side** via
  `EXPLAIN (ANALYZE)` execution time, which **excludes** the client↔cloud
  network round-trip — otherwise the internet RTT would unfairly dominate its
  engine latency. Its **build time includes network ingest** (COPY over the
  internet) and is therefore *not* comparable to the local systems' build times;
  it is marked accordingly. Its server RSS and disk are not readable from a
  managed instance, so those cells are blank.
- **qdrant / chroma latency includes local HTTP** (loopback REST), a small fixed
  per-query overhead the embedded vecdb path does not pay. This is inherent to
  comparing an embedded library against servers and is noted, not corrected.
- **Memory bases differ**: vecdb = the harness process RSS delta across index
  build (index only, dataset excluded); qdrant/chroma = server process RSS
  (includes their own runtime). Treat memory as order-of-magnitude.
- Single-run laptop timings vary run-to-run; recall is deterministic.

## Results — SIFT 100k

1,000 queries, k=10.

```
system      recall@10  build s   qps    p50 µs   p99 µs   mem MB  disk MB
vecdb        0.9966     477.8     254    3845     7027     153     143
qdrant       0.9996      17.9     130    7516    12093     358    2189
chroma       0.9943     203.9      31   30951    49153     395      —
pgvector     0.9727      68.3    1160     889     1174       —       —
```

- pgvector build (68.3s) **includes network ingest** to Neon; its latency is
  **server-side** (EXPLAIN ANALYZE), so QPS/p50/p99 are its engine, RTT excluded.
- chroma disk not captured here (—); qdrant/pgvector memory or disk (—) as noted.
- vecdb's per-query path has **no HTTP**; qdrant/chroma latency includes loopback
  REST. Read the latency columns with that asymmetry in mind.

## Results — SIFT 1M (primary, ef_construction=200)

_(filled from the 1M run)_

```
system      recall@10  build s   qps    p50 µs   p99 µs   mem MB  disk MB
```

## Where vecdb loses

_(written from the measured numbers — not softened)_

## Reproduce

```bash
# dataset
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar xzf sift.tar.gz

# competitors (native, no docker)
#  qdrant:   run qdrant.exe (v1.19.1) from a short path -> :6333
#  chroma:   py -3.11 -m pip install chromadb==0.5.20; chroma run --port 8000 --path <dir>
#  pgvector: a Neon (or any managed) Postgres URL in PGVECTOR_CONN

QDRANT_DATA=<qdrant storage dir> CHROMA_DATA=<chroma data dir> \
PGVECTOR_CONN='postgresql://.../db?sslmode=require' \
cargo run --release -p vecdb-compare -- <sift dir> --n 1000000 --queries 1000 \
  --systems vecdb,qdrant,chroma,pgvector
```
