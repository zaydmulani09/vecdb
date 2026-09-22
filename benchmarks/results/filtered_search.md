# Filtered search — predicate pushdown (measured)

Phase 3 measurement of filtered dense search. **Every number comes from an
actual run of the committed harness (`crates/vecdb-core/examples/filter_bench.rs`)
and is reproducible.** Nothing is estimated.

## What "pushdown" means here

The old path searched the ANN index and then discarded results that didn't match
the filter ("search then throw away"). At high selectivity that is both slow
(you must oversample massively) and **wrong** — the ANN index returns only its
bounded neighborhood, which rarely contains the filtered nearest neighbors.

The new path generates candidates **from the filter**: `query_filtered` asks the
metadata store for the matching records and does an exact, full-precision scan
over just those vectors. The metadata predicate itself is pushed into SQLite via
`json_extract`, and a declared **payload index** turns that into an index lookup
instead of a full-payload scan.

## Methodology

- **Dataset:** SIFT10K base (10,000 × 128d) + 100 queries, Euclidean.
- **Filter:** each vector gets a `bucket` in `[0, 20)`; the filter keeps one
  bucket ≈ **500 vectors (5%)**, i.e. **95% eliminated**.
- **Recall@10:** vs an exact brute-force over the *matching* subset (the true
  filtered nearest neighbors).
- **Modes:**
  - `unfiltered` — plain ANN over the whole index (latency reference).
  - `pushdown+idx` — `query_filtered` with a payload index on `bucket`.
  - `pushdown(scan)` — `query_filtered` with **no** payload index (full
    `json_extract` scan of every row).
  - `naive@200` / `naive@allN` — ANN top-N then drop non-matches (the old
    approach), at budget 200 and at budget N.
- **Hardware:** Intel Core i7-1355U, Windows 11, rustc 1.95.0,
  `x86_64-pc-windows-msvc`, `--release`.

## Results

```
mode              recall@10      mean µs     p50 µs     p99 µs
unfiltered              —          1411.3       1396       2223
pushdown+idx         1.0000        2464.6       2329       3428
pushdown(scan)       1.0000       16812.4      16467      19924
naive@200            0.2590        2642.7       2561       4369
naive@allN           0.2590        2581.7       2578       3388
```

## Reading it

- **Correctness — pushdown is exact, post-filter is not.** `pushdown` returns
  recall@10 = **1.000**. Naive post-filter returns **0.259** and **does not
  improve with budget** (200 vs all-N are identical): the ANN index only ever
  yields its bounded neighborhood, so the filtered nearest neighbors it never
  visited can't be recovered by asking for more. This is the core reason
  pushdown is needed, not just a speed tweak.
- **Latency — a payload index keeps filtered ≈ unfiltered.** `pushdown+idx` runs
  at **1.75×** the unfiltered ANN latency (same ballpark) while scanning only the
  ~5% that match. Without the index, the same query is **6.8× slower** (16.8 ms)
  because every payload must be `json_extract`-scanned to evaluate the predicate.

## Where it loses / caveats (honest)

- **Pushdown is not *faster* than unfiltered here.** It's comparable (1.75×), not
  cheaper. The exact filtered scan + index lookup + result hydration cost a bit
  more than a single sublinear ANN probe on a 10k set. The win is **exactness at
  high selectivity** plus **not ballooning**, not beating ANN outright.
- **A payload index must be declared.** `CollectionConfig::with_indexed_fields`
  (or the field list on the config) creates a SQLite expression index. Without
  it, filtering is an O(N) payload scan (the `pushdown(scan)` row).
- **Indexed fields assume a consistent scalar type** (the normal contract for an
  indexed column) so the predicate can use the index; non-indexed fields use a
  type-coercing superset predicate plus an in-process re-eval, matching the
  in-memory filter semantics exactly.
- **Selectivity threshold.** Pushdown brute-forces the matching set when it is
  ≤ max(N/4, 8192); weakly selective filters fall back to ANN + post-filter
  (few candidates dropped, so recall stays high there).
- Single-run laptop timings vary run-to-run; recall is deterministic.

## Reproduce

```
cargo run --release --example filter_bench -- <dir-with-siftsmall_*.fvecs>
```
