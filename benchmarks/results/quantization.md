# Scalar Quantization (int8) — measured results

Phase 2 measurement of the optional int8 scalar-quantization index mode vs
float32. **Every number below comes from an actual run of the committed harness
(`crates/vecdb-core/examples/quant_bench.rs`) and is reproducible.** Nothing here
is estimated.

## Methodology

- **Dataset:** SIFT10K (`siftsmall`) from the TEXMEX corpus — 10,000 base
  vectors × 128 dims, 100 query vectors, 100-NN ground truth. Standard, public,
  citeable. Fetch:
  ```
  curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
  tar xzf siftsmall.tar.gz
  ```
- **Metric:** Euclidean (SIFT's native metric; the ground truth is L2 nearest
  neighbors).
- **Recall@10:** mean over the 100 queries of
  `|returned_top10 ∩ groundtruth_top10| / 10`.
- **Modes:**
  - `f32-flat` — exact full-precision brute-force L2 scan. Recall 1.0 by
    construction (it reproduces the ground truth); the baseline that **isolates
    quantization loss** (same flat algorithm, only precision differs).
  - `int8-flat` — the `Quantization::ScalarInt8` index: per-dimension asymmetric
    int8 codes, flat scan with a precomputed-norm dot kernel.
  - `binary-flat` — the `Quantization::Binary` index: 1 bit/dim (per-dim
    mean threshold), Hamming distance via popcount.
  - `binary+rerank` — binary-Hamming to fetch the top 100 candidates, then an
    exact float32 L2 rerank of those down to top-10 (the intended BQ pipeline).
  - `f32-hnsw` — the default HNSW mode, for production context (approximate).
- **Index bytes:** resident vector-store bytes. `f32` = `N·D·4`; `int8` = codes
  (`N·D·1`) + per-vector ‖x‖² cache (`N·4`) + per-dim params. HNSW additionally
  holds graph structures not counted here (so its real footprint is larger than
  the f32 floor shown).
- **Hardware:** Intel Core i7-1355U (12 logical cores), Windows 11.
  rustc 1.95.0, `x86_64-pc-windows-msvc`, `--release`, default `target-cpu`
  (no AVX2 flag — applied equally to both modes).

## Results

```
mode              recall@10      index bytes      mean µs     p50 µs     p99 µs
f32-flat             1.0000          5120000       2571.8       2797       5050
int8-flat            0.9900          1321024       2996.1       2313       6776
binary-flat          0.3340           160512       1060.8        885       2874
binary+rerank        0.7620           160512        936.7        819       2021
f32-hnsw             0.9990          5120000        707.1        705       1129
```

_Single-run timings on a busy laptop vary run-to-run (f32-flat mean measured
1289–2572 µs across runs); treat latency as order-of-magnitude, not precise.
Recall and memory are deterministic. Phase 4 pins down latency with warmup +
repeated trials on a quiet machine._

## int8 deltas (int8-flat vs f32-flat, same algorithm)

| Dimension | f32-flat | int8-flat | Delta |
|-----------|----------|-----------|-------|
| Memory (index bytes) | 5,120,000 | 1,321,024 | **3.88× smaller** |
| Recall@10 | 1.0000 | 0.9900 | **−0.0100 (1.0% loss)** |
| Latency (mean µs/query) | ~1300–2600 | ~2300–3000 | **~1.2–1.8× slower** |

## binary (1-bit) deltas vs f32-flat

| Dimension | f32-flat | binary-flat | binary+rerank |
|-----------|----------|-------------|---------------|
| Memory (index bytes) | 5,120,000 | 160,512 (**31.9× smaller**) | 160,512 |
| Recall@10 | 1.0000 | **0.334** | **0.762** |
| Latency (mean µs/query) | ~1300–2600 | ~1060 | ~940 (incl. rerank) |

Binary alone is a coarse ~32× filter (recall 0.33); a full-precision rerank of
the top-100 Hamming candidates recovers recall to 0.76 at k=10 while keeping the
tiny index. Rerank uses the on-disk float32 vectors, so the stored index stays
32× smaller. Mean-threshold binary is the simplest scheme — random-rotation
variants (e.g. RaBitQ) would push recall higher; not implemented.

## Where int8 loses (honest)

- **Latency:** on SIFT10K the int8 flat scan is **~1.8× slower** than the f32
  flat scan, not faster. The whole base set (5 MB as f32) fits in cache, so the
  scan is **compute-bound**, and reconstructing/int8→f32-converting the codes
  costs more arithmetic than the pure-f32 FMA loop. Int8's advantage is memory
  **bandwidth**, which only dominates once the set no longer fits in cache — a
  scale effect that should appear at SIFT1M (512 MB f32 vs 128 MB int8),
  measured in Phase 4. A hand-written SIMD int8 kernel (`VPMOVSXBD`/`VCVTDQ2PS`)
  would also narrow this; not done yet.
- **Recall caveat:** SIFT vectors are natively small integers (0–255), so int8
  quantization is nearly lossless here (1%). Float-valued embedding datasets
  (e.g. GIST-960, sentence embeddings) would show a larger recall delta. Treat
  the 1% as a favorable lower bound, not a universal figure.

## Takeaway

- **int8:** ~4× smaller in-memory index at ~1% recall loss on SIFT10K — the
  memory reduction that lets more vectors fit in RAM. Not a speed win at this
  cache-resident scale; latency crossover deferred to Phase 4.
- **binary:** ~32× smaller, but coarse (recall 0.33 raw); with a top-100 rerank
  recall reaches 0.76 at k=10. Choose it for extreme memory pressure where a
  rerank pass is acceptable.

Both are opt-in per collection (`Quantization::{ScalarInt8, Binary}`) and leave
the default float32 path unchanged.

## Reproduce

```
cargo run --release --example quant_bench -- <dir-with-siftsmall_*.fvecs>
```
