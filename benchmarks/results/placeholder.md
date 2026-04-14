# vecdb Benchmark Results

## Synthetic Benchmarks (run_synthetic_bench.sh)

Results auto-generated — see `synthetic_results.md` (created when you run `run_synthetic_bench.sh`).

## MS MARCO Passage Retrieval

| Metric     | HNSW | IVF |
|------------|------|-----|
| Recall@10  | TBD  | TBD |
| p50 (ms)   | TBD  | TBD |
| p95 (ms)   | TBD  | TBD |
| QPS        | TBD  | TBD |

## BEIR (SciFact)

| Metric     | Dense | Sparse | Hybrid |
|------------|-------|--------|--------|
| Recall@10  | TBD   | TBD    | TBD    |
| p50 (ms)   | TBD   | TBD    | TBD    |
| QPS        | TBD   | TBD    | TBD    |

---

To fill these in:

1. Run download scripts:
   ```sh
   ./benchmarks/scripts/download_msmarco.sh
   ./benchmarks/scripts/download_beir.sh scifact
   ```

2. Embed passages/documents with your model of choice (sentence-transformers, OpenAI, etc.)
   Add a `vector` field (float32 array) to each JSONL record.

3. Ingest via vecdb-cli:
   ```sh
   vecdb ingest --collection msmarco --file benchmarks/data/msmarco_embedded.jsonl
   ```

4. Run vecdb-bench with the dataset file:
   ```sh
   ./target/release/vecdb-bench \
       --server http://localhost:8080 \
       --collection msmarco \
       --dataset-file benchmarks/data/msmarco_embedded.jsonl \
       --num-queries 1000 \
       --k 10 \
       --output json
   ```

5. Replace TBD values above with the measured numbers.
