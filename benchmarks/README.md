# vecdb Benchmarks

## Overview

This directory contains scripts to download benchmark datasets, run synthetic benchmarks, and record results.

---

## Quick Start — Synthetic Benchmark (no downloads needed)

Start a vecdb server, then:

```sh
./benchmarks/scripts/run_synthetic_bench.sh
```

Results are written to `benchmarks/results/synthetic_results.md`.

---

## Running a vecdb Server for Benchmarking

```sh
# From repo root, after cargo build --release
./target/release/vecdb-api --port 8080 --data-dir /tmp/vecdb-bench-data
# OR via Docker
docker compose up -d
```

---

## MS MARCO Passage Corpus

**What it is:** ~8.8 million passages from web documents, used for passage retrieval benchmarks. Standard in IR research.

**Size:** ~3 million passages after filtering, ~8.8 GB download.

**Hardware requirement:** 16 GB RAM recommended.

```sh
./benchmarks/scripts/download_msmarco.sh
# or dry-run to see what would be downloaded:
./benchmarks/scripts/download_msmarco.sh --dry-run
```

Output: `benchmarks/data/msmarco_passages.jsonl`

**Note:** MS MARCO is text-only. You must embed the passages with a model of your choice before ingesting into vecdb. Example with a Python embedding script:

```sh
# After download, embed passages (example — adapt to your model):
python3 -c "
import json
# Load passages from benchmarks/data/msmarco_passages.jsonl
# Generate embeddings with sentence-transformers or OpenAI API
# Write to benchmarks/data/msmarco_embedded.jsonl with 'id', 'vector', 'text' fields
print('Embed passages here')
"
# Then ingest:
vecdb ingest --collection msmarco --file benchmarks/data/msmarco_embedded.jsonl
# Then benchmark:
./target/release/vecdb-bench --server http://localhost:8080 --collection msmarco \
    --dataset-file benchmarks/data/msmarco_embedded.jsonl --queries 1000 --k 10
```

---

## BEIR Datasets

**What it is:** Benchmarking Information Retrieval — a heterogeneous benchmark suite with 18 datasets.

**Available datasets (smallest first):**

| Dataset    | Docs    | Domain       |
|-----------|---------|--------------|
| scifact   | ~5k     | Science      |
| nfcorpus  | ~3.6k   | Medical      |
| arguana   | ~8.7k   | Arguments    |
| fiqa      | ~57k    | Finance Q&A  |
| trec-covid| ~171k   | Biomedical   |

```sh
# Download scifact (default, smallest — good for testing)
./benchmarks/scripts/download_beir.sh

# Download a specific dataset
./benchmarks/scripts/download_beir.sh fiqa
```

Output: `benchmarks/data/beir/{name}/`

---

## Using vecdb-bench Directly

```sh
# Build bench harness
cargo build --release -p vecdb-bench

# Synthetic data (no download needed)
./target/release/vecdb-bench \
    --server http://localhost:8080 \
    --collection bench-test \
    --num-vectors 10000 \
    --dimension 128 \
    --num-queries 100 \
    --k 10

# From JSONL file (after downloading + embedding a dataset)
./target/release/vecdb-bench \
    --server http://localhost:8080 \
    --collection msmarco \
    --dataset-file benchmarks/data/msmarco_embedded.jsonl \
    --num-queries 1000 \
    --k 10 \
    --output json
```

---

## Expected Performance (to be filled after real runs)

See `benchmarks/results/placeholder.md` for the expected results format.

Current synthetic results (auto-generated): `benchmarks/results/synthetic_results.md`
