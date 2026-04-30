# vecdb Demo Notebook

Interactive walkthrough of vecdb's full feature set — collection management,
vector upsert, dense search, sparse search, hybrid search, SQL queries, and
the Python SDK. Uses **synthetic 8-dimensional vectors** so no embedding model
or GPU is required.

---

## Prerequisites

- Python 3.9+
- A running vecdb server (see below)

---

## Start the server

### Option A — Docker

```bash
docker run -p 8080:8080 vecdb:latest
```

### Option B — Build from source

```bash
cargo build --release -p vecdb-api
./target/release/vecdb-api --port 8080 --data-dir ./demo_data
```

The server listens on `http://localhost:8080` by default.

---

## Install notebook dependencies

```bash
cd notebooks/
pip install -r requirements.txt
```

This installs:
- `jupyter` and `notebook` — the Jupyter server
- `vecdb-client` — the vecdb Python SDK (from `../sdks/python`)
- `requests` — for direct HTTP calls in collection listing

---

## Run the notebook

```bash
jupyter notebook vecdb_demo.ipynb
```

Then open the URL printed in the terminal (usually `http://localhost:8888`).
Run cells top to bottom with **Shift+Enter** or use **Cell → Run All**.

---

## What the notebook demonstrates

The notebook creates a fresh `demo` collection, inserts **20 documents**
across three topic clusters (Machine Learning, Databases, Systems), then
walks through every search mode:

| Section | Feature |
|---------|---------|
| 1 | Setup — connect to server, verify health |
| 2 | Create a collection (dim=8, cosine metric) |
| 3 | Define 20 synthetic documents |
| 4 | Batch upsert all documents |
| 5 | Dense search — HNSW cosine similarity |
| 6 | Sparse search — BM25 keyword scoring |
| 7 | Hybrid search — weighted dense + sparse fusion, alpha tuning |
| 8 | SQL queries — VECTOR_SIM with metadata filters |
| 9 | Collection inspection — stats and listing |
| 10 | Get and delete — point lookups and deletes |
| 11 | Cleanup — delete the demo collection |

---

## Notes

- **No embedding model needed.** Vectors are hand-crafted 8-float arrays
  designed to cluster by topic. The math is easy to follow.
- **Runs instantly.** 20 documents at dim=8 completes in milliseconds.
- **Safe to re-run.** The notebook deletes the `demo` collection at startup
  (Section 2) and at the end (Section 11) so repeated runs are clean.
- **Real embeddings.** To use real embeddings, replace the `documents` list
  in Section 3 with your own vectors from a sentence transformer model
  (e.g. `sentence-transformers/all-MiniLM-L6-v2` at dim=384).
