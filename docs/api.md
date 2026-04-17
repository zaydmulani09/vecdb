# vecdb HTTP API Reference

Base URL: `http://localhost:6333` (default) or wherever you started the server.

All request and response bodies are JSON (`Content-Type: application/json`).

---

## Authentication

If the server was started with `--api-key` (or `VECDB__API_KEY` env var), every request must include the header:

```
X-Api-Key: <your-secret-key>
```

Without this header (when auth is configured), the server returns:

```json
{ "error": { "code": "unauthorized", "message": "missing or invalid api key" } }
```

HTTP status: `401 Unauthorized`.

When no API key is configured, all requests are accepted without authentication.

---

## Error Responses

All errors use this envelope:

```json
{
  "error": {
    "code": "not_found",
    "message": "collection 'my-col' does not exist"
  }
}
```

| HTTP Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed JSON or invalid field values |
| 401 | `unauthorized` | Missing or wrong `X-Api-Key` |
| 404 | `not_found` | Collection or vector does not exist |
| 409 | `collection_already_exists` | Creating a collection that already exists |
| 422 | `unprocessable` | Valid JSON but semantically invalid (e.g. wrong dimension) |
| 500 | `internal_error` | Unexpected server error |

---

## Endpoints

### GET /health

Returns server health and a summary of loaded collections.

**Response:**

| Field | Type | Description |
|---|---|---|
| `status` | string | Always `"ok"` when server is healthy |
| `version` | string | vecdb version string |
| `vector_count` | number | Total vectors across all collections |
| `collections` | number | Number of open collections |

**Example:**

```bash
curl http://localhost:6333/health
```

```json
{
  "status": "ok",
  "version": "0.1.0",
  "vector_count": 42000,
  "collections": 3
}
```

---

### POST /collections

Create a new collection.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `name` | string | ✓ | — | Collection name (unique) |
| `dimension` | number | ✓ | — | Vector dimensionality |
| `metric` | string | — | `"cosine"` | Distance metric: `"cosine"`, `"euclidean"`, `"dot"` |
| `index_type` | string | — | `"hnsw"` | Index backend: `"hnsw"` or `"ivf"` |

**Response:** `201 Created` — collection object.

| Field | Type | Description |
|---|---|---|
| `name` | string | Collection name |
| `dimension` | number | Vector dimensionality |
| `metric` | string | Distance metric |
| `index_type` | string | Index backend |
| `vector_count` | number | Always 0 at creation |
| `created_at` | string | RFC 3339 timestamp |

**Example:**

```bash
curl -X POST http://localhost:6333/collections \
  -H "Content-Type: application/json" \
  -d '{"name":"docs","dimension":1536,"metric":"cosine","index_type":"hnsw"}'
```

```json
{
  "name": "docs",
  "dimension": 1536,
  "metric": "cosine",
  "index_type": "hnsw",
  "vector_count": 0,
  "created_at": "2026-01-01T00:00:00Z"
}
```

**Error:** `409 Conflict` if a collection with this name already exists.

---

### GET /collections

List all collections.

**Response:** `200 OK` — array of collection objects (same shape as POST /collections response).

**Example:**

```bash
curl http://localhost:6333/collections
```

```json
[
  {
    "name": "docs",
    "dimension": 1536,
    "metric": "cosine",
    "index_type": "hnsw",
    "vector_count": 1000,
    "created_at": "2026-01-01T00:00:00Z"
  }
]
```

---

### GET /collections/:name

Get a single collection by name.

**Path parameter:** `name` — collection name.

**Response:** `200 OK` — collection object.

**Example:**

```bash
curl http://localhost:6333/collections/docs
```

**Error:** `404 Not Found` if the collection does not exist.

---

### DELETE /collections/:name

Delete a collection and all its data (vectors, index files, SQLite database).

**Path parameter:** `name` — collection name.

**Response:** `200 OK`

```json
{ "deleted": true }
```

**Example:**

```bash
curl -X DELETE http://localhost:6333/collections/docs
```

**Error:** `404 Not Found` if the collection does not exist.

---

### POST /collections/:name/vectors

Upsert (insert or update) one or more vectors.

**Path parameter:** `name` — collection name.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `records` | array | ✓ | Array of vector records |

Each record:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | string | ✓ | Unique identifier for this vector |
| `vector` | number[] | ✓ | Float32 values; length must equal collection dimension |
| `payload` | object | — | Arbitrary JSON metadata |
| `text` | string | — | Text content for BM25 sparse indexing |

**Response:** `200 OK`

| Field | Type | Description |
|---|---|---|
| `inserted` | number | Count of new records |
| `updated` | number | Count of overwritten records |
| `errors` | array | Per-record errors (empty on full success) |
| `time_ms` | number | Processing time in milliseconds |

**Example:**

```bash
curl -X POST http://localhost:6333/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "id": "doc1",
        "vector": [0.1, 0.2, 0.9],
        "text": "machine learning basics",
        "payload": {"topic": "AI", "year": 2024}
      },
      {
        "id": "doc2",
        "vector": [0.8, 0.1, 0.1],
        "text": "database systems overview",
        "payload": {"topic": "DB", "year": 2023}
      }
    ]
  }'
```

```json
{
  "inserted": 2,
  "updated": 0,
  "errors": [],
  "time_ms": 3
}
```

---

### DELETE /collections/:name/vectors

Delete vectors by ID.

**Path parameter:** `name` — collection name.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `ids` | string[] | ✓ | Array of vector IDs to delete |

**Response:** `200 OK`

```json
{ "deleted": 2, "errors": [], "time_ms": 1 }
```

**Example:**

```bash
curl -X DELETE http://localhost:6333/collections/docs/vectors \
  -H "Content-Type: application/json" \
  -d '{"ids":["doc1","doc2"]}'
```

---

### GET /collections/:name/vectors/:id

Retrieve a single vector by ID.

**Path parameters:** `name` — collection name, `id` — vector ID.

**Response:** `200 OK`

| Field | Type | Description |
|---|---|---|
| `id` | string | Vector ID |
| `vector` | number[] | Float32 values |
| `payload` | object | JSON metadata (empty object if none) |
| `text` | string or null | Text content |

**Example:**

```bash
curl http://localhost:6333/collections/docs/vectors/doc1
```

```json
{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.9],
  "payload": { "topic": "AI", "year": 2024 },
  "text": "machine learning basics"
}
```

**Error:** `404 Not Found` if the vector does not exist.

---

### POST /collections/:name/search/dense

Dense (HNSW/IVF) approximate nearest-neighbour search.

**Path parameter:** `name` — collection name.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `vector` | number[] | ✓ | — | Query vector; must match collection dimension |
| `k` | number | — | `10` | Number of results to return |
| `filter` | object | — | — | Payload filter (see Filter Syntax below) |

**Response:** `200 OK` — search response object.

| Field | Type | Description |
|---|---|---|
| `results` | array | Ranked search results |
| `count` | number | Number of results returned |
| `time_ms` | number | Search time in milliseconds |

Each result:

| Field | Type | Description |
|---|---|---|
| `id` | string | Vector ID |
| `score` | number | Similarity score (higher = more similar) |
| `dense_score` | number or null | Raw dense component score |
| `sparse_score` | number or null | Raw sparse component score |
| `payload` | object | JSON metadata |
| `text` | string or null | Text content |

**Example:**

```bash
curl -X POST http://localhost:6333/collections/docs/search/dense \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.9],"k":5}'
```

```json
{
  "results": [
    {
      "id": "doc1",
      "score": 0.987,
      "dense_score": 0.987,
      "sparse_score": null,
      "payload": { "topic": "AI" },
      "text": "machine learning basics"
    }
  ],
  "count": 1,
  "time_ms": 2
}
```

---

### POST /collections/:name/search/sparse

Sparse BM25 keyword search over the `text` field.

**Path parameter:** `name` — collection name.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | ✓ | — | Text query |
| `k` | number | — | `10` | Number of results to return |
| `filter` | object | — | — | Payload filter |

**Response:** Same search response shape as dense search.

**Example:**

```bash
curl -X POST http://localhost:6333/collections/docs/search/sparse \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning","k":5}'
```

---

### POST /collections/:name/search/hybrid

Hybrid search combining dense and sparse scores.

**Path parameter:** `name` — collection name.

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `vector` | number[] | — | — | Query vector (for dense component) |
| `query` | string | — | — | Text query (for sparse component) |
| `k` | number | — | `10` | Number of results |
| `alpha` | number | — | `0.7` | Dense weight; sparse weight = `1 - alpha` |
| `strategy` | string | — | `"weighted_sum"` | Fusion strategy: `"weighted_sum"` or `"reciprocal_rank_fusion"` |
| `filter` | object | — | — | Payload filter |

At least one of `vector` or `query` must be provided.

**Example:**

```bash
curl -X POST http://localhost:6333/collections/docs/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, 0.9],
    "query": "machine learning",
    "k": 10,
    "alpha": 0.7
  }'
```

---

### POST /query

Execute a SQL query with `VECTOR_SIM` extension.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `sql` | string | ✓ | SQL query string |

**Response:** Same search response shape as search endpoints.

**Example:**

```bash
curl -X POST http://localhost:6333/query \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT id, score FROM docs WHERE VECTOR_SIM(vec, [0.1,0.2,0.9]) > 0.5 LIMIT 10"}'
```

```bash
curl -X POST http://localhost:6333/query \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM docs WHERE VECTOR_SIM(vec, [0.1,0.2,0.9]) > 0.6 AND payload->>'\''topic'\'' = '\''AI'\'' LIMIT 5"}'
```

---

### POST /explain

Return the query plan without executing the search.

**Request body:**

| Field | Type | Required | Description |
|---|---|---|---|
| `sql` | string | — | SQL query string |
| `vector` | number[] | — | Query vector |
| `query` | string | — | Text query |
| `k` | number | — | Result count |
| `collection` | string | — | Collection name (required for non-SQL explain) |

**Response:** `200 OK`

| Field | Type | Description |
|---|---|---|
| `plan` | string | Human-readable plan description |
| `index_type` | string | Chosen index backend |
| `use_hybrid` | boolean | Whether hybrid fusion is activated |
| `candidate_k` | number | Oversample factor |
| `output_k` | number | Final result count |
| `estimated_cost` | number | Cost model estimate |

**Example:**

```bash
curl -X POST http://localhost:6333/explain \
  -H "Content-Type: application/json" \
  -d '{"sql":"SELECT * FROM docs WHERE VECTOR_SIM(vec, [0.1,0.2,0.9]) > 0.5 LIMIT 10"}'
```

---

### GET /metrics

Prometheus-format metrics.

**Response:** `200 OK`, `text/plain; version=0.0.4`

Exposes counters and histograms:

| Metric | Type | Description |
|---|---|---|
| `vecdb_requests_total` | counter | Total HTTP requests by operation |
| `vecdb_request_duration_ms` | histogram | Per-request latency |
| `vecdb_search_duration_ms` | histogram | Search-specific latency |
| `vecdb_search_results_count` | gauge | Results returned per search |
| `vecdb_upsert_total` | counter | Total upserted vectors |
| `vecdb_delete_total` | counter | Total deleted vectors |
| `vecdb_errors_total` | counter | Total errors by kind |

**Example:**

```bash
curl http://localhost:6333/metrics
```

---

## Filter Syntax

The `filter` field in search requests accepts a JSON object for equality filtering or an array of conditions:

**Equality (flat object):**

```json
{ "topic": "AI", "year": 2024 }
```

Matches records where `payload.topic == "AI"` AND `payload.year == 2024`.

**Structured conditions (array):**

```json
[
  { "field": "topic", "op": "=", "value": "AI" },
  { "field": "year",  "op": ">", "value": 2020 }
]
```

Supported operators: `=`, `!=`, `<`, `<=`, `>`, `>=`, `LIKE`.

`LIKE` supports `%` wildcards: `{"field":"title","op":"LIKE","value":"%machine%"}`.

Nested JSON paths: `{"field":"meta.author","op":"=","value":"Alice"}` — dot notation navigates nested objects.
