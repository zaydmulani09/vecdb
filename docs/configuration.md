# vecdb Configuration Reference

vecdb reads configuration from three sources in order of increasing precedence:

1. **Defaults** — built-in values listed below
2. **Config file** — TOML file passed with `--config vecdb.toml`
3. **Environment variables** — `VECDB__<KEY>` (double-underscore separator)
4. **CLI flags** — `--port`, `--data-dir`, `--api-key`, `--log-level`

Higher-precedence sources override lower ones. For example, a `VECDB__PORT=9000` env var overrides a `port = 8080` in the config file, and `--port 7000` on the command line overrides both.

---

## Environment Variables

> **Important:** vecdb uses the `config` crate with a **double-underscore** (`__`) separator.
> The variable for `port` is `VECDB__PORT`, not `VECDB_PORT`.

| Variable | Type | Default | Description |
|---|---|---|---|
| `VECDB__HOST` | string | `127.0.0.1` | IP address to bind. Use `0.0.0.0` to listen on all interfaces. |
| `VECDB__PORT` | u16 | `6333` | TCP port to listen on. |
| `VECDB__DATA_DIR` | string | `./data` | Directory where collection files are stored. Created if it does not exist. |
| `VECDB__API_KEY` | string | _(none)_ | If set, all HTTP requests must include `X-Api-Key: <value>`. Unset or empty = no auth. |
| `VECDB__LOG_LEVEL` | string | `info` | Tracing log level: `trace`, `debug`, `info`, `warn`, or `error`. |
| `VECDB__QUERY_TIMEOUT_MS` | u64 | `5000` | Per-request timeout in milliseconds. Requests that exceed this return `408`. |
| `VECDB__MAX_CONNECTIONS` | usize | `100` | Maximum number of concurrent HTTP connections. |

### Examples

```bash
# Start on all interfaces, port 8080, with API key auth
VECDB__HOST=0.0.0.0 VECDB__PORT=8080 VECDB__API_KEY=my-secret \
    ./vecdb-api --data-dir /var/lib/vecdb

# Debug logging
VECDB__LOG_LEVEL=debug ./vecdb-api

# Longer timeout for large queries
VECDB__QUERY_TIMEOUT_MS=30000 ./vecdb-api
```

---

## CLI Flags

| Flag | Description |
|---|---|
| `--config <path>` | Path to TOML config file |
| `--port <n>` | Override listen port |
| `--data-dir <path>` | Override data directory |
| `--api-key <key>` | Set API key (overrides env var) |
| `--log-level <level>` | Override log level |

---

## Config File (TOML)

Pass a config file with `--config vecdb.toml`. All fields are optional — omit any field to use the default.

```toml
# vecdb configuration file
# Pass with: vecdb-api --config vecdb.toml
#
# Precedence (highest to lowest):
#   CLI flags > environment variables > this file > defaults

# Network
host = "0.0.0.0"      # Bind address (default: 127.0.0.1)
port = 8080            # Listen port (default: 6333)

# Storage
data_dir = "/var/lib/vecdb"   # Collection storage (default: ./data)

# Logging
log_level = "info"     # trace / debug / info / warn / error

# Timeouts
query_timeout_ms = 30000   # Per-request timeout in ms (default: 5000)

# Connections
max_connections = 100  # Max concurrent HTTP connections (default: 100)

# Authentication
# Uncomment to require X-Api-Key header on all requests:
# api_key = "your-secret-key-here"
```

---

## Docker / docker-compose

When running via `docker compose`, environment variables are set in `docker-compose.yml`:

```yaml
environment:
  - VECDB__DATA_DIR=/data
  - VECDB__LOG_LEVEL=info
  # Uncomment to enable auth:
  # - VECDB__API_KEY=your-secret-key
```

The `VECDB__PORT` variable is not needed in docker-compose because the binary's `--port 8080` CMD argument sets it directly. Override it via:

```yaml
command: ["--port", "9000", "--data-dir", "/data"]
```

---

## Client Tools

The CLI (`vecdb-cli`) and benchmark harness (`vecdb-bench`) read the server URL from:

| Variable | Default | Description |
|---|---|---|
| `VECDB_SERVER` | `http://localhost:6333` | Server base URL for CLI and bench tools |

Note: this variable uses a **single** underscore — it is not part of the server config system, just a convenience variable read by the client binaries.

```bash
# Point CLI at a remote server
VECDB_SERVER=http://prod.example.com:8080 vecdb collection list
```

---

## Storage Layout

All collection files live under `data_dir`:

```
data_dir/
├── {collection}.db           # SQLite metadata + payload (r2d2 pool)
├── {collection}.db-wal       # SQLite WAL file
├── {collection}.vectors      # Memory-mapped flat f32 file
├── {collection}.wal          # vecdb write-ahead log
├── {collection}.hnsw.json    # HNSW index snapshot (if index_type=hnsw)
├── {collection}.ivf.json     # IVF index snapshot (if index_type=ivf)
└── {collection}.sparse.json  # BM25 inverted index snapshot
```

These files are written atomically on graceful shutdown (SIGINT/SIGTERM). Running `docker stop` triggers the graceful shutdown handler, which calls `checkpoint()` and `save_indexes()` for every open collection before the process exits.

---

## Performance Tuning

| Parameter | Recommendation |
|---|---|
| `query_timeout_ms` | Increase to `30000` for large collections or slow hardware |
| `max_connections` | Increase to `500` for high-concurrency workloads |
| `VECDB__LOG_LEVEL` | Set to `warn` in production to reduce I/O overhead |
| Index type | `hnsw` for collections under ~1M vectors; `ivf` for larger |
| `alpha` per query | Start at `0.7` (dense-heavy); tune toward `0.5` if keyword precision matters |
