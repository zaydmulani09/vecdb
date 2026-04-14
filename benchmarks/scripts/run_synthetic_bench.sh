#!/bin/sh
# Runs synthetic benchmarks against a running vecdb server and writes results to markdown.
# Usage: ./benchmarks/scripts/run_synthetic_bench.sh
#
# Requires:
#   - vecdb server running (set VECDB_SERVER env var, default http://localhost:8080)
#   - vecdb-bench binary built (cargo build --release -p vecdb-bench)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_BIN="$REPO_ROOT/target/release/vecdb-bench"
RESULTS_DIR="$REPO_ROOT/benchmarks/results"
RESULTS_FILE="$RESULTS_DIR/synthetic_results.md"
VECDB_SERVER="${VECDB_SERVER:-http://localhost:8080}"

echo "=== vecdb Synthetic Benchmark Runner ==="
echo "Server : $VECDB_SERVER"
echo ""

# Check binary exists
if [ ! -f "$BENCH_BIN" ]; then
    echo "ERROR: vecdb-bench binary not found at $BENCH_BIN"
    echo ""
    echo "Build it first:"
    echo "  PATH=\"\$PATH:/c/msys64/mingw64/bin\" cargo +stable-x86_64-pc-windows-gnu build --release -p vecdb-bench"
    echo "  # or on Linux/macOS:"
    echo "  cargo build --release -p vecdb-bench"
    exit 1
fi

# Check server is reachable
if ! wget --quiet --timeout=5 --spider "$VECDB_SERVER/health" 2>/dev/null; then
    echo "ERROR: vecdb server not reachable at $VECDB_SERVER"
    echo ""
    echo "Start the server first:"
    echo "  ./target/release/vecdb-api --port 8080 --data-dir /tmp/vecdb-bench"
    echo "  # or via Docker:"
    echo "  docker compose up -d"
    exit 1
fi

echo "Server is healthy. Starting benchmarks..."
echo ""

mkdir -p "$RESULTS_DIR"

# Write markdown header
TIMESTAMP=$(date -u '+%Y-%m-%d %H:%M:%S UTC' 2>/dev/null || echo "unknown")
cat > "$RESULTS_FILE" << HEADER
# vecdb Synthetic Benchmark Results

Generated: $TIMESTAMP
Server: $VECDB_SERVER

## Configuration

Synthetic data: random unit-normalized float32 vectors (seeded RNG).
Index: HNSW (default). Metric: cosine.

## Results

| Config  | Vectors | Dim | Queries | k  | p50 (ms) | p95 (ms) | p99 (ms) | QPS   | Recall@k |
|---------|---------|-----|---------|----|----------|----------|----------|-------|----------|
HEADER

# Benchmark configurations: "label n dim queries k"
CONFIGS="small:1000:128:100:10 medium:10000:128:100:10 large:50000:128:100:10"

RUN_FAILED=0

for cfg in $CONFIGS; do
    label=$(echo "$cfg" | cut -d: -f1)
    n=$(echo "$cfg" | cut -d: -f2)
    dim=$(echo "$cfg" | cut -d: -f3)
    queries=$(echo "$cfg" | cut -d: -f4)
    k=$(echo "$cfg" | cut -d: -f5)

    echo "Running: $label (n=$n, dim=$dim, queries=$queries, k=$k)..."

    collection="bench-synthetic-$label-$$"

    output=$("$BENCH_BIN" \
        --server "$VECDB_SERVER" \
        --collection "$collection" \
        --num-vectors "$n" \
        --dimension "$dim" \
        --num-queries "$queries" \
        --k "$k" \
        --output json 2>&1)
    exit_code=$?

    if [ $exit_code -ne 0 ]; then
        echo "  WARN: bench run failed (exit $exit_code). Using placeholder values."
        echo "  Output: $output"
        echo "| $label | $n | $dim | $queries | $k | ERR | ERR | ERR | ERR | ERR |" >> "$RESULTS_FILE"
        RUN_FAILED=1
        continue
    fi

    # Parse JSON output fields (portable, no jq required)
    p50=$(echo "$output" | grep -o '"p50_ms":[^,}]*' | grep -o '[0-9.]*' | head -1)
    p95=$(echo "$output" | grep -o '"p95_ms":[^,}]*' | grep -o '[0-9.]*' | head -1)
    p99=$(echo "$output" | grep -o '"p99_ms":[^,}]*' | grep -o '[0-9.]*' | head -1)
    qps=$(echo "$output" | grep -o '"qps":[^,}]*' | grep -o '[0-9.]*' | head -1)
    recall=$(echo "$output" | grep -o '"recall":[^,}]*' | grep -o '[0-9.]*' | head -1)

    p50="${p50:-N/A}"
    p95="${p95:-N/A}"
    p99="${p99:-N/A}"
    qps="${qps:-N/A}"
    recall="${recall:-N/A}"

    echo "  p50=${p50}ms  p95=${p95}ms  p99=${p99}ms  QPS=${qps}  Recall=${recall}"
    printf "| %-7s | %-7s | %-3s | %-7s | %-2s | %-8s | %-8s | %-8s | %-5s | %-8s |\n" \
        "$label" "$n" "$dim" "$queries" "$k" "$p50" "$p95" "$p99" "$qps" "$recall" \
        >> "$RESULTS_FILE"
done

# Append notes
cat >> "$RESULTS_FILE" << FOOTER

## Notes

- Synthetic vectors are random unit-normalized float32 (seeded StdRng).
- Recall is computed against brute-force ground truth on the same dataset.
- For real-world benchmarks, see download scripts in benchmarks/scripts/.
- Hardware: $(uname -m 2>/dev/null || echo "unknown arch"), $(uname -s 2>/dev/null || echo "unknown OS")
FOOTER

echo ""
echo "=== Results ==="
cat "$RESULTS_FILE"
echo ""
echo "Saved to: $RESULTS_FILE"

if [ $RUN_FAILED -ne 0 ]; then
    echo ""
    echo "WARNING: one or more benchmark runs failed. Check server logs."
    exit 1
fi

exit 0
