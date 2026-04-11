#!/bin/sh
# Manual health check script — use outside Docker for CI/CD pipelines
# Usage: ./scripts/docker-health-check.sh [host] [port]
HOST="${1:-localhost}"
PORT="${2:-8080}"
URL="http://${HOST}:${PORT}/health"

response=$(wget --quiet --timeout=5 --output-document=- "$URL" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "FAIL: could not reach $URL"
    exit 1
fi

status=$(echo "$response" | grep -o '"status":"ok"')
if [ -z "$status" ]; then
    echo "FAIL: unexpected response: $response"
    exit 1
fi

echo "OK: vecdb healthy at $URL"
exit 0
