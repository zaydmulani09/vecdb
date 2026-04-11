#!/bin/sh
# Validates Dockerfile and docker-compose.yml without running Docker
# Run from repo root: ./scripts/validate-docker.sh

PASS=0
FAIL=0

check() {
    if eval "$2"; then
        echo "PASS: $1"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $1"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1 — Dockerfile exists and has multi-stage build
check "Dockerfile has builder stage" \
    "grep -q 'AS builder' Dockerfile"

# Test 2 — Dockerfile uses musl target or documents fallback
check "Dockerfile targets linux build" \
    "grep -q 'linux' Dockerfile"

# Test 3 — docker-compose.yml has health check
check "docker-compose has healthcheck" \
    "grep -q 'healthcheck' docker-compose.yml"

# Test 4 — docker-compose.yml has named volume
check "docker-compose has named volume" \
    "grep -q 'vecdb-data' docker-compose.yml"

# Test 5 — health check script is executable
check "health check script is executable" \
    "test -x scripts/docker-health-check.sh"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
