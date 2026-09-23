# Stage 1 — builder: compile static binary with musl
FROM rust:1.83-slim AS builder

RUN apt-get update && apt-get install -y musl-tools && rm -rf /var/lib/apt/lists/*
RUN rustup target add x86_64-unknown-linux-musl

ENV CARGO_NET_GIT_FETCH_WITH_CLI=true

WORKDIR /build
COPY . .

RUN cargo build --release --target x86_64-unknown-linux-musl -p vecdb-api

# Stage 2 — runtime: minimal alpine image with binary only
FROM alpine:3.19

RUN apk add --no-cache ca-certificates wget

RUN adduser -D -u 1000 vecdb
RUN mkdir -p /data && chown vecdb:vecdb /data

COPY --from=builder /build/target/x86_64-unknown-linux-musl/release/vecdb-api /usr/local/bin/vecdb

USER vecdb

EXPOSE 8080

ENV VECDB_DATA_DIR=/data

VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=10s \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/vecdb"]
CMD ["--port", "8080", "--data-dir", "/data"]
