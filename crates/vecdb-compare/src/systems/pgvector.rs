//! pgvector adapter — managed Postgres (Neon) with pgvector.
//!
//! No local Postgres/compile on this box, so pgvector runs on a free managed
//! instance. Connection string comes from env `PGVECTOR_CONN` (libpq form, e.g.
//! `host=... user=... password=... dbname=... sslmode=require`); if unset the
//! system is skipped.
//!
//! Fairness: because the server is remote, per-query latency is measured
//! **server-side** via `EXPLAIN (ANALYZE)` execution time, which excludes the
//! client↔cloud network RTT. Build time necessarily includes network ingest and
//! is labeled as such in BENCHMARK.md.

use std::time::Instant;

use bytes::Bytes;
use futures_util::SinkExt;
use tokio::pin;
use tokio_postgres::CopyInSink;

use crate::dataset::{percentile, recall_at_k};
use crate::systems::process_mem_mb;
use crate::{Bench, Row};

pub fn run(b: &Bench) -> Option<Result<Row, String>> {
    let conn = std::env::var("PGVECTOR_CONN").ok()?;
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(rt.block_on(async { run_async(b, &conn).await }))
}

fn vec_literal(v: &[f32]) -> String {
    let mut s = String::with_capacity(v.len() * 8);
    s.push('[');
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&x.to_string());
    }
    s.push(']');
    s
}

async fn run_async(b: &Bench, conn_str: &str) -> Result<Row, String> {
    // TLS (managed Postgres requires it). Use the ring provider explicitly to
    // avoid aws-lc-rs's C/NASM build on Windows.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let tls_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(tls_config);

    let (client, connection) = tokio_postgres::connect(conn_str, tls)
        .await
        .map_err(|e| format!("pgvector connect failed: {e}"))?;
    tokio::spawn(async move {
        let _ = connection.await;
    });

    client
        .batch_execute(
            "CREATE EXTENSION IF NOT EXISTS vector; \
             DROP TABLE IF EXISTS items; \
             CREATE TABLE items (id int PRIMARY KEY, emb vector(128));",
        )
        .await
        .map_err(|e| e.to_string())?;

    // ── Build: COPY rows in, then create the HNSW index ──────────
    let t = Instant::now();
    {
        let sink: CopyInSink<Bytes> = client
            .copy_in("COPY items (id, emb) FROM STDIN")
            .await
            .map_err(|e| e.to_string())?;
        pin!(sink);
        let mut buf = String::new();
        for (i, v) in b.base.iter().enumerate() {
            buf.clear();
            buf.push_str(&i.to_string());
            buf.push('\t');
            buf.push_str(&vec_literal(v));
            buf.push('\n');
            sink.send(Bytes::copy_from_slice(buf.as_bytes()))
                .await
                .map_err(|e| e.to_string())?;
        }
        sink.close().await.map_err(|e| e.to_string())?;
    }
    // Default pgvector HNSW build params (m=16, ef_construction=64).
    client
        .batch_execute("CREATE INDEX ON items USING hnsw (emb vector_l2_ops);")
        .await
        .map_err(|e| e.to_string())?;
    let build_s = t.elapsed().as_secs_f64();

    // Memory: for a managed instance we can't read server RSS; report 0 and note
    // it in the writeup. (process_mem_mb kept for a future local-postgres path.)
    let mem_mb = process_mem_mb("this-process-does-not-exist").unwrap_or(0.0);
    let disk_mb = 0.0;

    // ── Query sweep: ids from the real query (recall), latency from
    //    server-side EXPLAIN ANALYZE execution time (excludes RTT) ──
    let mut results = Vec::with_capacity(b.queries.len());
    let mut lat = Vec::with_capacity(b.queries.len());
    for q in &b.queries {
        let lit = vec_literal(q);
        // Inline the vector literal (postgres infers a bound $1 as `vector`,
        // which a Rust String won't serialize as). Values are our own floats.
        let sel_sql = format!(
            "SELECT id FROM items ORDER BY emb <-> '{lit}'::vector LIMIT {}",
            b.k
        );
        let rows = client
            .query(sel_sql.as_str(), &[])
            .await
            .map_err(|e| e.to_string())?;
        results.push(rows.iter().map(|r| r.get::<_, i32>(0).to_string()).collect::<Vec<_>>());

        // Server-side execution time (text EXPLAIN, parse "Execution Time:").
        let explain_sql = format!(
            "EXPLAIN (ANALYZE) SELECT id FROM items ORDER BY emb <-> '{lit}'::vector LIMIT {}",
            b.k
        );
        let ex_rows = client.query(explain_sql.as_str(), &[]).await.map_err(|e| e.to_string())?;
        let mut ms = 0.0f64;
        for r in &ex_rows {
            let line: String = r.get(0);
            if let Some(rest) = line.trim().strip_prefix("Execution Time:") {
                ms = rest.trim().trim_end_matches("ms").trim().parse().unwrap_or(0.0);
            }
        }
        lat.push((ms * 1000.0) as u128); // µs
    }
    lat.sort_unstable();
    let qps = 1_000_000.0 / (lat.iter().sum::<u128>() as f64 / lat.len() as f64).max(1.0);

    Ok(Row {
        system: "pgvector(neon)".to_string(),
        recall10: recall_at_k(&results, &b.truth, b.k),
        build_s,
        qps,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
        mem_mb,
        disk_mb,
    })
}
