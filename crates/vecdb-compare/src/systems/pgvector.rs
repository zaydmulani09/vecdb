//! pgvector adapter — postgres + pgvector in the compose container (port 5433).

use std::time::Instant;

use bytes::Bytes;
use futures_util::SinkExt;
use tokio::pin;
use tokio_postgres::{CopyInSink, NoTls};

use crate::dataset::{percentile, recall_at_k};
use crate::systems::{container_disk_mb, container_mem_mb};
use crate::{Bench, Row};

const CONN: &str = "host=127.0.0.1 port=5433 user=postgres password=bench dbname=bench";

pub fn run(b: &Bench) -> Option<Result<Row, String>> {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(rt.block_on(async { run_async(b).await }))
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

async fn run_async(b: &Bench) -> Result<Row, String> {
    let (client, connection) = tokio_postgres::connect(CONN, NoTls)
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

    let mem_mb = container_mem_mb("compose-pgvector-1").unwrap_or(0.0);
    let disk_mb = container_disk_mb("compose-pgvector-1", "/var/lib/postgresql/data").unwrap_or(0.0);

    // ── Query sweep ──────────────────────────────────────────────
    let stmt = client
        .prepare("SELECT id FROM items ORDER BY emb <-> $1::vector LIMIT $2")
        .await
        .map_err(|e| e.to_string())?;
    let mut results = Vec::with_capacity(b.queries.len());
    let mut lat = Vec::with_capacity(b.queries.len());
    let t_all = Instant::now();
    for q in &b.queries {
        let lit = vec_literal(q);
        let t = Instant::now();
        let rows = client
            .query(&stmt, &[&lit, &(b.k as i64)])
            .await
            .map_err(|e| e.to_string())?;
        lat.push(t.elapsed().as_micros());
        results.push(rows.iter().map(|r| r.get::<_, i32>(0).to_string()).collect::<Vec<_>>());
    }
    let total_s = t_all.elapsed().as_secs_f64();
    lat.sort_unstable();

    Ok(Row {
        system: "pgvector".to_string(),
        recall10: recall_at_k(&results, &b.truth, b.k),
        build_s,
        qps: b.queries.len() as f64 / total_s,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
        mem_mb,
        disk_mb,
    })
}
