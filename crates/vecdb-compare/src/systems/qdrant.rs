//! qdrant adapter — native Windows binary (qdrant.exe) on REST port 6333.

use std::time::{Duration, Instant};

use serde_json::json;

use crate::dataset::{percentile, recall_at_k};
use crate::systems::{env_disk_mb, process_mem_mb};
use crate::{Bench, Row};

const BASE: &str = "http://127.0.0.1:6333";
const COLLECTION: &str = "bench";

pub fn run(b: &Bench) -> Option<Result<Row, String>> {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(rt.block_on(async { run_async(b).await }))
}

async fn run_async(b: &Bench) -> Result<Row, String> {
    let http = reqwest::Client::new();

    // Reachable?
    http.get(format!("{BASE}/healthz"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| format!("qdrant unreachable at {BASE}: {e}"))?;

    // Fresh collection (Euclidean).
    let _ = http.delete(format!("{BASE}/collections/{COLLECTION}")).send().await;
    let resp = http
        .put(format!("{BASE}/collections/{COLLECTION}"))
        .json(&json!({ "vectors": { "size": b.dim, "distance": "Euclid" } }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!("create collection failed: {}", resp.text().await.unwrap_or_default()));
    }

    // ── Build: bulk upsert + wait until indexed ──────────────────
    let t = Instant::now();
    let batch = 1000usize;
    for chunk_start in (0..b.base.len()).step_by(batch) {
        let end = (chunk_start + batch).min(b.base.len());
        let points: Vec<_> = (chunk_start..end)
            .map(|i| json!({ "id": i as u64, "vector": b.base[i] }))
            .collect();
        let resp = http
            .put(format!("{BASE}/collections/{COLLECTION}/points?wait=true"))
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!("upsert failed: {}", resp.text().await.unwrap_or_default()));
        }
    }
    // Wait for the optimizer to finish indexing all vectors.
    loop {
        let info: serde_json::Value = http
            .get(format!("{BASE}/collections/{COLLECTION}"))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        let res = &info["result"];
        let status = res["status"].as_str().unwrap_or("");
        let indexed = res["indexed_vectors_count"].as_u64().unwrap_or(0);
        if status == "green" && indexed as usize >= b.base.len() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let build_s = t.elapsed().as_secs_f64();

    let mem_mb = process_mem_mb("qdrant").unwrap_or(0.0);
    let disk_mb = env_disk_mb("QDRANT_DATA");

    // ── Query sweep ──────────────────────────────────────────────
    let mut results = Vec::with_capacity(b.queries.len());
    let mut lat = Vec::with_capacity(b.queries.len());
    let t_all = Instant::now();
    for q in &b.queries {
        let t = Instant::now();
        let resp: serde_json::Value = http
            .post(format!("{BASE}/collections/{COLLECTION}/points/search"))
            .json(&json!({ "vector": q, "limit": b.k }))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        lat.push(t.elapsed().as_micros());
        let ids: Vec<String> = resp["result"]
            .as_array()
            .map(|a| a.iter().filter_map(|p| p["id"].as_u64().map(|n| n.to_string())).collect())
            .unwrap_or_default();
        results.push(ids);
    }
    let total_s = t_all.elapsed().as_secs_f64();
    lat.sort_unstable();

    let _ = http.delete(format!("{BASE}/collections/{COLLECTION}")).send().await;
    Ok(Row {
        system: "qdrant".to_string(),
        recall10: recall_at_k(&results, &b.truth, b.k),
        build_s,
        qps: b.queries.len() as f64 / total_s,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
        mem_mb,
        disk_mb,
    })
}
