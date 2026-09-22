//! chroma adapter — native chroma server (pip chromadb) on REST port 8000.
//!
//! Targets the v1 API with explicit tenant/database, which chroma 0.5.x still
//! serves. If a deployment only exposes v2, this adapter reports the error
//! rather than guessing.

use std::time::{Duration, Instant};

use serde_json::json;

use crate::dataset::{percentile, recall_at_k};
use crate::systems::{env_disk_mb, process_mem_mb};
use crate::{Bench, Row};

const BASE: &str = "http://127.0.0.1:8000";

pub fn run(b: &Bench) -> Option<Result<Row, String>> {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(r) => r,
        Err(e) => return Some(Err(e.to_string())),
    };
    Some(rt.block_on(async { run_async(b).await }))
}

async fn run_async(b: &Bench) -> Result<Row, String> {
    let http = reqwest::Client::new();
    let v1 = format!("{BASE}/api/v1");
    let coll_url = format!("{v1}/collections?tenant=default_tenant&database=default_database");

    http.get(format!("{v1}/heartbeat"))
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| format!("chroma unreachable at {BASE}: {e}"))?;

    // Fresh collection (L2 to match SIFT ground truth).
    let _ = http
        .delete(format!(
            "{v1}/collections/bench?tenant=default_tenant&database=default_database"
        ))
        .send()
        .await;
    let created: serde_json::Value = http
        .post(&coll_url)
        .json(&json!({
            "name": "bench",
            // Raise search ef from the default 10 to a production-comparable
            // value so recall is in the same range as the other systems.
            "metadata": { "hnsw:space": "l2", "hnsw:search_ef": 100 }
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let cid = created["id"]
        .as_str()
        .ok_or_else(|| format!("chroma create returned no id: {created}"))?
        .to_string();

    // ── Build: add embeddings in batches ─────────────────────────
    let t = Instant::now();
    let batch = 2000usize;
    for start in (0..b.base.len()).step_by(batch) {
        let end = (start + batch).min(b.base.len());
        let ids: Vec<String> = (start..end).map(|i| i.to_string()).collect();
        let embs: Vec<&Vec<f32>> = (start..end).map(|i| &b.base[i]).collect();
        let resp = http
            .post(format!("{v1}/collections/{cid}/add"))
            .json(&json!({ "ids": ids, "embeddings": embs }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !resp.status().is_success() {
            return Err(format!(
                "chroma add failed: {}",
                resp.text().await.unwrap_or_default()
            ));
        }
    }
    let build_s = t.elapsed().as_secs_f64();

    let mem_mb = process_mem_mb("python").unwrap_or(0.0);
    let disk_mb = env_disk_mb("CHROMA_DATA");

    // ── Query sweep ──────────────────────────────────────────────
    let mut results = Vec::with_capacity(b.queries.len());
    let mut lat = Vec::with_capacity(b.queries.len());
    let t_all = Instant::now();
    for q in &b.queries {
        let t = Instant::now();
        let resp: serde_json::Value = http
            .post(format!("{v1}/collections/{cid}/query"))
            .json(&json!({ "query_embeddings": [q], "n_results": b.k }))
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
        lat.push(t.elapsed().as_micros());
        let ids: Vec<String> = resp["ids"][0]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        results.push(ids);
    }
    let total_s = t_all.elapsed().as_secs_f64();
    lat.sort_unstable();

    Ok(Row {
        system: "chroma".to_string(),
        recall10: recall_at_k(&results, &b.truth, b.k),
        build_s,
        qps: b.queries.len() as f64 / total_s,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
        mem_mb,
        disk_mb,
    })
}
