//! vecdb embedded adapter — no container, in-process.

use std::time::Instant;

use serde_json::json;
use vecdb_core::types::{CollectionConfig, DistanceMetric};
use vecdb_core::Db;

use crate::dataset::{percentile, recall_at_k};
use crate::systems::{dir_size, rss_bytes};
use crate::{Bench, Row};

pub fn run(b: &Bench) -> Result<Row, String> {
    let tmp = std::env::temp_dir().join("vecdb_compare_run");
    let _ = std::fs::remove_dir_all(&tmp);
    let db = Db::open(&tmp).map_err(|e| e.to_string())?;
    let mut cfg = CollectionConfig::new("bench", b.dim);
    cfg.metric = DistanceMetric::Euclidean;
    let mut c = db.create_collection_with(cfg).map_err(|e| e.to_string())?;

    let items: Vec<_> = b
        .base
        .iter()
        .enumerate()
        .map(|(i, v)| (i.to_string(), v.clone(), json!({})))
        .collect();

    let mem_before = rss_bytes();
    let t = Instant::now();
    c.insert_batch(items).map_err(|e| e.to_string())?;
    c.flush().map_err(|e| e.to_string())?;
    let build_s = t.elapsed().as_secs_f64();
    let mem_after = rss_bytes();

    let disk_mb = dir_size(&tmp) as f64 / 1e6;
    let mem_mb = mem_after.saturating_sub(mem_before) as f64 / 1e6;

    // Query sweep.
    let mut results = Vec::with_capacity(b.queries.len());
    let mut lat = Vec::with_capacity(b.queries.len());
    let t_all = Instant::now();
    for q in &b.queries {
        let t = Instant::now();
        let hits = c.query(q, b.k).map_err(|e| e.to_string())?;
        lat.push(t.elapsed().as_micros());
        results.push(hits.into_iter().map(|h| h.id).collect::<Vec<_>>());
    }
    let total_s = t_all.elapsed().as_secs_f64();
    lat.sort_unstable();

    let _ = std::fs::remove_dir_all(&tmp);
    Ok(Row {
        system: "vecdb".to_string(),
        recall10: recall_at_k(&results, &b.truth, b.k),
        build_s,
        qps: b.queries.len() as f64 / total_s,
        p50_us: percentile(&lat, 50.0),
        p99_us: percentile(&lat, 99.0),
        mem_mb,
        disk_mb,
    })
}
