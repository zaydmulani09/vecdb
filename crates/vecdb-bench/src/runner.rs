use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Instant;

use crate::client::BenchClient;
use crate::datasets::{generate_synthetic, load_jsonl};
use crate::types::{
    CreateCollectionRequest, DenseSearchRequest, SearchResponse, UpsertRecord, UpsertRequest,
    UpsertResponse,
};

// ── Dataset kind ──────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum DatasetKind {
    Synthetic { n: usize, dim: usize, seed: u64 },
    Jsonl { path: PathBuf },
}

// ── Search type ───────────────────────────────────────────────────────────

#[derive(Debug)]
#[allow(dead_code)]
pub enum SearchType {
    Dense,
    Sparse,
    Hybrid,
}

// ── Config and result ─────────────────────────────────────────────────────

pub struct BenchmarkConfig {
    pub dataset: DatasetKind,
    pub index_type: String,
    pub search_type: SearchType,
    pub query_count: usize,
    pub k: usize,
    pub alpha: f32,
    pub collection: String,
    pub batch_size: usize,
    pub server_url: String,
    pub api_key: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub dataset_size: usize,
    pub query_count: usize,
    pub k: usize,
    pub recall_at_k: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub mean_ms: f64,
    pub qps: f64,
    pub total_bench_time_ms: f64,
}

// ── Pure compute helpers ──────────────────────────────────────────────────

pub fn compute_recall(ground_truth: &[String], server_results: &[String], k: usize) -> f64 {
    let gt_set: std::collections::HashSet<&str> =
        ground_truth.iter().take(k).map(String::as_str).collect();
    let hits = server_results
        .iter()
        .take(k)
        .filter(|id| gt_set.contains(id.as_str()))
        .count();
    hits as f64 / k as f64
}

pub fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (sorted.len() - 1) * pct / 100;
    sorted[idx]
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

fn make_progress_bar(len: u64, msg: &'static str) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_style(
        ProgressStyle::with_template("[{bar:40}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-"),
    );
    pb.set_message(msg);
    pb
}

// ── Runner ────────────────────────────────────────────────────────────────

pub struct BenchmarkRunner;

impl BenchmarkRunner {
    pub async fn run(config: BenchmarkConfig) -> Result<BenchmarkResult> {
        // Destructure — fields not yet used in implementation are prefixed with _
        let BenchmarkConfig {
            dataset,
            index_type: _index_type,
            search_type: _search_type,
            query_count,
            k,
            alpha: _alpha,
            collection,
            batch_size,
            server_url,
            api_key,
        } = config;

        // Step 1 — Load dataset
        let corpus: Vec<(String, Vec<f32>)> = match dataset {
            DatasetKind::Synthetic { n, dim, seed } => generate_synthetic(n, dim, seed),
            DatasetKind::Jsonl { ref path } => load_jsonl(path)?,
        };

        if corpus.is_empty() {
            return Err(anyhow::anyhow!("dataset is empty"));
        }

        let dim = corpus[0].1.len();
        let query_start = corpus.len().saturating_sub(query_count);
        let queries: Vec<(String, Vec<f32>)> = corpus[query_start..].to_vec();
        let actual_query_count = queries.len();

        if actual_query_count == 0 {
            return Err(anyhow::anyhow!("no query vectors available"));
        }

        tracing::info!(
            corpus = corpus.len(),
            queries = actual_query_count,
            dim,
            k,
            "dataset loaded"
        );

        // Step 2 — Create collection (ignore error if it already exists)
        let client = BenchClient::new(&server_url, &api_key)?;
        let create_req = CreateCollectionRequest {
            name: collection.clone(),
            dimension: dim,
            metric: Some("cosine".to_string()),
        };
        let create_result: Result<serde_json::Value> =
            client.post("/collections", &create_req).await;
        if let Err(ref e) = create_result {
            tracing::warn!(
                "collection creation returned an error (may already exist): {}",
                e
            );
        }

        // Step 3 — Ingest corpus in batches
        let pb = make_progress_bar(corpus.len() as u64, "ingesting");
        for chunk in corpus.chunks(batch_size) {
            let records: Vec<UpsertRecord> = chunk
                .iter()
                .map(|(id, v)| UpsertRecord {
                    id: id.clone(),
                    vector: v.clone(),
                    text: None,
                    payload: None,
                })
                .collect();
            let body = UpsertRequest { records };
            let _resp: UpsertResponse = client
                .post(&format!("/collections/{}/vectors", collection), &body)
                .await?;
            pb.inc(chunk.len() as u64);
        }
        pb.finish_with_message("ingest complete");
        tracing::info!(vectors = corpus.len(), "ingest complete");

        // Step 4 — Brute-force ground truth (rayon parallelism)
        let ground_truth: Vec<Vec<String>> = queries
            .par_iter()
            .map(|(_, qv)| {
                let mut scored: Vec<(f32, &str)> = corpus
                    .iter()
                    .map(|(id, cv)| (cosine_similarity(qv, cv), id.as_str()))
                    .collect();
                scored.sort_by(|a, b| b.0.total_cmp(&a.0));
                scored
                    .iter()
                    .take(k)
                    .map(|(_, id)| id.to_string())
                    .collect()
            })
            .collect();

        // Step 5 — Warm-up (10% of queries, minimum 1)
        let warmup_count = (actual_query_count / 10).max(1);
        for (_, qv) in queries.iter().take(warmup_count) {
            let req = DenseSearchRequest {
                vector: qv.clone(),
                k,
            };
            let _: SearchResponse = client
                .post(&format!("/collections/{}/search/dense", collection), &req)
                .await?;
        }

        // Step 6 — Benchmark loop
        let pb2 = make_progress_bar(actual_query_count as u64, "benchmarking");
        let bench_start = Instant::now();
        let mut latencies: Vec<f64> = Vec::with_capacity(actual_query_count);
        let mut recalls: Vec<f64> = Vec::with_capacity(actual_query_count);

        for (i, (_, qv)) in queries.iter().enumerate() {
            let req = DenseSearchRequest {
                vector: qv.clone(),
                k,
            };
            let t0 = Instant::now();
            let resp: SearchResponse = client
                .post(&format!("/collections/{}/search/dense", collection), &req)
                .await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let server_ids: Vec<String> = resp.results.iter().map(|r| r.id.clone()).collect();
            let recall = compute_recall(&ground_truth[i], &server_ids, k);

            latencies.push(elapsed_ms);
            recalls.push(recall);
            pb2.inc(1);
        }
        let total_bench_time_ms = bench_start.elapsed().as_secs_f64() * 1000.0;
        pb2.finish_with_message("benchmark complete");

        // Step 7 — Compute stats
        latencies.sort_by(|a, b| a.total_cmp(b));
        let n = latencies.len();
        let p50 = percentile(&latencies, 50);
        let p95 = percentile(&latencies, 95);
        let p99 = percentile(&latencies, 99);
        let mean_ms = latencies.iter().sum::<f64>() / n as f64;
        let qps = actual_query_count as f64 / (total_bench_time_ms / 1000.0);
        let recall_at_k = recalls.iter().sum::<f64>() / recalls.len() as f64;

        // Step 8 — Delete collection (best-effort cleanup)
        let delete_result: Result<serde_json::Value> = client
            .delete_no_body(&format!("/collections/{}", collection))
            .await;
        if let Err(e) = delete_result {
            tracing::warn!("failed to delete collection '{}': {}", collection, e);
        }

        Ok(BenchmarkResult {
            dataset_size: corpus.len(),
            query_count: actual_query_count,
            k,
            recall_at_k,
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            mean_ms,
            qps,
            total_bench_time_ms,
        })
    }
}
