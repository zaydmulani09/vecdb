//! Phase 2 measurement harness: int8 scalar quantization vs float32.
//!
//! Reads a SIFT dataset (fvecs/ivecs) and reports, for each index mode,
//! recall@10 against the dataset ground truth, index memory footprint, and
//! query latency (mean / p50 / p99). The quantization deltas come from the
//! flat-f32 vs flat-int8 rows (same brute-force algorithm, precision is the
//! only difference); the HNSW-f32 row is the production default for context.
//!
//! Usage:
//!   cargo run --release --example quant_bench -- <dir-with-siftsmall_*.fvecs>
//!
//! Dataset (not committed — ~5 MB): the TEXMEX SIFT10K corpus.
//!   curl -O ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
//!   tar xzf siftsmall.tar.gz   # -> siftsmall/siftsmall_{base,query,groundtruth}.*

use std::io::{Read, Result as IoResult};
use std::path::Path;
use std::time::Instant;

use vecdb_core::index::{HnswIndex, IndexBackend, ScalarQuantizedIndex};
use vecdb_core::types::{CollectionConfig, DistanceMetric, Quantization};

fn read_fvecs(path: &Path) -> IoResult<(Vec<Vec<f32>>, usize)> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut dim = 0usize;
    let mut off = 0usize;
    while off < buf.len() {
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        dim = d;
        off += 4;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok((out, dim))
}

fn read_ivecs(path: &Path) -> IoResult<Vec<Vec<u32>>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok(out)
}

/// recall@k = mean over queries of |returned_topk ∩ groundtruth_topk| / k.
fn recall_at_k(results: &[Vec<String>], gt: &[Vec<u32>], k: usize) -> f64 {
    let mut sum = 0.0;
    for (i, res) in results.iter().enumerate() {
        let truth: std::collections::HashSet<u32> = gt[i].iter().take(k).copied().collect();
        let hit = res
            .iter()
            .take(k)
            .filter(|id| truth.contains(&id.parse::<u32>().unwrap()))
            .count();
        sum += hit as f64 / k as f64;
    }
    sum / results.len() as f64
}

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[idx]
}

/// Run all queries against a built index; return (per-query top-k ids, latencies µs).
fn run_queries(
    idx: &dyn IndexBackend,
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<String>>, Vec<u128>) {
    let mut results = Vec::with_capacity(queries.len());
    let mut lat = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let hits = idx.search(q, k).unwrap();
        lat.push(t.elapsed().as_micros());
        results.push(hits.into_iter().map(|(id, _)| id).collect());
    }
    (results, lat)
}

/// Exact full-precision brute-force L2 scan — the ground-truth-matching
/// baseline that isolates quantization loss (same flat algorithm as int8-flat,
/// only the stored precision differs).
fn f32_flat_queries(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<String>>, Vec<u128>) {
    let mut results = Vec::with_capacity(queries.len());
    let mut lat = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        let mut scored: Vec<(usize, f32)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                (i, d)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.truncate(k);
        lat.push(t.elapsed().as_micros());
        results.push(scored.into_iter().map(|(i, _)| i.to_string()).collect());
    }
    (results, lat)
}

struct Row {
    name: &'static str,
    recall10: f64,
    mem_bytes: usize,
    mean_us: f64,
    p50_us: u128,
    p99_us: u128,
}

fn summarize(
    name: &'static str,
    results: &[Vec<String>],
    lat: &mut [u128],
    gt: &[Vec<u32>],
    mem_bytes: usize,
) -> Row {
    lat.sort_unstable();
    let mean = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
    Row {
        name,
        recall10: recall_at_k(results, gt, 10),
        mem_bytes,
        mean_us: mean,
        p50_us: percentile(lat, 50.0),
        p99_us: percentile(lat, 99.0),
    }
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: quant_bench <dir-with-siftsmall_*.fvecs>");
    let dir = Path::new(&dir);

    let (base, dim) = read_fvecs(&dir.join("siftsmall_base.fvecs")).expect("base");
    let (queries, _) = read_fvecs(&dir.join("siftsmall_query.fvecs")).expect("query");
    let gt = read_ivecs(&dir.join("siftsmall_groundtruth.ivecs")).expect("groundtruth");
    let n = base.len();
    println!(
        "dataset: SIFT10K — {n} base × {dim}d, {} queries, {}-NN ground truth\n",
        queries.len(),
        gt[0].len()
    );

    let data: Vec<(String, Vec<f32>)> = base
        .iter()
        .enumerate()
        .map(|(i, v)| (i.to_string(), v.clone()))
        .collect();
    let k = 10;
    let f32_mem = n * dim * std::mem::size_of::<f32>();

    // Config: Euclidean (SIFT's native metric, and the ground truth's).
    let mut cfg = CollectionConfig::new("sift", dim);
    cfg.metric = DistanceMetric::Euclidean;

    // ── int8 flat (scalar quantized) ─────────────────────────────
    let mut sq = ScalarQuantizedIndex::new(&cfg);
    let t = Instant::now();
    sq.build(data.clone()).unwrap();
    let sq_build = t.elapsed();
    let sq_mem = sq.code_bytes();
    let (sq_res, mut sq_lat) = run_queries(&sq, &queries, k);
    let row_sq = summarize("int8-flat", &sq_res, &mut sq_lat, &gt, sq_mem);

    // ── f32 flat exact (isolates quantization: same algo, full precision) ─
    let (flat_res, mut flat_lat) = f32_flat_queries(&base, &queries, k);
    let row_flat = summarize("f32-flat", &flat_res, &mut flat_lat, &gt, f32_mem);

    // ── HNSW f32 (production default, approximate) for context ───────────
    let mut hnsw = HnswIndex::from_collection_config(&cfg);
    let t = Instant::now();
    hnsw.build(data.clone()).unwrap();
    let hnsw_build = t.elapsed();
    let (hnsw_res, mut hnsw_lat) = run_queries(&hnsw, &queries, k);
    // Memory: HNSW holds a full f32 copy (f32_mem) plus graph overhead not
    // counted here — report the f32 vector bytes as a floor.
    let row_hnsw = summarize("f32-hnsw", &hnsw_res, &mut hnsw_lat, &gt, f32_mem);

    let rows = [row_flat, row_sq, row_hnsw];

    println!(
        "{:<16} {:>10} {:>16} {:>12} {:>10} {:>10}",
        "mode", "recall@10", "index bytes", "mean µs", "p50 µs", "p99 µs"
    );
    for r in &rows {
        println!(
            "{:<16} {:>10.4} {:>16} {:>12.1} {:>10} {:>10}",
            r.name, r.recall10, r.mem_bytes, r.mean_us, r.p50_us, r.p99_us
        );
    }
    println!(
        "\nquantization deltas (int8-flat vs f32-flat, same algorithm):"
    );
    println!(
        "  memory:   {} → {} bytes  ({:.2}× smaller)",
        f32_mem,
        sq_mem,
        f32_mem as f64 / sq_mem as f64
    );
    println!(
        "  recall@10 loss: {:.4} (f32-flat is the exact baseline)",
        rows[0].recall10 - rows[1].recall10
    );
    println!(
        "  latency:  mean {:.1}µs → {:.1}µs",
        rows[0].mean_us, rows[1].mean_us
    );
    println!("build time: int8 {:.2?}, hnsw {:.2?}", sq_build, hnsw_build);
    let _ = Quantization::ScalarInt8;
}
