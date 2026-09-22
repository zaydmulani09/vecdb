//! Phase 4 comparison harness.
//!
//! Loads a SIFT dataset once and benchmarks vecdb (embedded, no container)
//! against qdrant / chroma / pgvector (containers) on the same base vectors,
//! the same query set, and the same recall math.
//!
//! Usage:
//!   vecdb-compare <dataset-dir> [--n N] [--queries Q] [--systems a,b,...]
//!
//! `<dataset-dir>` holds `sift_base.fvecs` + `sift_query.fvecs` (+ optional
//! `sift_groundtruth.ivecs`), or the `siftsmall_*` equivalents. When N is less
//! than the full base, exact ground truth is recomputed on the subset.

mod dataset;
mod systems;

use std::path::{Path, PathBuf};
use std::time::Instant;

use dataset::*;

/// One system's measured row.
pub struct Row {
    pub system: String,
    pub recall10: f64,
    pub build_s: f64,
    pub qps: f64,
    pub p50_us: u128,
    pub p99_us: u128,
    pub mem_mb: f64,
    pub disk_mb: f64,
}

pub struct Bench {
    pub base: Vec<Vec<f32>>,
    pub dim: usize,
    pub queries: Vec<Vec<f32>>,
    pub truth: Vec<Vec<String>>,
    pub k: usize,
}

fn find(dir: &Path, names: &[&str]) -> Option<PathBuf> {
    names.iter().map(|n| dir.join(n)).find(|p| p.exists())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .expect("usage: vecdb-compare <dataset-dir> [--n N] [--queries Q] [--systems ...]");
    let dir = PathBuf::from(dir);
    let mut n = 100_000usize;
    let mut nq = 1_000usize;
    let mut systems = vec!["vecdb".to_string()];
    while let Some(a) = args.next() {
        match a.as_str() {
            "--n" => n = args.next().unwrap().parse().unwrap(),
            "--queries" => nq = args.next().unwrap().parse().unwrap(),
            "--systems" => {
                systems = args
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|s| s.to_string())
                    .collect()
            }
            other => panic!("unknown arg {other}"),
        }
    }

    let base_path = find(&dir, &["sift_base.fvecs", "siftsmall_base.fvecs"]).expect("base fvecs");
    let query_path =
        find(&dir, &["sift_query.fvecs", "siftsmall_query.fvecs"]).expect("query fvecs");
    let gt_path = find(
        &dir,
        &["sift_groundtruth.ivecs", "siftsmall_groundtruth.ivecs"],
    );

    println!("loading base (first {n})...");
    let (base, dim) = read_fvecs(&base_path, Some(n)).expect("base");
    let n = base.len();
    let (all_q, _) = read_fvecs(&query_path, Some(nq)).expect("query");
    let queries: Vec<Vec<f32>> = all_q.into_iter().take(nq).collect();
    let k = 10;

    // Ground truth: dataset GT is valid only when the whole base is loaded (its
    // indices reference the full base). Otherwise recompute exact GT on the
    // subset.
    let full_base = base_path.metadata().map(|m| m.len()).unwrap_or(0) as usize
        == n.saturating_mul(dim * 4 + 4);
    let truth: Vec<Vec<String>> = if let (true, Some(gt_p)) = (full_base, gt_path.as_ref()) {
        println!("using dataset ground truth");
        let gt = read_ivecs(gt_p, Some(queries.len())).expect("gt");
        gt.into_iter()
            .map(|row| row.into_iter().take(k).map(|i| i.to_string()).collect())
            .collect()
    } else {
        println!(
            "recomputing exact ground truth on the {n}-vector subset ({} queries)...",
            queries.len()
        );
        let t = Instant::now();
        let g = exact_ground_truth(&base, &queries, k);
        println!("  exact GT in {:.1?}", t.elapsed());
        g
    };

    let bench = Bench {
        base,
        dim,
        queries,
        truth,
        k,
    };
    println!(
        "\ndataset: {n} × {dim}d base, {} queries, k={k}\n",
        bench.queries.len()
    );

    let mut rows: Vec<Row> = Vec::new();
    for sys in &systems {
        println!("── {sys} ──");
        let row = match sys.as_str() {
            "vecdb" => Some(systems::vecdb::run(&bench)),
            "qdrant" => systems::qdrant::run(&bench),
            "chroma" => systems::chroma::run(&bench),
            "pgvector" => systems::pgvector::run(&bench),
            other => {
                eprintln!("unknown system {other}, skipping");
                None
            }
        };
        match row {
            Some(Ok(r)) => {
                println!(
                    "  recall@10={:.4} build={:.1}s qps={:.0} p50={}µs p99={}µs mem={:.0}MB disk={:.0}MB\n",
                    r.recall10, r.build_s, r.qps, r.p50_us, r.p99_us, r.mem_mb, r.disk_mb
                );
                rows.push(r);
            }
            Some(Err(e)) => eprintln!("  {sys} failed: {e}\n"),
            None => eprintln!("  {sys} skipped\n"),
        }
    }

    println!(
        "{:<10} {:>10} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "system", "recall@10", "build s", "qps", "p50 µs", "p99 µs", "mem MB", "disk MB"
    );
    for r in &rows {
        println!(
            "{:<10} {:>10.4} {:>9.1} {:>9.0} {:>9} {:>9} {:>9.0} {:>9.0}",
            r.system, r.recall10, r.build_s, r.qps, r.p50_us, r.p99_us, r.mem_mb, r.disk_mb
        );
    }
}
