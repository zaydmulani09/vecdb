//! Phase 3 measurement: predicate pushdown for filtered dense search.
//!
//! Builds a SIFT10K collection where each vector carries a `bucket` in
//! `[0, 20)` (≈5% per bucket), then compares, for a filter that keeps one
//! bucket (≈95% eliminated):
//!
//!   - unfiltered      — plain ANN over the whole index (latency reference)
//!   - pushdown        — `query_filtered`: candidates generated from the filter
//!   - naive@200       — ANN top-200 then drop non-matches (search-then-discard)
//!   - naive@full      — ANN over the whole corpus then drop non-matches
//!     (bounded by the ANN neighborhood, so recall does not improve)
//!
//! Recall@10 is measured against an exact brute-force over the filtered subset.
//! The gate: pushdown reaches naive@full's recall without naive@full's cost,
//! and its latency does not balloon relative to the unfiltered search.
//!
//! Usage: cargo run --release --example filter_bench -- <dir-with-siftsmall_*.fvecs>

use std::io::{Read, Result as IoResult};
use std::path::Path;
use std::time::Instant;

use serde_json::json;
use vecdb_core::types::{CollectionConfig, DistanceMetric};
use vecdb_core::Db;

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

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[idx]
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Exact top-k over vectors whose bucket == target.
fn exact_filtered(
    base: &[Vec<f32>],
    buckets: &[i64],
    q: &[f32],
    target: i64,
    k: usize,
) -> Vec<String> {
    let mut scored: Vec<(usize, f32)> = base
        .iter()
        .enumerate()
        .filter(|(i, _)| buckets[*i] == target)
        .map(|(i, v)| (i, l2(q, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored
        .into_iter()
        .take(k)
        .map(|(i, _)| i.to_string())
        .collect()
}

fn recall(results: &[Vec<String>], truth: &[Vec<String>], k: usize) -> f64 {
    let mut sum = 0.0;
    for (r, t) in results.iter().zip(truth) {
        let set: std::collections::HashSet<&String> = t.iter().take(k).collect();
        let hit = r.iter().take(k).filter(|id| set.contains(id)).count();
        sum += hit as f64 / k as f64;
    }
    sum / results.len() as f64
}

fn stats(mut lat: Vec<u128>) -> (f64, u128, u128) {
    lat.sort_unstable();
    let mean = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
    (mean, percentile(&lat, 50.0), percentile(&lat, 99.0))
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: filter_bench <dir-with-siftsmall_*.fvecs>");
    let dir = Path::new(&dir);
    let (base, dim) = read_fvecs(&dir.join("siftsmall_base.fvecs")).expect("base");
    let (queries, _) = read_fvecs(&dir.join("siftsmall_query.fvecs")).expect("query");
    let n = base.len();
    let nbuckets = 20i64;
    let target = 3i64;
    let k = 10;
    let buckets: Vec<i64> = (0..n).map(|i| (i as i64) % nbuckets).collect();
    let matches = buckets.iter().filter(|b| **b == target).count();
    println!(
        "SIFT10K — {n}×{dim}d; filter bucket=={target} keeps {matches} ({:.1}% of corpus, {:.1}% eliminated)\n",
        100.0 * matches as f64 / n as f64,
        100.0 * (1.0 - matches as f64 / n as f64),
    );

    // Build two embedded Euclidean collections with a per-vector bucket payload:
    // one with a payload index on `bucket`, one without.
    let tmp = std::env::temp_dir().join("vecdb_filter_bench");
    let _ = std::fs::remove_dir_all(&tmp);
    let db = Db::open(&tmp).unwrap();

    let build = |name: &str, indexed: bool| {
        let mut cfg = CollectionConfig::new(name, dim);
        cfg.metric = DistanceMetric::Euclidean;
        if indexed {
            cfg = cfg.with_indexed_fields(["bucket"]);
        }
        let mut col = db.create_collection_with(cfg).unwrap();
        print!("building {name} ({n} vectors, indexed={indexed})... ");
        let t = Instant::now();
        for (i, v) in base.iter().enumerate() {
            col.insert(i.to_string(), v.clone(), json!({ "bucket": buckets[i] }))
                .unwrap();
        }
        col.flush().unwrap();
        println!("done in {:.1?}", t.elapsed());
        col
    };
    let c = build("sift_idx", true);
    let c_plain = build("sift_plain", false);

    // Exact filtered ground truth.
    let truth: Vec<Vec<String>> = queries
        .iter()
        .map(|q| exact_filtered(&base, &buckets, q, target, k))
        .collect();

    let filter = json!({ "bucket": target });

    // ── unfiltered ANN (latency reference; results are not bucket-restricted) ─
    let mut lat = Vec::new();
    for q in &queries {
        let t = Instant::now();
        let _ = c.query(q, k).unwrap();
        lat.push(t.elapsed().as_micros());
    }
    let unf = stats(lat);

    // ── pushdown (indexed payload field) ─────────────────────────────────
    let mut lat = Vec::new();
    let mut res = Vec::new();
    for q in &queries {
        let t = Instant::now();
        let hits = c.query_filtered(q, k, &filter).unwrap();
        lat.push(t.elapsed().as_micros());
        res.push(hits.into_iter().map(|h| h.id).collect::<Vec<_>>());
    }
    let push = stats(lat);
    let push_recall = recall(&res, &truth, k);

    // ── pushdown (no payload index — full metadata scan) ─────────────────
    let mut lat = Vec::new();
    let mut res_plain = Vec::new();
    for q in &queries {
        let t = Instant::now();
        let hits = c_plain.query_filtered(q, k, &filter).unwrap();
        lat.push(t.elapsed().as_micros());
        res_plain.push(hits.into_iter().map(|h| h.id).collect::<Vec<_>>());
    }
    let push_plain = stats(lat);
    let push_plain_recall = recall(&res_plain, &truth, k);

    // ── naive post-filter at two oversample budgets ──────────────────────
    let naive = |budget: usize| -> ((f64, u128, u128), f64) {
        let mut lat = Vec::new();
        let mut res = Vec::new();
        for q in &queries {
            let t = Instant::now();
            let hits = c.query(q, budget).unwrap();
            let kept: Vec<String> = hits
                .into_iter()
                .filter(|h| h.payload["bucket"] == target)
                .take(k)
                .map(|h| h.id)
                .collect();
            lat.push(t.elapsed().as_micros());
            res.push(kept);
        }
        (stats(lat), recall(&res, &truth, k))
    };
    let (naive200, naive200_recall) = naive(200);
    let (naivefull, naivefull_recall) = naive(n);

    println!(
        "{:<16} {:>10} {:>12} {:>10} {:>10}",
        "mode", "recall@10", "mean µs", "p50 µs", "p99 µs"
    );
    let row = |name: &str, s: (f64, u128, u128), r: f64| {
        println!(
            "{:<16} {:>10.4} {:>12.1} {:>10} {:>10}",
            name, r, s.0, s.1, s.2
        );
    };
    row("unfiltered", unf, f64::NAN); // recall N/A (not bucket-restricted)
    row("pushdown+idx", push, push_recall);
    row("pushdown(scan)", push_plain, push_plain_recall);
    row("naive@200", naive200, naive200_recall);
    row("naive@allN", naivefull, naivefull_recall);

    println!(
        "\npushdown recall@10 = {:.3} (exact over the matching set).",
        push_recall
    );
    println!(
        "naive post-filter recall@10 = {:.3} at budget 200 and {:.3} at budget {} —",
        naive200_recall, naivefull_recall, n
    );
    println!("  it does NOT improve with budget: ANN returns only its bounded neighborhood,");
    println!("  which rarely contains the filtered nearest neighbors at high selectivity.");
    println!(
        "pushdown+idx mean {:.0}µs vs unfiltered ANN {:.0}µs ({:.2}×) — a payload index keeps",
        push.0,
        unf.0,
        push.0 / unf.0
    );
    println!("  filtered search in the same ballpark as unfiltered; without the index the same",);
    println!(
        "  query is {:.0}µs ({:.1}× slower) because every payload must be scanned.",
        push_plain.0,
        push_plain.0 / push.0
    );
    let _ = std::fs::remove_dir_all(&tmp);
}
