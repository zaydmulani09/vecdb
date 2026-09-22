//! Phase 3 correctness: predicate-pushdown filtered search returns exactly the
//! true nearest neighbors *among the matching records* — same result as an
//! exact brute-force over the filtered subset — and never leaks non-matches.

use serde_json::json;
use vecdb_core::types::{CollectionConfig, DistanceMetric};
use vecdb_core::Db;

fn euclidean_collection(db: &Db, name: &str, dim: usize) -> vecdb_core::Collection {
    let mut cfg = CollectionConfig::new(name, dim);
    cfg.metric = DistanceMetric::Euclidean;
    db.create_collection_with(cfg).unwrap()
}

fn vec_for(i: usize, d: usize) -> Vec<f32> {
    // Spread across the space so nearest-neighbor order is well-defined.
    let mut v = vec![0.0f32; d];
    for (j, slot) in v.iter_mut().enumerate() {
        *slot = ((i * 7 + j * 13) % 97) as f32 + (i as f32) * 0.01;
    }
    v
}

/// Exact top-k L2 over the records whose bucket == target (the reference the
/// pushdown path must reproduce).
fn exact_filtered_topk(
    data: &[(String, Vec<f32>, i64)],
    query: &[f32],
    bucket: i64,
    k: usize,
) -> Vec<String> {
    let mut scored: Vec<(String, f32)> = data
        .iter()
        .filter(|(_, _, b)| *b == bucket)
        .map(|(id, v, _)| {
            let d: f32 = query.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
            (id.clone(), d)
        })
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(id, _)| id).collect()
}

#[test]
fn pushdown_matches_exact_filtered_bruteforce() {
    let dir = tempfile::tempdir().unwrap();
    let d = 12;
    let n = 1000;
    let buckets = 20i64; // ~5% per bucket → 95% eliminated by an equality filter

    let mut data: Vec<(String, Vec<f32>, i64)> = Vec::new();
    let db = Db::open(dir.path().join("f.vecdb")).unwrap();
    let mut c = euclidean_collection(&db, "docs", d);
    for i in 0..n {
        let bucket = (i as i64) % buckets;
        let v = vec_for(i, d);
        c.insert(format!("doc{i}"), v.clone(), json!({ "bucket": bucket }))
            .unwrap();
        data.push((format!("doc{i}"), v, bucket));
    }

    // High-selectivity filter (one bucket ≈ 5% of the corpus).
    let target = 3i64;
    let query = vec_for(500, d);
    let k = 10;

    let hits = c.query_filtered(&query, k, &json!({ "bucket": target })).unwrap();

    // 1. No non-matches leak through.
    for h in &hits {
        assert_eq!(h.payload["bucket"], target, "non-matching record {} leaked", h.id);
    }
    // 2. Exactly the true nearest neighbors among the matches.
    let expect = exact_filtered_topk(&data, &query, target, k);
    let got: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(got, expect, "pushdown result != exact filtered brute-force");
}

#[test]
fn pushdown_with_payload_index_matches_exact() {
    // Same as above but with a declared payload index on `bucket`, exercising
    // the index-friendly SQL path + ensure_payload_indexes. Result must be
    // identical to the exact filtered brute-force.
    let dir = tempfile::tempdir().unwrap();
    let d = 12;
    let n = 1000;
    let buckets = 20i64;

    let mut data: Vec<(String, Vec<f32>, i64)> = Vec::new();
    let db = Db::open(dir.path().join("fi.vecdb")).unwrap();
    let mut cfg = CollectionConfig::new("docs", d);
    cfg.metric = DistanceMetric::Euclidean;
    let mut c = db
        .create_collection_with(cfg.with_indexed_fields(["bucket"]))
        .unwrap();
    for i in 0..n {
        let bucket = (i as i64) % buckets;
        let v = vec_for(i, d);
        c.insert(format!("doc{i}"), v.clone(), json!({ "bucket": bucket }))
            .unwrap();
        data.push((format!("doc{i}"), v, bucket));
    }

    let target = 7i64;
    let query = vec_for(321, d);
    let k = 10;
    let hits = c.query_filtered(&query, k, &json!({ "bucket": target })).unwrap();
    for h in &hits {
        assert_eq!(h.payload["bucket"], target);
    }
    let got: Vec<String> = hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(got, exact_filtered_topk(&data, &query, target, k));
}

#[test]
fn empty_and_full_filters() {
    let dir = tempfile::tempdir().unwrap();
    let d = 8;
    let db = Db::open(dir.path().join("e.vecdb")).unwrap();
    let mut c = euclidean_collection(&db, "docs", d);
    for i in 0..200 {
        c.insert(format!("d{i}"), vec_for(i, d), json!({ "bucket": i % 4 }))
            .unwrap();
    }
    // Filter matching nothing → empty.
    let none = c.query_filtered(&vec_for(1, d), 5, &json!({ "bucket": 999 })).unwrap();
    assert!(none.is_empty(), "impossible filter must return no results");

    // Weakly selective filter (25% match) still returns only matches.
    let some = c.query_filtered(&vec_for(1, d), 5, &json!({ "bucket": 1 })).unwrap();
    assert!(!some.is_empty());
    assert!(some.iter().all(|h| h.payload["bucket"] == 1));
}
