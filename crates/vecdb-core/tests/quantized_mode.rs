//! Phase 2 proof: int8 scalar quantization is a real, opt-in per-collection
//! storage mode that works through the public embedded API and survives reopen,
//! while the default float32 path is untouched.

use serde_json::json;
use vecdb_core::Db;

/// Deterministic vector: dominant axis `i % D`, unique magnitude.
fn vec_for(i: usize, d: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; d];
    v[i % d] = 100.0 + (i as f32);
    v[(i + 1) % d] = 25.0;
    v
}

#[test]
fn quantized_collection_insert_query_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("q.vecdb");
    let d = 16;
    let n = 400; // > TRAIN_THRESHOLD so the quantizer trains and codes materialize

    let mut before_top: Vec<(usize, String)> = Vec::new();
    {
        let db = Db::open(&path).unwrap();
        let mut c = db.create_collection_quantized("docs", d).unwrap();
        for i in 0..n {
            c.insert(format!("doc{i}"), vec_for(i, d), json!({ "i": i }))
                .unwrap();
        }
        assert_eq!(c.len().unwrap(), n);

        // Recall@1 on exact queries. Quantization is lossy, so this is high but
        // not perfect — assert a sane floor rather than losslessness.
        let mut correct = 0;
        let mut total = 0;
        for i in (0..n).step_by(11) {
            let hits = c.query(&vec_for(i, d), 1).unwrap();
            before_top.push((i, hits[0].id.clone()));
            if hits[0].id == format!("doc{i}") {
                correct += 1;
            }
            total += 1;
        }
        assert!(
            correct as f64 / total as f64 >= 0.8,
            "quantized recall@1 {correct}/{total} below 0.8 floor"
        );

        c.flush().unwrap();
    }

    // Reopen: quantized index persisted; results must be identical to pre-close.
    let db = Db::open(&path).unwrap();
    let c = db.collection("docs").unwrap();
    assert_eq!(c.len().unwrap(), n, "quantized vectors must survive reopen");
    for (i, top_id) in &before_top {
        let hits = c.query(&vec_for(*i, d), 1).unwrap();
        assert_eq!(&hits[0].id, top_id, "top-1 for query {i} changed across reopen");
    }
}

#[test]
fn binary_collection_persists_and_is_stable_across_reopen() {
    // Binary (1-bit) mode is coarse, so we assert wiring + persistence + result
    // stability, not recall. Recall/rerank numbers live in the benchmark.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("b.vecdb");
    let d = 16;
    let n = 400;

    let mut before_top: Vec<(usize, String)> = Vec::new();
    {
        let db = Db::open(&path).unwrap();
        let mut c = db.create_collection_binary("docs", d).unwrap();
        for i in 0..n {
            c.insert(format!("doc{i}"), vec_for(i, d), json!({ "i": i }))
                .unwrap();
        }
        assert_eq!(c.len().unwrap(), n);
        for i in (0..n).step_by(37) {
            let hits = c.query(&vec_for(i, d), 1).unwrap();
            assert!(!hits.is_empty());
            before_top.push((i, hits[0].id.clone()));
        }
        c.flush().unwrap();
    }

    let db = Db::open(&path).unwrap();
    let c = db.collection("docs").unwrap();
    assert_eq!(c.len().unwrap(), n, "binary vectors must survive reopen");
    for (i, top_id) in &before_top {
        let hits = c.query(&vec_for(*i, d), 1).unwrap();
        assert_eq!(&hits[0].id, top_id, "binary top-1 for query {i} changed across reopen");
    }
}

#[test]
fn default_mode_is_unquantized_and_still_works() {
    // Same data through the default (float32) path — Phase 1 behavior intact.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.vecdb");
    let db = Db::open(&path).unwrap();
    let mut c = db.create_collection("docs", 8).unwrap();
    for i in 0..50 {
        c.insert(format!("d{i}"), vec_for(i, 8), json!({})).unwrap();
    }
    let hits = c.query(&vec_for(7, 8), 1).unwrap();
    assert_eq!(hits[0].id, "d7");
}
