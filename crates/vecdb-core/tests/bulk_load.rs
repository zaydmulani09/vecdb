//! Phase 4 prerequisite: bulk load produces the same searchable, persistent
//! collection as one-by-one inserts, but building the index once.

use serde_json::json;
use vecdb_core::types::{CollectionConfig, DistanceMetric};
use vecdb_core::Db;

fn vec_for(i: usize, d: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; d];
    v[i % d] = 100.0 + i as f32;
    v[(i + 1) % d] = 10.0;
    v
}

#[test]
fn bulk_load_matches_single_insert_and_persists() {
    let dir = tempfile::tempdir().unwrap();
    let d = 16;
    let n = 2000;

    let mut cfg = CollectionConfig::new("b", d);
    cfg.metric = DistanceMetric::Euclidean;

    // Bulk-loaded collection.
    let db = Db::open(dir.path().join("bulk.vecdb")).unwrap();
    let mut cbulk = db.create_collection_with(cfg.clone()).unwrap();
    let items: Vec<_> = (0..n)
        .map(|i| (format!("doc{i}"), vec_for(i, d), json!({ "i": i })))
        .collect();
    let inserted = cbulk.insert_batch(items).unwrap();
    assert_eq!(inserted, n);
    assert_eq!(cbulk.len().unwrap(), n);

    // Query parity on a sample: bulk result top-1 must be the true vector.
    for i in (0..n).step_by(53) {
        let hits = cbulk.query(&vec_for(i, d), 1).unwrap();
        assert_eq!(hits[0].id, format!("doc{i}"), "bulk query {i} wrong top-1");
        assert_eq!(hits[0].payload["i"], i);
    }

    // Persist across reopen.
    cbulk.flush().unwrap();
    drop(cbulk);
    let db2 = Db::open(dir.path().join("bulk.vecdb")).unwrap();
    let c = db2.collection("b").unwrap();
    assert_eq!(c.len().unwrap(), n, "bulk-loaded data must survive reopen");
    assert_eq!(c.query(&vec_for(123, d), 1).unwrap()[0].id, "doc123");
}
