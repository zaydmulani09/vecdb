//! Phase 1 proof: the whole embedded loop with zero setup, using only the
//! public `vecdb_core` API — no server, no HTTP, no docker. This is the test
//! that makes the "no server required" claim real.
//!
//! Create a db file/dir, insert N vectors + metadata, query, assert recall,
//! close (drop), reopen, confirm the data persisted.

use serde_json::json;
use vecdb_core::Db;

/// N distinct unit-ish vectors in D dims; vector i is the i-th basis direction
/// scaled, so the nearest neighbour of `query(i)` is unambiguously record i.
const N: usize = 200;
const D: usize = 16;

fn vec_for(i: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; D];
    v[i % D] = 1.0 + (i as f32) * 0.01; // dominant axis + small unique magnitude
    v[(i + 1) % D] = 0.25;
    v
}

#[test]
fn embedded_full_loop_insert_query_persist_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("app.vecdb");

    // ── Insert phase ─────────────────────────────────────────────
    {
        let db = Db::open(&path).unwrap();
        let mut docs = db.create_collection("docs", D).unwrap();
        for i in 0..N {
            docs.insert(format!("doc{i}"), vec_for(i), json!({ "i": i }))
                .unwrap();
        }
        assert_eq!(docs.len().unwrap(), N);

        // Recall@1 on a sample of queries: each query vector must return its
        // own record as the top hit.
        let mut correct = 0;
        for i in (0..N).step_by(7) {
            let hits = docs.query(&vec_for(i), 1).unwrap();
            if hits.first().map(|h| h.id.as_str()) == Some(&format!("doc{i}")[..]) {
                correct += 1;
            }
        }
        let total = (0..N).step_by(7).count();
        assert_eq!(correct, total, "recall@1 must be perfect on exact queries");

        docs.flush().unwrap();
        // db + docs dropped here — "close"
    }

    // ── Reopen phase (fresh handles, same files) ─────────────────
    {
        let db = Db::open(&path).unwrap();
        assert_eq!(db.list_collections().unwrap(), vec!["docs".to_string()]);

        let docs = db.collection("docs").unwrap();
        assert_eq!(docs.len().unwrap(), N, "all vectors must survive reopen");

        // Data + payload persisted and still queryable.
        let hits = docs.query(&vec_for(42), 3).unwrap();
        assert_eq!(hits[0].id, "doc42");
        assert_eq!(hits[0].payload["i"], 42);
    }
}

#[test]
fn embedded_delete_persists_across_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("del.vecdb");

    {
        let db = Db::open(&path).unwrap();
        let mut c = db.create_collection("c", 4).unwrap();
        c.insert("keep", vec![1.0, 0.0, 0.0, 0.0], json!({}))
            .unwrap();
        c.insert("gone", vec![0.0, 1.0, 0.0, 0.0], json!({}))
            .unwrap();
        c.delete("gone").unwrap();
        assert_eq!(c.len().unwrap(), 1);
        c.flush().unwrap();
    }

    let db = Db::open(&path).unwrap();
    let c = db.collection("c").unwrap();
    assert_eq!(c.len().unwrap(), 1, "delete must persist across reopen");
    let hits = c.query(&[0.0, 1.0, 0.0, 0.0], 5).unwrap();
    assert!(
        hits.iter().all(|h| h.id != "gone"),
        "deleted record must not resurface"
    );
}
