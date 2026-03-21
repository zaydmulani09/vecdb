#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    use crate::index::backend::IndexBackend;
    use crate::index::ivf::IvfIndex;
    use crate::index::AnyIndex;
    use crate::storage::Storage;
    use crate::types::{CollectionConfig, IndexType, VectorRecord};

    fn ivf_config(name: &str, dim: usize) -> CollectionConfig {
        let mut cfg = CollectionConfig::new(name, dim);
        cfg.index_type = IndexType::IVF;
        cfg
    }

    fn dummy_record(id: &str, vector: Vec<f32>) -> VectorRecord {
        VectorRecord {
            id: id.to_string(),
            vector,
            payload: json!({ "id": id }),
            text: Some(format!("text for {id}")),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // ── Test 1 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_insert_and_search_basic() {
        let cfg = ivf_config("basic", 4);
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..20usize {
            idx.insert(format!("v{i}"), vec![i as f32, 1.0, 0.0, 0.0])
                .unwrap();
        }

        // Query close to v10
        let results = idx.search(&vec![10.0, 1.0, 0.0, 0.0], 5).unwrap();
        assert!(!results.is_empty(), "expected at least 1 result");
        assert!(
            results.iter().any(|(_, s)| *s > 0.0),
            "expected score > 0.0"
        );
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_search_returns_at_most_k() {
        let cfg = ivf_config("topk", 4);
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..50usize {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0, 0.0, 1.0])
                .unwrap();
        }

        let results = idx.search(&vec![25.0, 0.0, 0.0, 1.0], 5).unwrap();
        assert!(
            results.len() <= 5,
            "expected <= 5 results, got {}",
            results.len()
        );
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_delete_excludes_from_results() {
        let cfg = ivf_config("del", 4);
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..10usize {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0, 1.0, 0.0])
                .unwrap();
        }

        idx.delete(&"v5".to_string()).unwrap();

        let results = idx.search(&vec![5.0, 0.0, 1.0, 0.0], 10).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            !ids.contains(&"v5"),
            "deleted id 'v5' must not appear in results"
        );
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_rebuild_after_inserts() {
        let cfg = ivf_config("rebuild", 4);
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..100usize {
            idx.insert(format!("v{i}"), vec![i as f32, 1.0, 0.0, 0.5])
                .unwrap();
        }

        // Trigger explicit rebuild via the IndexBackend trait
        let all: Vec<(String, Vec<f32>)> = (0..100usize)
            .map(|i| (format!("v{i}"), vec![i as f32, 1.0, 0.0, 0.5]))
            .collect();
        idx.rebuild(all).unwrap();

        let results = idx.search(&vec![50.0, 1.0, 0.0, 0.5], 5).unwrap();
        assert!(!results.is_empty(), "expected results after rebuild");
    }

    // ── Test 5 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ivf.json");
        let cfg = ivf_config("saveload", 4);

        let mut idx = IvfIndex::new(&cfg);
        for i in 0..20usize {
            idx.insert(format!("v{i}"), vec![i as f32, 2.0, 0.0, 1.0])
                .unwrap();
        }

        let query = vec![10.0, 2.0, 0.0, 1.0];
        let before = idx.search(&query, 3).unwrap();
        assert!(!before.is_empty(), "pre-save search should return results");

        idx.save(&path).unwrap();

        let loaded = IvfIndex::load_file(&path).unwrap();
        let after = loaded.search(&query, 3).unwrap();
        assert!(!after.is_empty(), "post-load search should return results");
    }

    // ── Test 6 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_ivf_degenerate_fewer_vectors_than_lists() {
        // n_lists=256 (default) but only 10 vectors — should not panic
        let cfg = ivf_config("degen", 4);
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..10usize {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0, 1.0, 0.0])
                .unwrap();
        }

        let results = idx.search(&vec![5.0, 0.0, 1.0, 0.0], 5).unwrap();
        assert!(
            !results.is_empty(),
            "degenerate case must still return results"
        );
    }

    // ── Test 7 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_storage_ivf_full_roundtrip() {
        let dir = tempdir().unwrap();
        let mut cfg = CollectionConfig::new("ivfstore", 4);
        cfg.index_type = IndexType::IVF;

        {
            let mut storage = Storage::create(dir.path(), &cfg).unwrap();
            for i in 0..50usize {
                storage
                    .upsert(VectorRecord {
                        id: format!("v{i}"),
                        vector: vec![i as f32, (i as f32) * 0.5, 1.0, 0.0],
                        payload: json!({}),
                        text: None,
                        created_at: Utc::now(),
                        updated_at: Utc::now(),
                    })
                    .unwrap();
            }
            let results = storage.search(&vec![25.0, 12.5, 1.0, 0.0], 5).unwrap();
            assert!(
                !results.is_empty(),
                "search before save must return results"
            );
            storage.save_indexes().unwrap();
        }

        let storage = Storage::open(dir.path(), "ivfstore").unwrap();
        let results = storage.search(&vec![25.0, 12.5, 1.0, 0.0], 5).unwrap();
        assert!(
            !results.is_empty(),
            "search after reopen must return results"
        );

        // Verify the saved file is .ivf.json
        assert!(
            dir.path().join("ivfstore.ivf.json").exists(),
            "ivf.json file must exist after save_indexes()"
        );
    }

    // ── Test 8 ───────────────────────────────────────────────────────────────
    #[test]
    fn test_anyindex_hnsw_still_works() {
        let dir = tempdir().unwrap();
        let cfg = CollectionConfig::new("hnsworeg", 4);
        assert_eq!(cfg.index_type, IndexType::HNSW, "default must be HNSW");

        let mut storage = Storage::create(dir.path(), &cfg).unwrap();
        for i in 0..20usize {
            storage
                .upsert(dummy_record(
                    &format!("h{i}"),
                    vec![i as f32, 1.0, 0.0, 0.5],
                ))
                .unwrap();
        }

        let results = storage.search(&vec![10.0, 1.0, 0.0, 0.5], 5).unwrap();
        assert!(
            !results.is_empty(),
            "HNSW search must still work after AnyIndex refactor"
        );

        // Verify the index variant
        assert!(
            matches!(storage.index, AnyIndex::Hnsw(_)),
            "HNSW config must produce AnyIndex::Hnsw variant"
        );
    }
}
