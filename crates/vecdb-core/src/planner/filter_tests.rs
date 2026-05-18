#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    use crate::planner::executor::apply_json_filter;
    use crate::storage::Storage;
    use crate::types::{CollectionConfig, SearchResult, VectorRecord};

    fn make_result(id: &str, payload: serde_json::Value) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score: 1.0,
            dense_score: None,
            sparse_score: None,
            payload,
            text: None,
        }
    }

    // ── Test 1 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_equality_string() {
        let r = make_result("doc1", json!({"region": "US"}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "region", "op": "=", "value": "US"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "region", "op": "=", "value": "EU"})
        ));
    }

    // ── Test 2 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_not_equal() {
        let r = make_result("doc2", json!({"status": "active"}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "status", "op": "!=", "value": "inactive"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "status", "op": "!=", "value": "active"})
        ));
    }

    // ── Test 3 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_numeric_gt() {
        let r = make_result("doc3", json!({"price": 150}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "price", "op": ">", "value": 100})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "price", "op": ">", "value": 200})
        ));
    }

    // ── Test 4 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_numeric_lte() {
        let r = make_result("doc4", json!({"score": 0.5}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "score", "op": "<=", "value": 0.5})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "score", "op": "<=", "value": 0.4})
        ));
    }

    // ── Test 5 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_like_prefix() {
        let r = make_result("doc5", json!({"name": "hello world"}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "name", "op": "LIKE", "value": "hello%"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "name", "op": "LIKE", "value": "world%"})
        ));
    }

    // ── Test 6 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_like_contains() {
        let r = make_result("doc6", json!({"name": "hello world"}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "name", "op": "LIKE", "value": "%world%"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "name", "op": "LIKE", "value": "%xyz%"})
        ));
    }

    // ── Test 7 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_nested_path() {
        let r = make_result("doc7", json!({"meta": {"region": "EU"}}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "meta.region", "op": "=", "value": "EU"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "meta.region", "op": "=", "value": "US"})
        ));
    }

    // ── Test 8 ───────────────────────────────────────────────────
    #[test]
    fn test_filter_id_field() {
        let r = make_result("doc-123", json!({}));
        assert!(apply_json_filter(
            &r,
            &json!({"field": "id", "op": "=", "value": "doc-123"})
        ));
        assert!(!apply_json_filter(
            &r,
            &json!({"field": "id", "op": "=", "value": "doc-999"})
        ));
    }

    // ── Test 9 ───────────────────────────────────────────────────
    #[test]
    fn test_projection_keeps_only_selected_columns() {
        // Call apply_projection indirectly via execute_sql.
        // Direct test: build a SearchResult and manually call the projection logic
        // by running execute_sql with a SELECT id, region query.
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("projcol", 3);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        storage
            .upsert(VectorRecord {
                id: "p1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({"title": "foo", "region": "US", "price": 100}),
                text: None,
                created_at: now,
                updated_at: now,
            })
            .unwrap();

        let sql = "SELECT id, region FROM projcol \
                   WHERE VECTOR_SIM(embedding, [1.0, 0.0, 0.0]) > 0.0 LIMIT 5";
        let results = storage.execute_sql(sql).unwrap();

        assert!(!results.is_empty(), "should return at least one result");
        for r in &results {
            // "region" should be in payload
            assert!(
                r.payload.get("region").is_some(),
                "payload must contain 'region'"
            );
            // "title" and "price" should NOT be in payload (not selected)
            assert!(
                r.payload.get("title").is_none(),
                "payload must not contain 'title'"
            );
            assert!(
                r.payload.get("price").is_none(),
                "payload must not contain 'price'"
            );
        }
    }

    // ── Test 10 ──────────────────────────────────────────────────
    #[test]
    fn test_storage_sql_with_filter_and_projection() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("docs", 3);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        for i in 0..3usize {
            storage
                .upsert(VectorRecord {
                    id: format!("us{i}"),
                    vector: vec![i as f32, 1.0, 0.0],
                    payload: json!({"region": "US", "extra": i}),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }
        for i in 0..2usize {
            storage
                .upsert(VectorRecord {
                    id: format!("eu{i}"),
                    vector: vec![i as f32, 0.0, 1.0],
                    payload: json!({"region": "EU", "extra": i}),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let sql = "SELECT id, region FROM docs \
                   WHERE VECTOR_SIM(embedding, [1.0, 1.0, 0.0]) > 0.0 \
                   AND region = 'US' LIMIT 10";
        let results = storage.execute_sql(sql).unwrap();

        assert!(!results.is_empty(), "should return US results");
        for r in &results {
            assert_eq!(
                r.payload.get("region").and_then(|v| v.as_str()),
                Some("US"),
                "all results must have region=US"
            );
            assert!(
                r.payload.get("extra").is_none(),
                "payload must not contain 'extra' (not selected)"
            );
        }
    }
}
