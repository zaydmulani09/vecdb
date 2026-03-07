use std::sync::{Arc, OnceLock};

use axum::{
    body::{to_bytes, Body},
    http::{Request, StatusCode},
    Router,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::sync::RwLock;
use tower::ServiceExt;
use vecdb_core::{storage::Storage, CollectionConfig};

use crate::{
    collection_manager::CollectionManager,
    middleware::auth_middleware,
    routes::router,
    state::{AppState, SharedState},
};

// ── Metrics OnceLock — installs recorder at most once per test process ────

static TEST_METRICS_HANDLE: OnceLock<PrometheusHandle> = OnceLock::new();

fn get_or_init_metrics() -> PrometheusHandle {
    TEST_METRICS_HANDLE
        .get_or_init(|| {
            PrometheusBuilder::new()
                .install_recorder()
                .expect("failed to install test recorder")
        })
        .clone()
}

// ── Test helpers ──────────────────────────────────────────────────────────

fn make_state(dir: &TempDir, dimension: usize, api_key: Option<String>) -> SharedState {
    let config = CollectionConfig::new("default", dimension);
    let storage = Storage::create(dir.path(), &config).expect("create storage");
    let mut manager = CollectionManager::new(dir.path().to_path_buf());
    manager.insert("default", storage);
    Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key,
    })
}

/// Build a fresh test app with a 3-dim "default" collection and api_key = "test-key".
/// The metrics handle is global (OnceLock) so counters accumulate across apps.
fn build_test_app() -> Router {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    // Prevent TempDir cleanup while storage has files open on Windows.
    std::mem::forget(dir);
    let config = CollectionConfig::new("default", 3);
    let storage = Storage::create(&path, &config).expect("create storage");
    let mut manager = CollectionManager::new(path);
    manager.insert("default", storage);
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: Some("test-key".to_string()),
    });
    build_app(state)
}

fn build_app(state: SharedState) -> Router {
    use axum::middleware;
    use crate::middleware::security_headers;
    router(state.clone())
        .layer(middleware::from_fn_with_state(state, auth_middleware))
        .layer(middleware::from_fn(security_headers))
}

async fn post_json(app: Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn get_req(app: Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn get_req_with_key(app: Router, uri: &str, key: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .header("X-Api-Key", key)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn delete_json(app: Router, uri: &str, body: Value) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn delete_req(app: Router, uri: &str) -> (StatusCode, Value) {
    let req = Request::builder()
        .method("DELETE")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

async fn body_to_string(response: axum::response::Response) -> String {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn vec_records(n: usize, dim: usize) -> Value {
    let records: Vec<Value> = (0..n)
        .map(|i| {
            let v: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32 * 0.1).collect();
            json!({
                "id": format!("vec{}", i),
                "vector": v,
                "payload": { "idx": i },
                "text": format!("machine learning document number {}", i),
            })
        })
        .collect();
    json!({ "records": records })
}

// ── Tests 1-10 (original) ─────────────────────────────────────────────────

#[tokio::test]
async fn test_health_endpoint() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let (status, body) = get_req(app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_upsert_and_get_vector() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let records = json!({
        "records": [
            { "id": "a", "vector": [1.0f32, 0.0, 0.0, 0.0] },
            { "id": "b", "vector": [0.0f32, 1.0, 0.0, 0.0] },
            { "id": "c", "vector": [0.0f32, 0.0, 1.0, 0.0] },
        ]
    });

    let (status, body) = post_json(app.clone(), "/collections/default/vectors", records).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["inserted"], 3);

    let (status, body) = get_req(app, "/collections/default/vectors/a").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["id"], "a");
    let v = &body["vector"];
    assert!((v[0].as_f64().unwrap() - 1.0).abs() < 1e-5);
    assert!((v[1].as_f64().unwrap()).abs() < 1e-5);
}

#[tokio::test]
async fn test_dense_search() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = vec_records(5, 4);
    let (status, _) = post_json(app.clone(), "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);

    let search_body = json!({ "vector": [1.0f32, 0.0, 0.0, 0.0], "k": 3 });
    let (status, body) = post_json(app, "/collections/default/search/dense", search_body).await;
    assert_eq!(status, StatusCode::OK);

    let results = body["results"].as_array().unwrap();
    assert!(results.len() <= 3);
    assert!(!results.is_empty());
    for r in results {
        assert!(r["score"].as_f64().is_some());
    }
}

#[tokio::test]
async fn test_sparse_search() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = vec_records(5, 4);
    let (status, _) = post_json(app.clone(), "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);

    let search_body = json!({ "query": "machine learning", "k": 3 });
    let (status, body) = post_json(app, "/collections/default/search/sparse", search_body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["results"].as_array().is_some());
}

#[tokio::test]
async fn test_hybrid_search() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = vec_records(5, 4);
    let (status, _) = post_json(app.clone(), "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);

    let search_body = json!({
        "vector": [1.0f32, 0.0, 0.0, 0.0],
        "query": "machine learning",
        "k": 3
    });
    let (status, body) = post_json(app, "/collections/default/search/hybrid", search_body).await;
    assert_eq!(status, StatusCode::OK);

    let results = body["results"].as_array().unwrap();
    assert!(!results.is_empty());
    for r in results {
        assert!(r["dense_score"].is_number() || r["sparse_score"].is_number());
    }
}

#[tokio::test]
async fn test_sql_query() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = vec_records(5, 4);
    let (status, _) = post_json(app.clone(), "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);

    let sql_body = json!({
        "sql": "SELECT * FROM default WHERE VECTOR_SIM(embedding, [1.0, 0.0, 0.0, 0.0]) > 0.0 LIMIT 5"
    });
    let (status, body) = post_json(app, "/query", sql_body).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["results"].as_array().is_some());
}

#[tokio::test]
async fn test_delete_vector() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = json!({
        "records": [{ "id": "del_me", "vector": [1.0f32, 0.0, 0.0, 0.0] }]
    });
    let (status, _) = post_json(app.clone(), "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);

    let delete_body = json!({ "ids": ["del_me"] });
    let (status, body) =
        delete_json(app.clone(), "/collections/default/vectors", delete_body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["deleted"], 1);

    let (status, _) = get_req(app, "/collections/default/vectors/del_me").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_auth_middleware_blocks_without_key() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, Some("secret123".to_string()));
    let app = build_app(state);

    let (status, _) = get_req(app, "/health").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_middleware_allows_with_correct_key() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, Some("secret123".to_string()));
    let app = build_app(state);

    let (status, body) = get_req_with_key(app, "/health", "secret123").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
}

#[tokio::test]
async fn test_upsert_dimension_mismatch() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 4, None);
    let app = build_app(state);

    let upsert_body = json!({
        "records": [
            { "id": "good", "vector": [1.0f32, 0.0, 0.0, 0.0] },
            { "id": "bad",  "vector": [1.0f32, 0.0, 0.0] },
        ]
    });
    let (status, body) = post_json(app, "/collections/default/vectors", upsert_body).await;
    assert_eq!(status, StatusCode::OK);
    let errors = body["errors"].as_array().unwrap();
    assert!(!errors.is_empty());
    assert_eq!(errors[0]["id"], "bad");
}

// ── Tests 11-18 (metrics / observability) ────────────────────────────────

#[tokio::test]
async fn test_metrics_endpoint_returns_200() {
    let app = build_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_metrics_endpoint_returns_prometheus_content_type() {
    let app = build_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap();
    assert!(content_type.contains("text/plain"));
}

#[tokio::test]
async fn test_requests_total_increments_after_health_check() {
    let app = build_test_app();
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri("/health")
            .header("X-Api-Key", "test-key")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();

    let app2 = build_test_app();
    let response = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = body_to_string(response).await;
    assert!(body.contains("vecdb_requests_total"));
}

#[tokio::test]
async fn test_metrics_contains_duration_histogram() {
    let app = build_test_app();
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri("/health")
            .header("X-Api-Key", "test-key")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();

    let app2 = build_test_app();
    let response = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = body_to_string(response).await;
    assert!(body.contains("vecdb_request_duration_ms"));
}

#[tokio::test]
async fn test_upsert_increments_upsert_total() {
    let app = build_test_app();
    let body = serde_json::json!({
        "records": [{
            "id": "metric-vec-1",
            "vector": [0.1_f32, 0.2_f32, 0.3_f32]
        }]
    });
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/collections/default/vectors")
            .header("X-Api-Key", "test-key")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap();

    let app2 = build_test_app();
    let response = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body_str = body_to_string(response).await;
    assert!(body_str.contains("vecdb_upsert_total"));
}

#[tokio::test]
async fn test_errors_total_increments_on_not_found() {
    let app = build_test_app();
    app.oneshot(
        Request::builder()
            .method("GET")
            .uri("/collections/default/vectors/does-not-exist-xyz")
            .header("X-Api-Key", "test-key")
            .body(Body::empty())
            .unwrap(),
    )
    .await
    .unwrap();

    let app2 = build_test_app();
    let response = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body_str = body_to_string(response).await;
    assert!(body_str.contains("vecdb_errors_total"));
}

#[tokio::test]
async fn test_search_results_gauge_present_after_dense_search() {
    let app = build_test_app();
    let upsert_body = serde_json::json!({
        "records": [{
            "id": "gauge-test-vec",
            "vector": [0.1_f32, 0.2_f32, 0.3_f32]
        }]
    });
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/collections/default/vectors")
            .header("X-Api-Key", "test-key")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&upsert_body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap();

    let app2 = build_test_app();
    let search_body = serde_json::json!({
        "vector": [0.1_f32, 0.2_f32, 0.3_f32],
        "k": 1
    });
    app2.oneshot(
        Request::builder()
            .method("POST")
            .uri("/collections/default/search/dense")
            .header("X-Api-Key", "test-key")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(&search_body).unwrap()))
            .unwrap(),
    )
    .await
    .unwrap();

    let app3 = build_test_app();
    let response = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .header("X-Api-Key", "test-key")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body_str = body_to_string(response).await;
    assert!(body_str.contains("vecdb_search_results_count"));
}

#[tokio::test]
async fn test_metrics_endpoint_requires_auth() {
    let app = build_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
}

// ── Tests 19-26 (multi-collection) ───────────────────────────────────────

#[tokio::test]
async fn test_create_collection_returns_201() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body = json!({ "name": "my_col", "dimension": 8 });
    let (status, resp) = post_json(app, "/collections", body).await;
    assert_eq!(status, StatusCode::CREATED);
    assert_eq!(resp["name"], "my_col");
    assert_eq!(resp["dimension"], 8);
}

#[tokio::test]
async fn test_create_duplicate_collection_returns_409() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body = json!({ "name": "dup_col", "dimension": 4 });
    let (status, _) = post_json(app.clone(), "/collections", body.clone()).await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, _) = post_json(app, "/collections", body).await;
    assert_eq!(status, StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_get_collection_returns_404_for_unknown() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let (status, _) = get_req(app, "/collections/does_not_exist").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_collection_returns_200() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body = json!({ "name": "to_delete", "dimension": 4 });
    let (status, _) = post_json(app.clone(), "/collections", body).await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, resp) = delete_req(app, "/collections/to_delete").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["deleted"], true);
    assert_eq!(resp["name"], "to_delete");
}

#[tokio::test]
async fn test_delete_unknown_collection_returns_404() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let (status, _) = delete_req(app, "/collections/ghost").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_vectors_in_unknown_collection_returns_404() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body = json!({ "records": [{ "id": "x", "vector": [1.0f32, 0.0] }] });
    let (status, _) = post_json(app, "/collections/no_such/vectors", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_search_in_unknown_collection_returns_404() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body = json!({ "vector": [1.0f32, 0.0, 0.0], "k": 5 });
    let (status, _) = post_json(app, "/collections/no_such/search/dense", body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_list_collections_shows_created() {
    let dir = TempDir::new().unwrap();
    let manager = CollectionManager::new(dir.path().to_path_buf());
    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle: get_or_init_metrics(),
        api_key: None,
    });
    let app = build_app(state);

    let body_a = json!({ "name": "alpha", "dimension": 4 });
    let body_b = json!({ "name": "beta", "dimension": 8 });
    let (status, _) = post_json(app.clone(), "/collections", body_a).await;
    assert_eq!(status, StatusCode::CREATED);
    let (status, _) = post_json(app.clone(), "/collections", body_b).await;
    assert_eq!(status, StatusCode::CREATED);

    let (status, resp) = get_req(app, "/collections").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["count"], 2);
    let names: Vec<&str> = resp["collections"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["name"].as_str().unwrap())
        .collect();
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
}

// ── Tests 27-28 (shutdown / connection-pool smoke) ────────────────────────

// Test 27 — simulates the shutdown flush: checkpoint + save_indexes must succeed
// and produce the index file on disk.
#[tokio::test]
async fn test_shutdown_flush_checkpoint_and_indexes() {
    use chrono::Utc;
    use serde_json::json;
    use vecdb_core::types::VectorRecord;

    let dir = TempDir::new().unwrap();
    let config = CollectionConfig::new("flushcol", 3);
    let mut storage = Storage::create(dir.path(), &config).expect("create storage");

    let now = Utc::now();
    for i in 0..3usize {
        storage
            .upsert(VectorRecord {
                id: format!("f{i}"),
                vector: vec![i as f32, 1.0, 0.0],
                payload: json!({}),
                text: None,
                created_at: now,
                updated_at: now,
            })
            .expect("upsert");
    }

    storage.checkpoint().expect("checkpoint must succeed");
    storage.save_indexes().expect("save_indexes must succeed");

    // At least one index file must exist after save_indexes.
    let hnsw = dir.path().join("flushcol.hnsw.json");
    let ivf = dir.path().join("flushcol.ivf.json");
    assert!(
        hnsw.exists() || ivf.exists(),
        "expected an index file after save_indexes"
    );
}

// Test 28 — the full app stack (with pooled MetadataStore) handles GET /health.
#[tokio::test]
async fn test_server_handles_health_with_pooled_metadata() {
    let app = build_test_app();
    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .header("X-Api-Key", "test-key")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value = serde_json::from_slice(
        &to_bytes(response.into_body(), usize::MAX).await.unwrap(),
    )
    .unwrap();
    assert_eq!(body["status"], "ok");
}

// ── Tests 29-36 (security hardening) ─────────────────────────────────────

// Test 29 — collection name with invalid characters returns 400.
#[tokio::test]
async fn test_invalid_collection_name_chars_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let body = json!({ "vector": [0.1f32, 0.2, 0.3], "k": 5 });
    let (status, resp) = post_json(app, "/collections/bad@name!/search/dense", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 30 — collection name longer than 64 characters returns 400.
#[tokio::test]
async fn test_collection_name_too_long_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let long_name = "a".repeat(65);
    let body = json!({ "vector": [0.1f32, 0.2, 0.3], "k": 5 });
    let (status, resp) = post_json(
        app,
        &format!("/collections/{}/search/dense", long_name),
        body,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 31 — k=0 is rejected with 400.
#[tokio::test]
async fn test_search_k_zero_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let body = json!({ "vector": [0.1f32, 0.2, 0.3], "k": 0 });
    let (status, resp) = post_json(app, "/collections/default/search/dense", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 32 — k=10001 (above 10000) is rejected with 400.
#[tokio::test]
async fn test_search_k_too_large_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let body = json!({ "vector": [0.1f32, 0.2, 0.3], "k": 10001 });
    let (status, resp) = post_json(app, "/collections/default/search/dense", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 33 — alpha below 0.0 is rejected with 400.
#[tokio::test]
async fn test_hybrid_alpha_too_low_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let body = json!({
        "vector": [0.1f32, 0.2, 0.3],
        "query": "test query",
        "k": 5,
        "alpha": -0.1
    });
    let (status, resp) = post_json(app, "/collections/default/search/hybrid", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 34 — alpha above 1.0 is rejected with 400.
#[tokio::test]
async fn test_hybrid_alpha_too_high_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let body = json!({
        "vector": [0.1f32, 0.2, 0.3],
        "query": "test query",
        "k": 5,
        "alpha": 1.1
    });
    let (status, resp) = post_json(app, "/collections/default/search/hybrid", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 35 — SQL query over 4096 characters is rejected with 400.
#[tokio::test]
async fn test_sql_too_long_rejected() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    // Build a SQL string that is well over 4096 characters.
    let long_sql = format!(
        "SELECT * FROM default WHERE {}",
        "x = 1 AND ".repeat(500)
    );
    assert!(long_sql.len() > 4096, "test setup: sql must exceed 4096 chars");

    let body = json!({ "sql": long_sql });
    let (status, resp) = post_json(app, "/query", body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "invalid_query");
}

// Test 36 — security response headers are present on every response.
#[tokio::test]
async fn test_security_headers_present() {
    let dir = TempDir::new().unwrap();
    let state = make_state(&dir, 3, None);
    let app = build_app(state);

    let req = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let response = app.oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let headers = response.headers();
    assert_eq!(
        headers
            .get("x-content-type-options")
            .expect("x-content-type-options header missing")
            .to_str()
            .unwrap(),
        "nosniff"
    );
    assert_eq!(
        headers
            .get("x-frame-options")
            .expect("x-frame-options header missing")
            .to_str()
            .unwrap(),
        "DENY"
    );
    assert_eq!(
        headers
            .get("x-xss-protection")
            .expect("x-xss-protection header missing")
            .to_str()
            .unwrap(),
        "0"
    );
    assert_eq!(
        headers
            .get("referrer-policy")
            .expect("referrer-policy header missing")
            .to_str()
            .unwrap(),
        "strict-origin-when-cross-origin"
    );
}
