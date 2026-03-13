use serde_json::json;
use tempfile::TempDir;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::{client::VecDbClient, commands};

// ── helpers ───────────────────────────────────────────────────────────────

fn health_body() -> serde_json::Value {
    json!({
        "status": "ok",
        "version": "0.1.0",
        "vector_count": 0,
        "collection": "default"
    })
}

fn collection_body(name: &str) -> serde_json::Value {
    json!({
        "name": name,
        "dimension": 3,
        "metric": "cosine",
        "index_type": "hnsw",
        "vector_count": 0,
        "created_at": "2024-01-01T00:00:00Z"
    })
}

fn upsert_response(inserted: usize) -> serde_json::Value {
    json!({
        "inserted": inserted,
        "updated": 0,
        "errors": [],
        "time_ms": 5
    })
}

fn search_response() -> serde_json::Value {
    json!({
        "results": [
            {
                "id": "doc1",
                "score": 0.99,
                "dense_score": 0.99,
                "sparse_score": null,
                "payload": {},
                "text": null
            }
        ],
        "count": 1,
        "time_ms": 3
    })
}

// ── Test 1 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ping_hits_health_endpoint() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200).set_body_json(health_body()))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::ping::run(&client).await.unwrap();

    // Verify exactly one GET /health was received
    let reqs = mock_server.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1);
    assert_eq!(reqs[0].url.path(), "/health");
}

// ── Test 2 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_collection_list() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/collections"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "collections": [collection_body("default")],
            "count": 1
        })))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::collection::run(&client, commands::collection::CollectionCmd::List)
        .await
        .unwrap();
}

// ── Test 3 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_collection_create() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/collections"))
        .respond_with(ResponseTemplate::new(201).set_body_json(collection_body("mycol")))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::collection::run(
        &client,
        commands::collection::CollectionCmd::Create {
            name: "mycol".to_string(),
            dimension: 3,
            metric: "cosine".to_string(),
        },
    )
    .await
    .unwrap();
}

// ── Test 4 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_jsonl() {
    let mock_server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let jsonl = r#"{"id":"a","vector":[0.1,0.2,0.3]}
{"id":"b","vector":[0.4,0.5,0.6]}
{"id":"c","vector":[0.7,0.8,0.9]}
"#;
    let file = dir.path().join("data.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    Mock::given(method("POST"))
        .and(path("/collections/test/vectors"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upsert_response(3)))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::ingest::run(&client, "test", Some(&file), 100, false)
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1, "expected exactly one POST request");

    // Verify all 3 records were sent in the single batch
    let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(
        body["records"].as_array().unwrap().len(),
        3,
        "expected 3 records in the batch"
    );
}

// ── Test 5 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_csv() {
    let mock_server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let csv =
        "id,vector,text\ndoc1,\"[0.1,0.2,0.3]\",hello world\ndoc2,\"[0.4,0.5,0.6]\",foo bar\n";
    let file = dir.path().join("data.csv");
    std::fs::write(&file, csv).unwrap();

    Mock::given(method("POST"))
        .and(path("/collections/test/vectors"))
        .respond_with(ResponseTemplate::new(200).set_body_json(upsert_response(2)))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::ingest::run(&client, "test", Some(&file), 100, false)
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1, "expected exactly one POST request");

    let body: serde_json::Value = serde_json::from_slice(&reqs[0].body).unwrap();
    assert_eq!(
        body["records"].as_array().unwrap().len(),
        2,
        "expected 2 records from CSV"
    );
}

// ── Test 6 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_ingest_dry_run() {
    let mock_server = MockServer::start().await;
    let dir = TempDir::new().unwrap();

    let jsonl =
        "{\"id\":\"a\",\"vector\":[0.1,0.2,0.3]}\n{\"id\":\"b\",\"vector\":[0.4,0.5,0.6]}\n";
    let file = dir.path().join("data.jsonl");
    std::fs::write(&file, jsonl).unwrap();

    // No mock for POST — if it is called the test should still pass but we
    // verify below that zero requests arrived.

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::ingest::run(&client, "test", Some(&file), 100, true)
        .await
        .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    assert!(
        reqs.is_empty(),
        "dry_run must not send any HTTP requests, got: {}",
        reqs.len()
    );
}

// ── Test 7 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_search_dense() {
    let mock_server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/collections/test/search/dense"))
        .respond_with(ResponseTemplate::new(200).set_body_json(search_response()))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    commands::search::run(
        &client,
        commands::search::SearchCmd::Dense {
            collection: "test".to_string(),
            vector: "[0.1,0.2,0.3]".to_string(),
            k: 5,
        },
    )
    .await
    .unwrap();

    let reqs = mock_server.received_requests().await.unwrap();
    assert_eq!(reqs.len(), 1);
}

// ── Test 8 ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_client_401_returns_clear_error() {
    let mock_server = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(401).set_body_json(json!({
            "error": { "code": "unauthorized", "message": "invalid or missing API key" }
        })))
        .mount(&mock_server)
        .await;

    let client = VecDbClient::new(mock_server.uri(), None).unwrap();
    let result = commands::ping::run(&client).await;

    assert!(result.is_err(), "expected an error for 401 response");
    let err_msg = format!("{:#}", result.unwrap_err()).to_lowercase();
    assert!(
        err_msg.contains("api-key") || err_msg.contains("unauthorized"),
        "error message should mention api-key or unauthorized; got: {}",
        err_msg
    );
}
