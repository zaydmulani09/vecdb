use serde::{Deserialize, Serialize};

// ── Health ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub vector_count: usize,
    pub collection: String,
}

// ── Collections ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CollectionResponse {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub index_type: String,
    pub vector_count: usize,
    pub created_at: String,
}

#[derive(Debug, Deserialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<CollectionResponse>,
    pub count: usize,
}

#[derive(Debug, Deserialize)]
pub struct DeleteCollectionResponse {
    pub deleted: bool,
    pub name: String,
}

// ── Vectors ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct UpsertRequest {
    pub records: Vec<UpsertRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpsertRecord {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertResponse {
    pub inserted: usize,
    pub updated: usize,
    pub errors: Vec<UpsertError>,
    pub time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct UpsertError {
    pub id: String,
    pub error: String,
}

// ── Search ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DenseSearchRequest {
    pub vector: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct SparseSearchRequest {
    pub query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct HybridSearchRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub alpha: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct SqlQueryRequest {
    pub sql: String,
}

#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub count: usize,
    pub time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub payload: serde_json::Value,
    pub text: Option<String>,
}
