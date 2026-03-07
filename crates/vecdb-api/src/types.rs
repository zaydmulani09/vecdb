use serde::{Deserialize, Serialize};
use vecdb_core::{CollectionConfig, IndexStats, SearchResult};

// ── Collection endpoints ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    pub metric: Option<String>,
    pub index_type: Option<String>,
    pub hnsw_m: Option<usize>,
    pub hnsw_ef_construction: Option<usize>,
    pub bm25_k1: Option<f32>,
    pub bm25_b: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct CollectionResponse {
    pub name: String,
    pub dimension: usize,
    pub metric: String,
    pub index_type: String,
    pub vector_count: usize,
    pub created_at: String,
}

impl CollectionResponse {
    pub fn from_config_and_stats(config: &CollectionConfig, stats: &IndexStats) -> Self {
        Self {
            name: config.name.clone(),
            dimension: config.dimension,
            metric: config.metric.to_string(),
            index_type: config.index_type.to_string(),
            vector_count: stats.vector_count,
            created_at: config.created_at.to_rfc3339(),
        }
    }
}

// ── Vector endpoints ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct UpsertVectorsRequest {
    pub records: Vec<VectorRecordInput>,
}

#[derive(Debug, Deserialize)]
pub struct VectorRecordInput {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: Option<serde_json::Value>,
    pub text: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UpsertVectorsResponse {
    pub inserted: usize,
    pub updated: usize,
    pub errors: Vec<UpsertError>,
    pub time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct UpsertError {
    pub id: String,
    pub error: String,
}

#[derive(Debug, Deserialize)]
pub struct DeleteVectorsRequest {
    pub ids: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct DeleteVectorsResponse {
    pub deleted: usize,
    pub errors: Vec<UpsertError>,
    pub time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct GetVectorResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub payload: serde_json::Value,
    pub text: Option<String>,
}

// ── Search endpoints ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct DenseSearchRequest {
    pub vector: Vec<f32>,
    pub k: Option<usize>,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct SparseSearchRequest {
    pub query: String,
    pub k: Option<usize>,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct HybridSearchRequest {
    pub vector: Option<Vec<f32>>,
    pub query: Option<String>,
    pub k: Option<usize>,
    pub alpha: Option<f32>,
    pub strategy: Option<String>,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
pub struct SqlQueryRequest {
    pub sql: String,
}

#[derive(Debug, Deserialize)]
pub struct ExplainRequest {
    pub sql: Option<String>,
    pub vector: Option<Vec<f32>>,
    pub query: Option<String>,
    pub k: Option<usize>,
    pub collection: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
    pub count: usize,
    pub time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
    pub id: String,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub payload: serde_json::Value,
    pub text: Option<String>,
}

impl From<SearchResult> for SearchResultResponse {
    fn from(r: SearchResult) -> Self {
        Self {
            id: r.id,
            score: r.score,
            dense_score: r.dense_score,
            sparse_score: r.sparse_score,
            payload: r.payload,
            text: r.text,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ExplainResponse {
    pub plan: String,
    pub index_type: String,
    pub use_hybrid: bool,
    pub candidate_k: usize,
    pub output_k: usize,
    pub estimated_cost: f32,
}

// ── Health endpoint ───────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub vector_count: usize,
    pub collections: usize,
}
