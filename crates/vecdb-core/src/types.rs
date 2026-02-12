use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;

pub type VectorId = String;
pub type Vector = Vec<f32>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum DistanceMetric {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

impl fmt::Display for DistanceMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistanceMetric::Cosine => write!(f, "cosine"),
            DistanceMetric::Euclidean => write!(f, "euclidean"),
            DistanceMetric::DotProduct => write!(f, "dot_product"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub enum IndexType {
    #[default]
    HNSW,
    IVF,
}

impl fmt::Display for IndexType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexType::HNSW => write!(f, "hnsw"),
            IndexType::IVF => write!(f, "ivf"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    pub name: String,
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub index_type: IndexType,
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    pub hnsw_ef_search: usize,
    pub bm25_k1: f32,
    pub bm25_b: f32,
    pub created_at: DateTime<Utc>,
}

impl CollectionConfig {
    pub fn new(name: impl Into<String>, dimension: usize) -> Self {
        Self {
            name: name.into(),
            dimension,
            metric: DistanceMetric::default(),
            index_type: IndexType::default(),
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            hnsw_ef_search: 50,
            bm25_k1: 1.5,
            bm25_b: 0.75,
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: VectorId,
    pub vector: Vector,
    pub payload: serde_json::Value,
    pub text: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: VectorId,
    pub score: f32,
    pub dense_score: Option<f32>,
    pub sparse_score: Option<f32>,
    pub payload: serde_json::Value,
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub vector: Option<Vector>,
    pub query_text: Option<String>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    pub collection: String,
    pub filter: Option<serde_json::Value>,
}

fn default_k() -> usize {
    10
}
fn default_alpha() -> f32 {
    0.7
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub records: Vec<VectorRecord>,
    pub collection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpsertResponse {
    pub inserted: usize,
    pub updated: usize,
    pub errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub ids: Vec<VectorId>,
    pub collection: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub collection: String,
    pub vector_count: usize,
    pub index_type: IndexType,
    pub dimension: usize,
    pub disk_bytes: u64,
    pub memory_bytes: u64,
}
