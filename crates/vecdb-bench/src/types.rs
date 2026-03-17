#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ── Upsert ────────────────────────────────────────────────────────────────

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
    pub errors: Vec<serde_json::Value>,
    pub time_ms: f64,
}

// ── Search ────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct DenseSearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
}

#[derive(Debug, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub count: usize,
    pub time_ms: f64,
}

#[derive(Debug, Deserialize)]
pub struct SearchResult {
    pub id: String,
    pub score: f64,
}

// ── Collections ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
}
