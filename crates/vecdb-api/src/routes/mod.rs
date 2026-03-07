pub mod collections;
pub mod health;
pub mod search;
pub mod vectors;

#[cfg(test)]
mod tests;

use std::time::Instant;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};

use crate::state::SharedState;

// ── Router ────────────────────────────────────────────────────────────────

pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/health", get(health::health))
        .route(
            "/collections",
            post(collections::create_collection).get(collections::list_collections),
        )
        .route(
            "/collections/:name",
            get(collections::get_collection).delete(collections::delete_collection),
        )
        .route(
            "/collections/:name/vectors",
            post(vectors::upsert_vectors).delete(vectors::delete_vectors),
        )
        .route("/collections/:name/vectors/:id", get(vectors::get_vector))
        .route(
            "/collections/:name/search/dense",
            post(search::search_dense),
        )
        .route(
            "/collections/:name/search/sparse",
            post(search::search_sparse),
        )
        .route(
            "/collections/:name/search/hybrid",
            post(search::search_hybrid),
        )
        .route("/query", post(search::query_sql))
        .route("/explain", post(search::explain_query))
        .route("/metrics", get(metrics_handler))
        .with_state(state)
}

// ── Metrics handler ───────────────────────────────────────────────────────

async fn metrics_handler(State(state): State<SharedState>) -> impl IntoResponse {
    let body = state.metrics_handle.render();
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
}

// ── Request timer ─────────────────────────────────────────────────────────

pub struct RequestTimer {
    operation: &'static str,
    collection: String,
    started: Instant,
}

impl RequestTimer {
    pub fn start(operation: &'static str, collection: Option<&str>) -> Self {
        let collection = collection.unwrap_or("none").to_string();
        metrics::counter!(
            crate::metrics::REQUESTS_TOTAL,
            "operation" => operation.to_string(),
            "collection" => collection.clone(),
        )
        .increment(1);
        tracing::info!(operation, collection, "request started");
        Self {
            operation,
            collection,
            started: Instant::now(),
        }
    }
}

impl Drop for RequestTimer {
    fn drop(&mut self) {
        let elapsed_ms = self.started.elapsed().as_secs_f64() * 1000.0;
        metrics::histogram!(
            crate::metrics::REQUEST_DURATION_MS,
            "operation" => self.operation.to_string(),
            "collection" => self.collection.clone(),
        )
        .record(elapsed_ms);
        tracing::info!(
            operation = self.operation,
            collection = self.collection,
            elapsed_ms,
            "request finished"
        );
    }
}
