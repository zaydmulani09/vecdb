use axum::{extract::State, Json};

use crate::{error::ApiResult, state::SharedState, types::HealthResponse};

pub async fn health(State(state): State<SharedState>) -> ApiResult<Json<HealthResponse>> {
    let _timer = super::RequestTimer::start("health", None);

    let (collection_count, arcs) = {
        let manager = state.collections.read().await;
        (manager.len(), manager.all_arcs())
    };

    let mut total_vectors = 0usize;
    for arc in arcs {
        let storage = arc.lock().await;
        total_vectors += storage.metadata.count_active().unwrap_or(0);
    }

    Ok(Json(HealthResponse {
        status: "ok".into(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        vector_count: total_vectors,
        collections: collection_count,
    }))
}
