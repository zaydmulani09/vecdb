use axum::{
    extract::{Path, State},
    Json,
};
use chrono::Utc;
use vecdb_core::{storage::UpsertResult, VecDbError, VectorRecord};

use crate::{
    error::{ApiError, ApiResult},
    middleware::validate_collection_name,
    state::SharedState,
    types::{
        DeleteVectorsRequest, DeleteVectorsResponse, GetVectorResponse, UpsertError,
        UpsertVectorsRequest, UpsertVectorsResponse,
    },
};

pub async fn upsert_vectors(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<UpsertVectorsRequest>,
) -> ApiResult<Json<UpsertVectorsResponse>> {
    let _timer = super::RequestTimer::start("upsert_vectors", Some(&name));

    validate_collection_name(&name)?;

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let start = std::time::Instant::now();
    let mut storage = arc.lock().await;
    let mut inserted = 0usize;
    let mut updated = 0usize;
    let mut errors: Vec<UpsertError> = Vec::new();

    for input in req.records {
        let record = VectorRecord {
            id: input.id.clone(),
            vector: input.vector,
            payload: input
                .payload
                .unwrap_or_else(|| serde_json::Value::Object(Default::default())),
            text: input.text,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        match storage.upsert(record) {
            Ok(UpsertResult::Inserted) => inserted += 1,
            Ok(UpsertResult::Updated) => updated += 1,
            Err(e) => errors.push(UpsertError {
                id: input.id,
                error: e.to_string(),
            }),
        }
    }

    metrics::counter!(
        crate::metrics::UPSERT_TOTAL,
        "collection" => name.clone(),
    )
    .increment((inserted + updated) as u64);

    Ok(Json(UpsertVectorsResponse {
        inserted,
        updated,
        errors,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

pub async fn delete_vectors(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<DeleteVectorsRequest>,
) -> ApiResult<Json<DeleteVectorsResponse>> {
    let _timer = super::RequestTimer::start("delete_vectors", Some(&name));

    validate_collection_name(&name)?;

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let start = std::time::Instant::now();
    let mut storage = arc.lock().await;
    let mut deleted = 0usize;
    let mut errors: Vec<UpsertError> = Vec::new();

    for id in req.ids {
        match storage.delete(&id) {
            Ok(()) => deleted += 1,
            Err(e) => errors.push(UpsertError {
                id,
                error: e.to_string(),
            }),
        }
    }

    metrics::counter!(
        crate::metrics::DELETE_TOTAL,
        "collection" => name.clone(),
    )
    .increment(deleted as u64);

    Ok(Json(DeleteVectorsResponse {
        deleted,
        errors,
        time_ms: start.elapsed().as_millis() as u64,
    }))
}

pub async fn get_vector(
    State(state): State<SharedState>,
    Path((name, id)): Path<(String, String)>,
) -> ApiResult<Json<GetVectorResponse>> {
    let _timer = super::RequestTimer::start("get_vector", Some(&name));

    validate_collection_name(&name)?;

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let storage = arc.lock().await;
    let (mmap_idx, record) = storage.metadata.get(&id)?;
    let vector = storage.vectors.get(mmap_idx)?;

    Ok(Json(GetVectorResponse {
        id: record.id,
        vector,
        payload: record.payload,
        text: record.text,
    }))
}
