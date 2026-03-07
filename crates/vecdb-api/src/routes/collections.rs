use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde_json::json;
use vecdb_core::{CollectionConfig, DistanceMetric, IndexType, VecDbError};

use crate::{
    error::{ApiError, ApiResult},
    middleware::validate_collection_name,
    state::SharedState,
    types::{CollectionResponse, CreateCollectionRequest},
};

pub async fn create_collection(
    State(state): State<SharedState>,
    Json(req): Json<CreateCollectionRequest>,
) -> ApiResult<(StatusCode, Json<CollectionResponse>)> {
    let _timer = super::RequestTimer::start("create_collection", None);

    validate_collection_name(&req.name)?;

    if req.dimension == 0 || req.dimension > 65_536 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "dimension must be between 1 and 65536".into(),
        )));
    }

    let metric = match req.metric.as_deref().unwrap_or("cosine") {
        "cosine" => DistanceMetric::Cosine,
        "euclidean" => DistanceMetric::Euclidean,
        "dot" => DistanceMetric::DotProduct,
        other => {
            return Err(ApiError(VecDbError::InvalidQuery(format!(
                "unknown metric '{}'; valid: cosine, euclidean, dot",
                other
            ))))
        }
    };

    let index_type = match req.index_type.as_deref().unwrap_or("hnsw") {
        "hnsw" => IndexType::HNSW,
        "ivf" => IndexType::IVF,
        other => {
            return Err(ApiError(VecDbError::InvalidQuery(format!(
                "unknown index_type '{}'; valid: hnsw, ivf",
                other
            ))))
        }
    };

    let mut config = CollectionConfig::new(req.name.clone(), req.dimension);
    config.metric = metric;
    config.index_type = index_type;
    if let Some(m) = req.hnsw_m {
        config.hnsw_m = m;
    }
    if let Some(ef) = req.hnsw_ef_construction {
        config.hnsw_ef_construction = ef;
    }
    if let Some(k1) = req.bm25_k1 {
        config.bm25_k1 = k1;
    }
    if let Some(b) = req.bm25_b {
        config.bm25_b = b;
    }

    // Create collection — returns CollectionAlreadyExists (→ 409) if duplicate.
    {
        let mut manager = state.collections.write().await;
        manager.create(&config).await?;
    }

    // Fetch stats from the newly-created storage.
    let stats = {
        let manager = state.collections.read().await;
        let arc = manager.get(&config.name).ok_or_else(|| {
            ApiError(VecDbError::StorageError(
                "collection vanished after create".into(),
            ))
        })?;
        drop(manager);
        let storage = arc.lock().await;
        storage.stats()?
    };

    Ok((
        StatusCode::CREATED,
        Json(CollectionResponse::from_config_and_stats(&config, &stats)),
    ))
}

pub async fn list_collections(
    State(state): State<SharedState>,
) -> ApiResult<Json<serde_json::Value>> {
    let _timer = super::RequestTimer::start("list_collections", None);

    let (names, arcs) = {
        let manager = state.collections.read().await;
        let names = manager.list();
        let arcs: Vec<_> = names.iter().filter_map(|n| manager.get(n)).collect();
        (names, arcs)
    };

    let mut collections: Vec<CollectionResponse> = Vec::with_capacity(names.len());
    for arc in arcs {
        let storage = arc.lock().await;
        let config = storage.config.clone();
        let stats = storage.stats()?;
        collections.push(CollectionResponse::from_config_and_stats(&config, &stats));
    }

    let count = collections.len();
    Ok(Json(json!({ "collections": collections, "count": count })))
}

pub async fn get_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> ApiResult<Json<CollectionResponse>> {
    let _timer = super::RequestTimer::start("get_collection", Some(&name));

    validate_collection_name(&name)?;

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let storage = arc.lock().await;
    let config = storage.config.clone();
    let stats = storage.stats()?;
    Ok(Json(CollectionResponse::from_config_and_stats(
        &config, &stats,
    )))
}

pub async fn delete_collection(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> ApiResult<Json<serde_json::Value>> {
    let _timer = super::RequestTimer::start("delete_collection", Some(&name));

    validate_collection_name(&name)?;

    let mut manager = state.collections.write().await;
    manager.delete(&name).await?;

    Ok(Json(json!({ "deleted": true, "name": name })))
}
