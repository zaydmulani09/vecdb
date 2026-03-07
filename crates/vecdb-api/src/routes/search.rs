use axum::{
    extract::{Path, State},
    Json,
};
use vecdb_core::{
    CollectionConfig, FusionStrategy, QueryPlanner, SearchRequest, SqlParser, VecDbError,
};

use crate::{
    error::{ApiError, ApiResult},
    middleware::validate_collection_name,
    state::SharedState,
    types::{
        DenseSearchRequest, ExplainRequest, ExplainResponse, HybridSearchRequest, SearchResponse,
        SearchResultResponse, SparseSearchRequest, SqlQueryRequest,
    },
};

pub async fn search_dense(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<DenseSearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let _timer = super::RequestTimer::start("search_dense", Some(&name));

    validate_collection_name(&name)?;

    if req.vector.is_empty() {
        return Err(ApiError(VecDbError::InvalidQuery(
            "vector must not be empty".into(),
        )));
    }

    let k = req.k.unwrap_or(10);
    if k == 0 || k > 10_000 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "k must be between 1 and 10000 (inclusive)".into(),
        )));
    }

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let search_start = std::time::Instant::now();
    let storage = arc.lock().await;
    let results = storage.search_dense(&req.vector, k)?;
    let elapsed_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let result_count = results.len();

    metrics::histogram!(
        crate::metrics::SEARCH_DURATION_MS,
        "type" => "dense",
        "collection" => name.clone(),
    )
    .record(elapsed_ms);
    metrics::gauge!(
        crate::metrics::SEARCH_RESULTS_COUNT,
        "type" => "dense",
        "collection" => name.clone(),
    )
    .set(result_count as f64);
    tracing::info!(
        collection = %name,
        result_count,
        elapsed_ms,
        "dense search completed"
    );

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
        count: result_count,
        time_ms: elapsed_ms as u64,
    }))
}

pub async fn search_sparse(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<SparseSearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let _timer = super::RequestTimer::start("search_sparse", Some(&name));

    validate_collection_name(&name)?;

    if req.query.is_empty() {
        return Err(ApiError(VecDbError::InvalidQuery(
            "query must not be empty".into(),
        )));
    }

    let k = req.k.unwrap_or(10);
    if k == 0 || k > 10_000 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "k must be between 1 and 10000 (inclusive)".into(),
        )));
    }

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let search_start = std::time::Instant::now();
    let storage = arc.lock().await;
    let results = storage.search_sparse(&req.query, k)?;
    let elapsed_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let result_count = results.len();

    metrics::histogram!(
        crate::metrics::SEARCH_DURATION_MS,
        "type" => "sparse",
        "collection" => name.clone(),
    )
    .record(elapsed_ms);
    metrics::gauge!(
        crate::metrics::SEARCH_RESULTS_COUNT,
        "type" => "sparse",
        "collection" => name.clone(),
    )
    .set(result_count as f64);
    tracing::info!(
        collection = %name,
        result_count,
        elapsed_ms,
        "sparse search completed"
    );

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
        count: result_count,
        time_ms: elapsed_ms as u64,
    }))
}

pub async fn search_hybrid(
    State(state): State<SharedState>,
    Path(name): Path<String>,
    Json(req): Json<HybridSearchRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let _timer = super::RequestTimer::start("search_hybrid", Some(&name));

    validate_collection_name(&name)?;

    if req.vector.is_none() && req.query.is_none() {
        return Err(ApiError(VecDbError::InvalidQuery(
            "at least one of 'vector' or 'query' must be provided".into(),
        )));
    }

    let k = req.k.unwrap_or(10);
    if k == 0 || k > 10_000 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "k must be between 1 and 10000 (inclusive)".into(),
        )));
    }

    let alpha = req.alpha.unwrap_or(0.7);
    if !(0.0..=1.0).contains(&alpha) {
        return Err(ApiError(VecDbError::InvalidQuery(
            "alpha must be between 0.0 and 1.0 (inclusive)".into(),
        )));
    }

    let strategy = match req.strategy.as_deref().unwrap_or("weighted_sum") {
        "weighted_sum" => FusionStrategy::WeightedSum,
        "rrf" => FusionStrategy::ReciprocalRankFusion { k: 60.0 },
        other => {
            return Err(ApiError(VecDbError::InvalidQuery(format!(
                "unknown strategy '{}'; valid: weighted_sum, rrf",
                other
            ))))
        }
    };

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&name)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(name.clone())))?
    };

    let search_start = std::time::Instant::now();
    let storage = arc.lock().await;
    let results = storage.search_hybrid(
        req.vector.as_ref(),
        req.query.as_deref(),
        k,
        alpha,
        strategy,
    )?;
    let elapsed_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let result_count = results.len();

    metrics::histogram!(
        crate::metrics::SEARCH_DURATION_MS,
        "type" => "hybrid",
        "collection" => name.clone(),
    )
    .record(elapsed_ms);
    metrics::gauge!(
        crate::metrics::SEARCH_RESULTS_COUNT,
        "type" => "hybrid",
        "collection" => name.clone(),
    )
    .set(result_count as f64);
    tracing::info!(
        collection = %name,
        alpha,
        result_count,
        elapsed_ms,
        "hybrid search completed"
    );

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
        count: result_count,
        time_ms: elapsed_ms as u64,
    }))
}

pub async fn query_sql(
    State(state): State<SharedState>,
    Json(req): Json<SqlQueryRequest>,
) -> ApiResult<Json<SearchResponse>> {
    let _timer = super::RequestTimer::start("sql_query", None);

    if req.sql.is_empty() {
        return Err(ApiError(VecDbError::InvalidQuery(
            "sql must not be empty".into(),
        )));
    }

    if req.sql.len() > 4096 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "sql must not exceed 4096 characters".into(),
        )));
    }

    // Parse SQL to extract collection (table) name from the FROM clause.
    let stmt = SqlParser::parse(&req.sql)?;
    let col = stmt.table.clone();

    let arc = {
        let manager = state.collections.read().await;
        manager
            .get(&col)
            .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(col.clone())))?
    };

    let search_start = std::time::Instant::now();
    let storage = arc.lock().await;
    let results = storage.execute_sql(&req.sql)?;
    let elapsed_ms = search_start.elapsed().as_secs_f64() * 1000.0;
    let result_count = results.len();

    Ok(Json(SearchResponse {
        results: results
            .into_iter()
            .map(SearchResultResponse::from)
            .collect(),
        count: result_count,
        time_ms: elapsed_ms as u64,
    }))
}

pub async fn explain_query(
    State(state): State<SharedState>,
    Json(req): Json<ExplainRequest>,
) -> ApiResult<Json<ExplainResponse>> {
    let _timer = super::RequestTimer::start("explain", None);

    if let Some(sql) = &req.sql {
        // Extract collection name from SQL FROM clause.
        let stmt = SqlParser::parse(sql)?;
        let col = stmt.table.clone();

        let arc = {
            let manager = state.collections.read().await;
            manager
                .get(&col)
                .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(col.clone())))?
        };
        let storage = arc.lock().await;
        let vector_count = storage.metadata.count_active()?;
        let plan = storage.planner.plan_sql(sql, vector_count)?;
        let plan_str = storage.planner.explain(&plan);

        Ok(Json(ExplainResponse {
            plan: plan_str,
            index_type: plan.index_type.to_string(),
            use_hybrid: plan.use_hybrid,
            candidate_k: plan.candidate_k,
            output_k: plan.output_k,
            estimated_cost: plan.estimated_cost,
        }))
    } else if req.vector.is_some() || req.query.is_some() {
        let col = req.collection.clone().unwrap_or_default();

        let (plan, plan_str) = if col.is_empty() {
            // No collection specified — use an aggregate count and a default planner.
            let arcs = {
                let manager = state.collections.read().await;
                manager.all_arcs()
            };
            let mut total_vectors = 0usize;
            for arc in &arcs {
                let s = arc.lock().await;
                total_vectors += s.metadata.count_active().unwrap_or(0);
            }
            let planner = QueryPlanner::new(CollectionConfig::new("explain", 1536));
            let search_req = SearchRequest {
                vector: req.vector.clone(),
                query_text: req.query.clone(),
                k: req.k.unwrap_or(10),
                alpha: 0.7,
                collection: String::new(),
                filter: None,
            };
            let p = planner.plan_search(&search_req, total_vectors)?;
            let ps = planner.explain(&p);
            (p, ps)
        } else {
            let arc = {
                let manager = state.collections.read().await;
                manager
                    .get(&col)
                    .ok_or_else(|| ApiError(VecDbError::CollectionNotFound(col.clone())))?
            };
            let storage = arc.lock().await;
            let vector_count = storage.metadata.count_active()?;
            let search_req = SearchRequest {
                vector: req.vector.clone(),
                query_text: req.query.clone(),
                k: req.k.unwrap_or(10),
                alpha: 0.7,
                collection: col,
                filter: None,
            };
            let p = storage.planner.plan_search(&search_req, vector_count)?;
            let ps = storage.planner.explain(&p);
            (p, ps)
        };

        Ok(Json(ExplainResponse {
            plan: plan_str,
            index_type: plan.index_type.to_string(),
            use_hybrid: plan.use_hybrid,
            candidate_k: plan.candidate_k,
            output_k: plan.output_k,
            estimated_cost: plan.estimated_cost,
        }))
    } else {
        Err(ApiError(VecDbError::InvalidQuery(
            "provide 'sql' or at least one of 'vector'/'query'".into(),
        )))
    }
}
