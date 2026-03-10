use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

// ── Metric name constants ──────────────────────────────────────────────────

pub const REQUESTS_TOTAL: &str = "vecdb_requests_total";
pub const ERRORS_TOTAL: &str = "vecdb_errors_total";
pub const UPSERT_TOTAL: &str = "vecdb_upsert_total";
pub const DELETE_TOTAL: &str = "vecdb_delete_total";

pub const REQUEST_DURATION_MS: &str = "vecdb_request_duration_ms";
pub const SEARCH_DURATION_MS: &str = "vecdb_search_duration_ms";

pub const SEARCH_RESULTS_COUNT: &str = "vecdb_search_results_count";
pub const INDEX_SIZE_VECTORS: &str = "vecdb_index_size_vectors";

// ── Recorder init ──────────────────────────────────────────────────────────

/// Install the global Prometheus recorder. Must be called exactly once per
/// process before the Axum router is built. Returns the handle used to
/// render the /metrics scrape response.
///
/// Panics if called more than once.
pub fn install_recorder() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}
