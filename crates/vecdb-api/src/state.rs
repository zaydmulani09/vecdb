use std::sync::Arc;

use metrics_exporter_prometheus::PrometheusHandle;
use tokio::sync::RwLock;

use crate::collection_manager::CollectionManager;

pub struct AppState {
    pub collections: RwLock<CollectionManager>,
    pub metrics_handle: PrometheusHandle,
    pub api_key: Option<String>,
}

pub type SharedState = Arc<AppState>;
