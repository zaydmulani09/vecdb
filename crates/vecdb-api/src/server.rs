use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::middleware;
use tokio::sync::RwLock;
use tower_governor::{governor::GovernorConfigBuilder, GovernorLayer};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use vecdb_core::ServerConfig;

use crate::{
    collection_manager::CollectionManager,
    middleware::{auth_middleware, security_headers},
    routes::router,
    state::AppState,
};

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("shutdown signal received — draining in-flight requests");
}

pub async fn run(config: ServerConfig) -> anyhow::Result<()> {
    // Install metrics recorder first — must happen before router is built.
    let metrics_handle = crate::metrics::install_recorder();

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                config
                    .log_level
                    .parse()
                    .unwrap_or_else(|_| "info".parse().unwrap())
            }),
        )
        .try_init();

    let data_dir = &config.data_dir;
    std::fs::create_dir_all(data_dir)?;

    let mut manager = CollectionManager::new(data_dir.clone());
    let n = manager.load_existing().await?;
    tracing::info!(
        "vecdb started: {} collection(s) loaded from '{}'",
        n,
        data_dir.display()
    );

    let state = Arc::new(AppState {
        collections: RwLock::new(manager),
        metrics_handle,
        api_key: config.api_key.clone(),
    });

    // Rate-limiting: 1000 requests per minute per IP (≈17/s with burst of 1000).
    let governor_conf = Arc::new(
        GovernorConfigBuilder::default()
            .per_second(17)
            .burst_size(1000)
            .finish()
            .expect("valid rate-limit governor config"),
    );

    let timeout = Duration::from_millis(config.query_timeout_ms);

    // Layer ordering: last .layer() = outermost (processes request first).
    // Desired request path:
    //   RequestBodyLimit → Governor → SecurityHeaders → Trace → Timeout → Cors → Auth → handler
    let app = router(state.clone())
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        )) // 7 — innermost
        .layer(CorsLayer::permissive()) // 6
        .layer(TimeoutLayer::new(timeout)) // 5
        .layer(TraceLayer::new_for_http()) // 4
        .layer(middleware::from_fn(security_headers)) // 3
        .layer(GovernorLayer { config: governor_conf }) // 2
        .layer(RequestBodyLimitLayer::new(32 * 1024 * 1024)); // 1 — outermost (32 MiB)

    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("vecdb listening on http://{}", addr);

    // ConnectInfo<SocketAddr> is needed by PeerIpKeyExtractor inside GovernorLayer.
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;

    // Drain complete — flush all open collections before exit.
    tracing::info!("flushing all collections before exit");
    let manager = state.collections.write().await;
    for name in manager.list() {
        if let Some(storage_arc) = manager.get(&name) {
            let mut storage = storage_arc.lock().await;
            match storage.checkpoint() {
                Ok(_) => tracing::info!("checkpointed '{}'", name),
                Err(e) => tracing::error!("checkpoint failed for '{}': {}", name, e),
            }
            match storage.save_indexes() {
                Ok(_) => tracing::info!("saved indexes for '{}'", name),
                Err(e) => tracing::error!("save_indexes failed for '{}': {}", name, e),
            }
        }
    }
    tracing::info!("vecdb shutdown complete");

    Ok(())
}
