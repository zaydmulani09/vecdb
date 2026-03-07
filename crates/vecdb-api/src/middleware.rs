use axum::{
    extract::{Request, State},
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use vecdb_core::VecDbError;

use crate::{error::ApiError, state::SharedState};

/// Validate that a collection name contains only `[a-zA-Z0-9_-]` and is 1–64 chars.
/// Returns `Err(ApiError)` (HTTP 400) on violation so handlers can use `?` directly.
pub fn validate_collection_name(name: &str) -> Result<(), ApiError> {
    if name.is_empty() || name.len() > 64 {
        return Err(ApiError(VecDbError::InvalidQuery(
            "collection name must be between 1 and 64 characters".into(),
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(ApiError(VecDbError::InvalidQuery(
            "collection name must contain only [a-zA-Z0-9_-]".into(),
        )));
    }
    Ok(())
}

/// Axum middleware that appends security response headers to every response.
pub async fn security_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        HeaderValue::from_static("0"),
    );
    headers.insert(
        header::HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );
    response
}

pub async fn auth_middleware(
    State(state): State<SharedState>,
    request: Request,
    next: Next,
) -> Response {
    let required_key = match &state.api_key {
        None => return next.run(request).await,
        Some(k) => k.clone(),
    };

    let provided_key = request
        .headers()
        .get("X-Api-Key")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if provided_key != required_key {
        let body = json!({
            "error": {
                "code": "unauthorized",
                "message": "invalid or missing API key — provide X-Api-Key header"
            }
        });
        return (StatusCode::UNAUTHORIZED, Json(body)).into_response();
    }

    next.run(request).await
}
