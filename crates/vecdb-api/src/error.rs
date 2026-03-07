use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use vecdb_core::VecDbError;

pub struct ApiError(pub VecDbError);

impl ApiError {
    pub fn kind_str(&self) -> &'static str {
        match &self.0 {
            VecDbError::NotFound { .. } => "not_found",
            VecDbError::CollectionNotFound(_) => "collection_not_found",
            VecDbError::CollectionAlreadyExists(_) => "collection_exists",
            VecDbError::DimensionMismatch { .. } => "dimension_mismatch",
            VecDbError::InvalidQuery(_) => "invalid_query",
            VecDbError::StorageError(_) => "storage_error",
            VecDbError::IndexError(_) => "index_error",
            VecDbError::SparseError(_) => "sparse_error",
            VecDbError::MetadataError(_) => "metadata_error",
            VecDbError::SerializationError(_) => "serialization_error",
            VecDbError::IoError(_) => "io_error",
            VecDbError::SqliteError(_) => "sqlite_error",
        }
    }
}

impl From<VecDbError> for ApiError {
    fn from(e: VecDbError) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = self.kind_str();
        metrics::counter!(crate::metrics::ERRORS_TOTAL, "kind" => kind).increment(1);

        let (status, code, message) = match &self.0 {
            VecDbError::NotFound { id } => (
                StatusCode::NOT_FOUND,
                "not_found",
                format!("document '{}' not found", id),
            ),
            VecDbError::CollectionNotFound(name) => (
                StatusCode::NOT_FOUND,
                "collection_not_found",
                format!("collection '{}' not found", name),
            ),
            VecDbError::CollectionAlreadyExists(name) => (
                StatusCode::CONFLICT,
                "collection_exists",
                format!("collection '{}' already exists", name),
            ),
            VecDbError::DimensionMismatch { expected, got } => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "dimension_mismatch",
                format!("expected dimension {}, got {}", expected, got),
            ),
            VecDbError::InvalidQuery(msg) => {
                (StatusCode::BAD_REQUEST, "invalid_query", msg.clone())
            }
            VecDbError::StorageError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "storage_error",
                msg.clone(),
            ),
            VecDbError::IndexError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "index_error",
                msg.clone(),
            ),
            VecDbError::SparseError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "sparse_error",
                msg.clone(),
            ),
            VecDbError::MetadataError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "metadata_error",
                msg.clone(),
            ),
            VecDbError::SerializationError(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "serialization_error",
                msg.clone(),
            ),
            VecDbError::IoError(e) => {
                (StatusCode::INTERNAL_SERVER_ERROR, "io_error", e.to_string())
            }
            VecDbError::SqliteError(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "sqlite_error",
                e.to_string(),
            ),
        };

        let body = json!({
            "error": {
                "code": code,
                "message": message,
            }
        });

        (status, Json(body)).into_response()
    }
}

pub type ApiResult<T> = std::result::Result<T, ApiError>;
