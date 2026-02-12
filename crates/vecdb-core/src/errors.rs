use thiserror::Error;

#[derive(Debug, Error)]
pub enum VecDbError {
    #[error("storage error: {0}")]
    StorageError(String),

    #[error("index error: {0}")]
    IndexError(String),

    #[error("sparse index error: {0}")]
    SparseError(String),

    #[error("metadata error: {0}")]
    MetadataError(String),

    #[error("not found: {id}")]
    NotFound { id: String },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("collection not found: {0}")]
    CollectionNotFound(String),

    #[error("collection already exists: {0}")]
    CollectionAlreadyExists(String),

    #[error("invalid query: {0}")]
    InvalidQuery(String),

    #[error("serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SQLite error: {0}")]
    SqliteError(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, VecDbError>;
