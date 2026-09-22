//! Embedded, no-server API — the primary way to use vecdb.
//!
//! Open a directory as a [`Db`], create or open a [`Collection`], then
//! insert and query in-process. No network, no daemon, no docker.
//!
//! ```no_run
//! use vecdb_core::Db;
//! use serde_json::json;
//!
//! let db = Db::open("data.vecdb")?;
//! let mut docs = db.open_or_create("docs", 4)?;
//! docs.insert("a", vec![0.1, 0.2, 0.3, 0.4], json!({ "title": "hello" }))?;
//! let hits = docs.query(&[0.1, 0.2, 0.3, 0.4], 5)?;
//! # Ok::<(), vecdb_core::VecDbError>(())
//! ```

use std::path::{Path, PathBuf};

use chrono::Utc;
use serde_json::Value;

use crate::errors::{Result, VecDbError};
use crate::hybrid::FusionStrategy;
use crate::storage::{Storage, UpsertResult};
use crate::types::{CollectionConfig, IndexStats, SearchResult, Vector, VectorId, VectorRecord};

/// A vecdb database: a handle to a directory that holds one or more
/// collections. Cheap to clone-by-reopen; owns no file handles itself —
/// each [`Collection`] it hands out owns its own storage.
#[derive(Debug, Clone)]
pub struct Db {
    root: PathBuf,
}

impl Db {
    /// Open (creating if absent) a database rooted at `path`.
    ///
    /// `path` is a directory. A `.vecdb` suffix is conventional but not
    /// required — the directory holds the per-collection files.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| {
            VecDbError::StorageError(format!("create_dir_all({root:?}) failed: {e}"))
        })?;
        Ok(Self { root })
    }

    /// Root directory of this database.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Create a new collection with default HNSW config. Errors if a
    /// collection of that name already exists.
    pub fn create_collection(
        &self,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<Collection> {
        self.create_collection_with(CollectionConfig::new(name, dimension))
    }

    /// Create a new collection from a full [`CollectionConfig`].
    pub fn create_collection_with(&self, config: CollectionConfig) -> Result<Collection> {
        if self.collection_exists(&config.name) {
            return Err(VecDbError::CollectionAlreadyExists(config.name));
        }
        let storage = Storage::create(&self.root, &config)?;
        Ok(Collection { storage })
    }

    /// Create a new collection whose in-memory index uses int8 scalar
    /// quantization (~4× smaller index; on-disk vectors stay float32).
    pub fn create_collection_quantized(
        &self,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<Collection> {
        self.create_collection_with(
            CollectionConfig::new(name, dimension)
                .with_quantization(crate::types::Quantization::ScalarInt8),
        )
    }

    /// Create a new collection whose in-memory index uses 1-bit binary
    /// quantization (~32× smaller; coarse Hamming distance — best paired with a
    /// full-precision rerank of top candidates).
    pub fn create_collection_binary(
        &self,
        name: impl Into<String>,
        dimension: usize,
    ) -> Result<Collection> {
        self.create_collection_with(
            CollectionConfig::new(name, dimension)
                .with_quantization(crate::types::Quantization::Binary),
        )
    }

    /// Open an existing collection. Errors with `CollectionNotFound` if absent.
    pub fn collection(&self, name: &str) -> Result<Collection> {
        if !self.collection_exists(name) {
            return Err(VecDbError::CollectionNotFound(name.to_string()));
        }
        let storage = Storage::open(&self.root, name)?;
        Ok(Collection { storage })
    }

    /// Open the collection if it exists, otherwise create it with default
    /// HNSW config at `dimension`.
    pub fn open_or_create(&self, name: &str, dimension: usize) -> Result<Collection> {
        if self.collection_exists(name) {
            self.collection(name)
        } else {
            self.create_collection(name, dimension)
        }
    }

    /// Whether a collection with this name exists on disk (by its `.db` file).
    pub fn collection_exists(&self, name: &str) -> bool {
        self.root.join(format!("{name}.db")).exists()
    }

    /// Sorted names of all collections in this database (scans `*.db` files).
    pub fn list_collections(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let entries = std::fs::read_dir(&self.root)
            .map_err(|e| VecDbError::StorageError(format!("read_dir failed: {e}")))?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "db").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    names.push(stem.to_string());
                }
            }
        }
        names.sort();
        Ok(names)
    }

    /// Delete a collection and all its files from disk.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        if !self.collection_exists(name) {
            return Err(VecDbError::CollectionNotFound(name.to_string()));
        }
        for suffix in &[
            ".db",
            ".wal",
            ".vectors",
            ".hnsw.json",
            ".ivf.json",
            ".sq.json",
            ".bq.json",
            ".sparse.json",
        ] {
            let path = self.root.join(format!("{name}{suffix}"));
            if path.exists() {
                std::fs::remove_file(&path).map_err(|e| {
                    VecDbError::StorageError(format!("remove {path:?} failed: {e}"))
                })?;
            }
        }
        Ok(())
    }
}

/// A single collection: an owned, in-process handle to one collection's
/// storage. Insert with `&mut`, query with `&`. Data is durable via the WAL
/// on every insert; call [`Collection::flush`] (or drop the handle) to also
/// persist the dense/sparse indexes for a fast reopen.
pub struct Collection {
    storage: Storage,
}

impl Collection {
    /// Insert or update a vector with JSON metadata and no text. Timestamps
    /// are set to now.
    pub fn insert(
        &mut self,
        id: impl Into<VectorId>,
        vector: Vector,
        payload: Value,
    ) -> Result<UpsertResult> {
        self.upsert(id, vector, payload, None)
    }

    /// Insert or update a vector with JSON metadata and full-text (enables
    /// BM25 sparse and hybrid search for this record).
    pub fn insert_text(
        &mut self,
        id: impl Into<VectorId>,
        vector: Vector,
        payload: Value,
        text: impl Into<String>,
    ) -> Result<UpsertResult> {
        self.upsert(id, vector, payload, Some(text.into()))
    }

    fn upsert(
        &mut self,
        id: impl Into<VectorId>,
        vector: Vector,
        payload: Value,
        text: Option<String>,
    ) -> Result<UpsertResult> {
        let now = Utc::now();
        self.storage.upsert(VectorRecord {
            id: id.into(),
            vector,
            payload,
            text,
            created_at: now,
            updated_at: now,
        })
    }

    /// Insert or update a fully-specified record.
    pub fn upsert_record(&mut self, record: VectorRecord) -> Result<UpsertResult> {
        self.storage.upsert(record)
    }

    /// Bulk-insert many `(id, vector, payload)` triples, building the index once
    /// at the end. Far faster than repeated [`Collection::insert`] for loading a
    /// large collection. Returns the number inserted.
    pub fn insert_batch(&mut self, items: Vec<(VectorId, Vector, Value)>) -> Result<usize> {
        let now = Utc::now();
        let records = items
            .into_iter()
            .map(|(id, vector, payload)| VectorRecord {
                id,
                vector,
                payload,
                text: None,
                created_at: now,
                updated_at: now,
            })
            .collect();
        self.storage.bulk_upsert(records)
    }

    /// Delete a record by id.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        self.storage.delete(&id.to_string())
    }

    /// Dense k-NN search over the vector index.
    pub fn query(&self, vector: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.storage.search_dense(&vector.to_vec(), k)
    }

    /// Dense k-NN search restricted to records whose payload matches `filter`,
    /// with predicate pushdown: a selective filter scans only the matching
    /// vectors (exact, ~proportional to the matches) rather than searching the
    /// whole index and discarding non-matches.
    ///
    /// `filter` is the same JSON predicate shape accepted elsewhere, e.g.
    /// `json!({ "genre": "sci-fi" })` or `json!({ "year": { "$gte": 2000 } })`.
    pub fn query_filtered(
        &self,
        vector: &[f32],
        k: usize,
        filter: &Value,
    ) -> Result<Vec<SearchResult>> {
        self.storage
            .search_dense_filtered(&vector.to_vec(), k, filter)
    }

    /// Full-text BM25 search over the sparse index.
    pub fn query_text(&self, text: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.storage.search_sparse(text, k)
    }

    /// Two-stage hybrid search. At least one of `vector` / `text` required.
    /// `alpha` weights dense vs sparse (`alpha * dense + (1-alpha) * sparse`).
    pub fn query_hybrid(
        &self,
        vector: Option<&[f32]>,
        text: Option<&str>,
        k: usize,
        alpha: f32,
    ) -> Result<Vec<SearchResult>> {
        let owned = vector.map(|v| v.to_vec());
        self.storage
            .search_hybrid(owned.as_ref(), text, k, alpha, FusionStrategy::WeightedSum)
    }

    /// Run a `VECTOR_SIM` SQL query against this collection.
    pub fn sql(&self, sql: &str) -> Result<Vec<SearchResult>> {
        self.storage.execute_sql(sql)
    }

    /// Number of active (non-deleted) vectors.
    pub fn len(&self) -> Result<usize> {
        Ok(self.storage.stats()?.vector_count)
    }

    /// Whether the collection has no active vectors.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Collection statistics (vector count, on-disk bytes, dimension, …).
    pub fn stats(&self) -> Result<IndexStats> {
        self.storage.stats()
    }

    /// Collection name.
    pub fn name(&self) -> &str {
        &self.storage.config.name
    }

    /// Persist indexes and checkpoint the WAL. Data is already durable via the
    /// WAL without this; flushing makes the next reopen fast (no index rebuild).
    pub fn flush(&mut self) -> Result<()> {
        self.storage.save_indexes()?;
        self.storage.checkpoint()
    }
}

impl Drop for Collection {
    fn drop(&mut self) {
        // Best-effort: persist indexes so reopen skips the rebuild. WAL already
        // guarantees durability, so failures here are non-fatal.
        let _ = self.storage.save_indexes();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn open_create_insert_query() {
        let dir = tempdir().unwrap();
        let db = Db::open(dir.path()).unwrap();
        let mut c = db.create_collection("docs", 3).unwrap();
        c.insert("a", vec![1.0, 0.0, 0.0], json!({ "n": 1 }))
            .unwrap();
        c.insert("b", vec![0.0, 1.0, 0.0], json!({ "n": 2 }))
            .unwrap();
        let hits = c.query(&[1.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits[0].id, "a");
    }

    #[test]
    fn persist_and_reopen() {
        let dir = tempdir().unwrap();
        {
            let db = Db::open(dir.path()).unwrap();
            let mut c = db.create_collection("k", 2).unwrap();
            c.insert("x", vec![1.0, 2.0], json!({})).unwrap();
            c.flush().unwrap();
        }
        let db = Db::open(dir.path()).unwrap();
        let c = db.collection("k").unwrap();
        assert_eq!(c.len().unwrap(), 1);
        assert_eq!(c.query(&[1.0, 2.0], 1).unwrap()[0].id, "x");
    }

    #[test]
    fn open_or_create_is_idempotent() {
        let dir = tempdir().unwrap();
        let db = Db::open(dir.path()).unwrap();
        {
            let mut c = db.open_or_create("c", 2).unwrap();
            c.insert("1", vec![0.0, 1.0], json!({})).unwrap();
            c.flush().unwrap();
        }
        let c = db.open_or_create("c", 2).unwrap();
        assert_eq!(c.len().unwrap(), 1);
    }

    #[test]
    fn create_existing_errors() {
        let dir = tempdir().unwrap();
        let db = Db::open(dir.path()).unwrap();
        let _c = db.create_collection("dup", 2).unwrap();
        drop(_c);
        assert!(matches!(
            db.create_collection("dup", 2),
            Err(VecDbError::CollectionAlreadyExists(_))
        ));
    }

    #[test]
    fn list_and_drop() {
        let dir = tempdir().unwrap();
        let db = Db::open(dir.path()).unwrap();
        db.create_collection("a", 2).unwrap().flush().unwrap();
        db.create_collection("b", 2).unwrap().flush().unwrap();
        assert_eq!(db.list_collections().unwrap(), vec!["a", "b"]);
        db.drop_collection("a").unwrap();
        assert_eq!(db.list_collections().unwrap(), vec!["b"]);
        assert!(!db.collection_exists("a"));
    }

    #[test]
    fn hybrid_and_text_query() {
        let dir = tempdir().unwrap();
        let db = Db::open(dir.path()).unwrap();
        let mut c = db.create_collection("h", 3).unwrap();
        c.insert_text("d1", vec![1.0, 0.0, 0.0], json!({}), "rust vector database")
            .unwrap();
        c.insert_text(
            "d2",
            vec![0.0, 1.0, 0.0],
            json!({}),
            "python machine learning",
        )
        .unwrap();
        let text_hits = c.query_text("rust database", 5).unwrap();
        assert_eq!(text_hits[0].id, "d1");
        let hybrid = c
            .query_hybrid(Some(&[1.0, 0.0, 0.0]), Some("rust"), 2, 0.5)
            .unwrap();
        assert!(!hybrid.is_empty());
    }
}
