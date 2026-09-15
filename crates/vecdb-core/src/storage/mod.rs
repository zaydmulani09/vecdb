pub mod metadata;
pub mod mmap;
pub mod wal;

pub use metadata::MetadataStore;
pub use mmap::MmapVectorStore;
pub use wal::{WalEntry, WriteAheadLog};

use std::path::{Path, PathBuf};

use crate::errors::{Result, VecDbError};
use crate::hybrid::{FusionStrategy, HybridEngine};
use crate::index::{AnyIndex, IndexBackend};
use crate::planner::{PlanExecutor, QueryPlanner};
use crate::sparse::SparseIndex;
use crate::types::{
    CollectionConfig, IndexStats, SearchRequest, SearchResult, Vector, VectorId, VectorRecord,
};

pub enum UpsertResult {
    Inserted,
    Updated,
}

pub struct Storage {
    pub vectors: MmapVectorStore,
    pub wal: WriteAheadLog,
    pub metadata: MetadataStore,
    pub index: AnyIndex,
    pub sparse: SparseIndex,
    pub hybrid: HybridEngine,
    pub planner: QueryPlanner,
    pub config: CollectionConfig,
    data_dir: PathBuf,
}

impl Storage {
    pub fn create(data_dir: &Path, config: &CollectionConfig) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| VecDbError::StorageError(format!("create_dir_all failed: {e}")))?;

        let vectors_path = data_dir.join(format!("{}.vectors", config.name));
        let wal_path = data_dir.join(format!("{}.wal", config.name));
        let metadata_path = data_dir.join(format!("{}.db", config.name));
        let sparse_path = data_dir.join(format!("{}.sparse.json", config.name));

        let vectors = MmapVectorStore::create(&vectors_path, config.dimension, 1024)?;
        let wal = WriteAheadLog::open(&wal_path)?;
        let metadata = MetadataStore::open(&metadata_path)?;
        metadata.save_collection(config)?;
        metadata.ensure_payload_indexes(&config.indexed_payload_fields)?;

        let index = AnyIndex::from_config(config);
        let sparse = SparseIndex::create(&sparse_path);
        let hybrid = HybridEngine::default();
        let planner = QueryPlanner::new(config.clone());

        tracing::info!("Storage created for collection '{}'", config.name);

        Ok(Self {
            vectors,
            wal,
            metadata,
            index,
            sparse,
            hybrid,
            planner,
            config: config.clone(),
            data_dir: data_dir.to_path_buf(),
        })
    }

    pub fn open(data_dir: &Path, collection_name: &str) -> Result<Self> {
        let vectors_path = data_dir.join(format!("{collection_name}.vectors"));
        let wal_path = data_dir.join(format!("{collection_name}.wal"));
        let metadata_path = data_dir.join(format!("{collection_name}.db"));
        let sparse_path = data_dir.join(format!("{collection_name}.sparse.json"));

        let vectors = MmapVectorStore::open(&vectors_path)?;
        let wal = WriteAheadLog::open(&wal_path)?;
        let metadata = MetadataStore::open(&metadata_path)?;
        let config = metadata.load_collection(collection_name)?;
        metadata.ensure_payload_indexes(&config.indexed_payload_fields)?;

        let (index, saved_index_exists) =
            AnyIndex::load_or_create(data_dir, collection_name, &config);

        let sparse = SparseIndex::open(&sparse_path)?;
        let hybrid = HybridEngine::default();
        let planner = QueryPlanner::new(config.clone());

        let mut storage = Self {
            vectors,
            wal,
            metadata,
            index,
            sparse,
            hybrid,
            planner,
            config,
            data_dir: data_dir.to_path_buf(),
        };

        let recovered = storage.recover_from_wal()?;

        // Rebuild dense index if no saved file or WAL entries were replayed.
        if !saved_index_exists || recovered > 0 {
            storage.rebuild_index()?;
        }

        tracing::info!(
            "Storage opened for collection '{}', {} vectors ({} recovered from WAL)",
            collection_name,
            storage.vectors.len(),
            recovered
        );

        Ok(storage)
    }

    pub fn upsert(&mut self, record: VectorRecord) -> Result<UpsertResult> {
        if record.vector.len() != self.config.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.config.dimension,
                got: record.vector.len(),
            });
        }

        let id = record.id.clone();
        let existing = self.metadata.get(&id);

        let (mmap_index, result) = match existing {
            Ok((idx, _)) => (idx, UpsertResult::Updated),
            Err(VecDbError::NotFound { .. }) => {
                let idx = self.vectors.append(&record.vector)?;
                (idx, UpsertResult::Inserted)
            }
            Err(e) => return Err(e),
        };

        // WAL first.
        self.wal.append(&WalEntry::Insert {
            id: id.clone(),
            mmap_index,
            record: record.clone(),
        })?;

        // MMAP overwrite for updates (inserts already written via append above).
        if matches!(result, UpsertResult::Updated) {
            self.vectors.overwrite(mmap_index, &record.vector)?;
        }

        // Metadata.
        self.metadata.upsert(&id, mmap_index, &record)?;

        // Dense index.
        self.index
            .insert(record.id.clone(), record.vector.clone())?;

        // Sparse index — only when text is present.
        if let Some(ref text) = record.text {
            if !text.is_empty() {
                self.sparse.index_document(&id, text)?;
            }
        }

        Ok(result)
    }

    pub fn delete(&mut self, id: &VectorId) -> Result<()> {
        self.wal.append(&WalEntry::Delete { id: id.clone() })?;
        self.metadata.delete(id)?;
        // Soft-delete from dense index (ignore NotFound).
        let _ = self.index.delete(id);
        // Soft-delete from sparse index (ignore NotFound — may have no text).
        let _ = self.sparse.remove_document(id);
        Ok(())
    }

    /// Search the dense HNSW index and hydrate results with metadata payloads.
    pub fn search(&self, query: &crate::types::Vector, k: usize) -> Result<Vec<SearchResult>> {
        let hits = self.index.search(query, k)?;
        let mut results = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            match self.metadata.get(&id) {
                Ok((_, rec)) => results.push(SearchResult {
                    id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                    payload: rec.payload,
                    text: rec.text,
                }),
                Err(_) => continue, // metadata inconsistency — skip
            }
        }
        Ok(results)
    }

    /// Full-text BM25 search using the sparse index.
    pub fn search_sparse(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let hits = self.sparse.search(query, k)?;
        let mut results = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            match self.metadata.get(&id) {
                Ok((_, rec)) => results.push(SearchResult {
                    id,
                    score,
                    dense_score: None,
                    sparse_score: Some(score),
                    payload: rec.payload,
                    text: rec.text,
                }),
                Err(_) => continue,
            }
        }
        Ok(results)
    }

    /// Parse and execute a SQL string against this collection.
    ///
    /// The SQL must include a `VECTOR_SIM(col, [...]) op threshold` predicate.
    /// Optional scalar equality conditions (`AND field = 'value'`) are applied
    /// as post-scan metadata filters.
    pub fn execute_sql(&self, sql: &str) -> Result<Vec<SearchResult>> {
        use crate::planner::sql::{AstConverter, SqlParser};

        let vector_count = self.metadata.count_active()?;

        // Parse once — extract query vector before conversion consumes the stmt.
        let stmt = SqlParser::parse(sql)?;
        let query_vector: Option<Vector> = AstConverter::extract_query_vector(&stmt);

        // Convert AST → PhysicalPlan.
        let converter = AstConverter::new(QueryPlanner::new(self.config.clone()));
        let plan = converter.convert(stmt, vector_count)?;

        tracing::debug!("SQL plan:\n{}", self.planner.explain(&plan));
        PlanExecutor::new(self).execute(&plan, query_vector.as_ref(), None)
    }

    /// Route a `SearchRequest` through the query planner and execute the
    /// resulting physical plan.
    ///
    /// This is the primary search entry point for callers that want the full
    /// planner pipeline (cost model, hybrid decision, candidate oversampling).
    pub fn execute_search(&self, request: &SearchRequest) -> Result<Vec<SearchResult>> {
        let vector_count = self.metadata.count_active()?;
        let plan = self.planner.plan_search(request, vector_count)?;
        tracing::debug!("Query plan:\n{}", self.planner.explain(&plan));
        let executor = PlanExecutor::new(self);
        executor.execute(
            &plan,
            request.vector.as_ref(),
            request.query_text.as_deref(),
        )
    }

    /// Dense HNSW search with full metadata hydration.
    ///
    /// Equivalent to `search` but makes the intent explicit and is used by
    /// callers that distinguish dense vs hybrid paths.
    pub fn search_dense(&self, query_vector: &Vector, k: usize) -> Result<Vec<SearchResult>> {
        if query_vector.len() != self.config.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.config.dimension,
                got: query_vector.len(),
            });
        }
        let hits = self.index.search(query_vector, k)?;
        let mut results = Vec::with_capacity(hits.len());
        for (id, score) in hits {
            match self.metadata.get(&id) {
                Ok((_, rec)) => results.push(SearchResult {
                    id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                    payload: rec.payload,
                    text: rec.text,
                }),
                Err(_) => continue,
            }
        }
        Ok(results)
    }

    /// Dense k-NN search restricted to records whose payload matches `filter`,
    /// with predicate pushdown.
    ///
    /// When the filter is selective (the allowed set is small relative to the
    /// collection), candidates are generated **from the filter**: an exact,
    /// full-precision brute-force scan over only the matching vectors. This
    /// never scores the excluded majority, so a 95%-eliminating filter costs
    /// roughly 5% of the work — and is exact, unlike oversample-then-discard.
    ///
    /// When the filter is weakly selective, it falls back to an ANN search with
    /// a post-filter (few candidates are dropped, so recall stays high).
    pub fn search_dense_filtered(
        &self,
        query: &Vector,
        k: usize,
        filter: &serde_json::Value,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.config.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.config.dimension,
                got: query.len(),
            });
        }

        let allowed = self
            .metadata
            .filter_ids_indexed(filter, &self.config.indexed_payload_fields)?;
        if allowed.is_empty() {
            return Ok(vec![]);
        }
        let n = self.metadata.count_active()?;
        // Brute-force the allowed set when it is small enough that scanning it
        // is cheaper (and exact) than an oversampled ANN search. Covers all
        // high-selectivity filters (the gate case) and small collections.
        let bruteforce_threshold = (n / 4).max(8192);
        let metric = &self.config.metric;

        if allowed.len() <= bruteforce_threshold {
            let mut scored: Vec<(VectorId, f32)> = Vec::with_capacity(allowed.len());
            for (id, mmap_index) in &allowed {
                let v = self.vectors.get(*mmap_index)?;
                let dist = crate::index::compute_distance(query, &v, metric);
                scored.push((id.clone(), crate::index::to_score(dist, metric)));
            }
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);

            let mut results = Vec::with_capacity(scored.len());
            for (id, score) in scored {
                if let Ok((_, rec)) = self.metadata.get(&id) {
                    results.push(SearchResult {
                        id,
                        score,
                        dense_score: Some(score),
                        sparse_score: None,
                        payload: rec.payload,
                        text: rec.text,
                    });
                }
            }
            return Ok(results);
        }

        // Weakly selective: ANN + post-filter.
        let allowed_set: std::collections::HashSet<VectorId> =
            allowed.into_iter().map(|(id, _)| id).collect();
        let oversample = (k * 4).max(k + 50);
        let hits = self.index.search(query, oversample)?;
        let mut results = Vec::with_capacity(k);
        for (id, score) in hits {
            if !allowed_set.contains(&id) {
                continue;
            }
            if let Ok((_, rec)) = self.metadata.get(&id) {
                results.push(SearchResult {
                    id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                    payload: rec.payload,
                    text: rec.text,
                });
            }
            if results.len() >= k {
                break;
            }
        }
        Ok(results)
    }

    /// Two-stage hybrid retrieval pipeline.
    ///
    /// At least one of `query_vector` / `query_text` must be provided.
    ///
    /// Stage 1 — Dense: fetch `k * 5` HNSW candidates (or empty if no vector).
    /// Stage 2 — Sparse: re-score dense candidates with BM25 (or full-corpus
    ///           search when no dense candidates exist).
    /// Stage 3 — Fusion: combine via `strategy` and `alpha`.
    /// Stage 4 — Hydrate: populate `payload` and `text` from metadata.
    pub fn search_hybrid(
        &self,
        query_vector: Option<&Vector>,
        query_text: Option<&str>,
        k: usize,
        alpha: f32,
        strategy: FusionStrategy,
    ) -> Result<Vec<SearchResult>> {
        // Validate inputs.
        if query_vector.is_none() && query_text.is_none() {
            return Err(VecDbError::InvalidQuery(
                "hybrid search requires at least one of query_vector or query_text".into(),
            ));
        }
        if let Some(qv) = query_vector {
            if qv.len() != self.config.dimension {
                return Err(VecDbError::DimensionMismatch {
                    expected: self.config.dimension,
                    got: qv.len(),
                });
            }
        }

        let oversample_k = k * 5;

        // Stage 1 — Dense retrieval.
        let dense_results: Vec<(VectorId, f32)> = if let Some(qv) = query_vector {
            self.index.search(qv, oversample_k)?
        } else {
            vec![]
        };

        // Stage 2 — Sparse re-scoring.
        let sparse_results: Vec<(VectorId, f32)> = if let Some(qt) = query_text {
            if !dense_results.is_empty() {
                // Re-score only the dense candidates.
                let candidate_ids: Vec<VectorId> =
                    dense_results.iter().map(|(id, _)| id.clone()).collect();
                self.sparse.score(qt, &candidate_ids)?
            } else {
                // No dense candidates — full sparse search.
                self.sparse.score_all(qt, oversample_k)?
            }
        } else {
            vec![]
        };

        // Stage 3 — Fusion.
        let engine = HybridEngine::new(alpha, strategy, 5);
        let mut fused = engine.fuse(dense_results, sparse_results, k)?;

        // Stage 4 — Hydrate payloads + text from metadata.
        for result in &mut fused {
            match self.metadata.get(&result.id) {
                Ok((_, rec)) => {
                    result.payload = rec.payload;
                    result.text = rec.text;
                }
                Err(_) => {
                    tracing::warn!(
                        "hybrid search: metadata not found for id '{}', keeping null payload",
                        result.id
                    );
                }
            }
        }

        Ok(fused)
    }

    /// Rebuild the dense index from the current metadata + mmap store.
    /// Also rebuilds the sparse index from stored text payloads.
    pub fn rebuild_index(&mut self) -> Result<()> {
        let active = self.metadata.list_active()?;
        let mut pairs: Vec<(VectorId, crate::types::Vector)> = Vec::with_capacity(active.len());
        for (id, mmap_idx) in &active {
            let vec = self.vectors.get(*mmap_idx)?;
            pairs.push((id.clone(), vec));
        }
        let count = pairs.len();
        self.index.build(pairs)?;

        // Rebuild sparse from stored text.
        self.sparse = SparseIndex::create(
            &self
                .data_dir
                .join(format!("{}.sparse.json", self.config.name)),
        );
        for (id, _) in &active {
            if let Ok((_, rec)) = self.metadata.get(id) {
                if let Some(ref text) = rec.text {
                    if !text.is_empty() {
                        let _ = self.sparse.index_document(id, text);
                    }
                }
            }
        }

        tracing::info!("Index rebuilt from {} vectors", count);
        Ok(())
    }

    /// Persist both the dense index and the sparse index to disk.
    pub fn save_indexes(&self) -> Result<()> {
        self.index
            .save_for_collection(&self.data_dir, &self.config.name)?;
        self.sparse.save()?;
        Ok(())
    }

    pub fn recover_from_wal(&mut self) -> Result<usize> {
        let entries = self.wal.replay()?;
        let mut applied = 0usize;

        for entry in &entries {
            match entry {
                WalEntry::Insert {
                    id,
                    mmap_index,
                    record,
                } => match self.metadata.get(id) {
                    Ok((existing_idx, _)) if existing_idx == *mmap_index => {
                        // Already applied — skip.
                    }
                    _ => {
                        self.metadata.upsert(id, *mmap_index, record)?;
                        applied += 1;
                    }
                },
                WalEntry::Delete { id } => match self.metadata.get(id) {
                    Ok(_) => {
                        self.metadata.delete(id)?;
                        applied += 1;
                    }
                    Err(VecDbError::NotFound { .. }) => {}
                    Err(e) => return Err(e),
                },
                WalEntry::Checkpoint { .. } => {}
            }
        }

        tracing::info!("WAL recovery applied {} entries", applied);
        Ok(applied)
    }

    pub fn checkpoint(&mut self) -> Result<()> {
        self.wal.checkpoint()?;
        self.wal.truncate_after_checkpoint()?;
        tracing::info!(
            "Storage checkpoint complete for collection '{}'",
            self.config.name
        );
        Ok(())
    }

    pub fn stats(&self) -> Result<IndexStats> {
        let vector_count = self.metadata.count_active()?;
        let vectors_path = self.data_dir.join(format!("{}.vectors", self.config.name));
        let disk_bytes = std::fs::metadata(&vectors_path)
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(IndexStats {
            collection: self.config.name.clone(),
            vector_count,
            index_type: self.config.index_type.clone(),
            dimension: self.config.dimension,
            disk_bytes,
            memory_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    use crate::types::{CollectionConfig, Vector, VectorRecord};

    fn dummy_record(id: &str, vector: Vector) -> VectorRecord {
        let now = Utc::now();
        VectorRecord {
            id: id.to_string(),
            vector,
            payload: json!({ "id": id }),
            text: Some(format!("text for {id}")),
            created_at: now,
            updated_at: now,
        }
    }

    #[test]
    fn test_mmap_create_and_open() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vectors");

        {
            let mut store = MmapVectorStore::create(&path, 4, 10).unwrap();
            store.append(&vec![1.0, 2.0, 3.0, 4.0]).unwrap();
            store.append(&vec![5.0, 6.0, 7.0, 8.0]).unwrap();
            store.append(&vec![9.0, 10.0, 11.0, 12.0]).unwrap();
            assert_eq!(store.len(), 3);
        }

        let store = MmapVectorStore::open(&path).unwrap();
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(0).unwrap(), vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(store.get(1).unwrap(), vec![5.0f32, 6.0, 7.0, 8.0]);
        assert_eq!(store.get(2).unwrap(), vec![9.0f32, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_mmap_dimension_mismatch() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vectors");
        let mut store = MmapVectorStore::create(&path, 4, 10).unwrap();
        let result = store.append(&vec![1.0, 2.0, 3.0]);
        assert!(matches!(
            result,
            Err(VecDbError::DimensionMismatch {
                expected: 4,
                got: 3
            })
        ));
    }

    #[test]
    fn test_mmap_grow() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.vectors");
        let mut store = MmapVectorStore::create(&path, 2, 2).unwrap();
        store.append(&vec![1.0f32, 2.0]).unwrap();
        store.append(&vec![3.0f32, 4.0]).unwrap();
        store.append(&vec![5.0f32, 6.0]).unwrap(); // forces grow
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(0).unwrap(), vec![1.0f32, 2.0]);
        assert_eq!(store.get(1).unwrap(), vec![3.0f32, 4.0]);
        assert_eq!(store.get(2).unwrap(), vec![5.0f32, 6.0]);
    }

    #[test]
    fn test_wal_append_and_replay() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let mut wal = WriteAheadLog::open(&path).unwrap();

        wal.append(&WalEntry::Insert {
            id: "doc1".into(),
            mmap_index: 0,
            record: dummy_record("doc1", vec![1.0f32, 2.0]),
        })
        .unwrap();
        wal.append(&WalEntry::Insert {
            id: "doc2".into(),
            mmap_index: 1,
            record: dummy_record("doc2", vec![3.0f32, 4.0]),
        })
        .unwrap();
        wal.append(&WalEntry::Insert {
            id: "doc3".into(),
            mmap_index: 2,
            record: dummy_record("doc3", vec![5.0f32, 6.0]),
        })
        .unwrap();
        wal.append(&WalEntry::Delete { id: "doc1".into() }).unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 4);
        assert!(matches!(&entries[0], WalEntry::Insert { id, .. } if id == "doc1"));
        assert!(matches!(&entries[3], WalEntry::Delete { id } if id == "doc1"));
    }

    #[test]
    fn test_wal_checkpoint_and_truncate() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.wal");
        let mut wal = WriteAheadLog::open(&path).unwrap();

        let r = dummy_record("x", vec![1.0f32]);
        for i in 0..3usize {
            wal.append(&WalEntry::Insert {
                id: format!("doc{i}"),
                mmap_index: i,
                record: r.clone(),
            })
            .unwrap();
        }
        wal.checkpoint().unwrap();
        for i in 3..5usize {
            wal.append(&WalEntry::Insert {
                id: format!("doc{i}"),
                mmap_index: i,
                record: r.clone(),
            })
            .unwrap();
        }
        wal.truncate_after_checkpoint().unwrap();

        let entries = wal.replay().unwrap();
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_metadata_upsert_and_get() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let store = MetadataStore::open(&path).unwrap();

        let record = VectorRecord {
            id: "doc1".into(),
            vector: vec![],
            payload: json!({ "title": "hello" }),
            text: Some("hello world".into()),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        store.upsert("doc1", 0, &record).unwrap();
        let (mmap_index, retrieved) = store.get("doc1").unwrap();

        assert_eq!(mmap_index, 0);
        assert_eq!(retrieved.id, "doc1");
        assert_eq!(retrieved.text.as_deref(), Some("hello world"));
        assert_eq!(retrieved.payload["title"], "hello");
    }

    #[test]
    fn test_metadata_soft_delete() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let store = MetadataStore::open(&path).unwrap();

        store
            .upsert("doc2", 0, &dummy_record("doc2", vec![]))
            .unwrap();
        store.delete("doc2").unwrap();

        assert!(matches!(
            store.get("doc2"),
            Err(VecDbError::NotFound { .. })
        ));
        assert_eq!(store.count_active().unwrap(), 0);
    }

    #[test]
    fn test_storage_full_roundtrip() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("testcol", 3);

        {
            let mut storage = Storage::create(dir.path(), &config).unwrap();
            for i in 0..5u32 {
                let r = dummy_record(
                    &format!("doc{i}"),
                    vec![i as f32, i as f32 + 1.0, i as f32 + 2.0],
                );
                storage.upsert(r).unwrap();
            }
            assert_eq!(storage.stats().unwrap().vector_count, 5);
            storage.delete(&"doc0".to_string()).unwrap();
            assert_eq!(storage.stats().unwrap().vector_count, 4);
        }

        let mut storage = Storage::open(dir.path(), "testcol").unwrap();
        assert_eq!(storage.stats().unwrap().vector_count, 4);
        storage.checkpoint().unwrap();
    }

    #[test]
    fn test_wal_recovery() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("waltest", 2);

        {
            let mut storage = Storage::create(dir.path(), &config).unwrap();
            storage
                .upsert(dummy_record("r1", vec![1.0f32, 2.0]))
                .unwrap();
            storage
                .upsert(dummy_record("r2", vec![3.0f32, 4.0]))
                .unwrap();
            storage
                .upsert(dummy_record("r3", vec![5.0f32, 6.0]))
                .unwrap();
            // Drop without checkpoint — simulates crash.
        }

        let storage = Storage::open(dir.path(), "waltest").unwrap();
        assert!(storage.metadata.get("r1").is_ok());
        assert!(storage.metadata.get("r2").is_ok());
        assert!(storage.metadata.get("r3").is_ok());
    }

    #[test]
    fn test_sparse_search_after_upsert() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("sparsecol", 3);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        storage
            .upsert(VectorRecord {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                payload: json!({}),
                text: Some("rust vector database storage engine".to_string()),
                created_at: now,
                updated_at: now,
            })
            .unwrap();
        storage
            .upsert(VectorRecord {
                id: "v2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                payload: json!({}),
                text: Some("python machine learning numpy".to_string()),
                created_at: now,
                updated_at: now,
            })
            .unwrap();

        let results = storage.search_sparse("rust database", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "v1");
        assert!(results[0].sparse_score.is_some());
        assert!(results[0].dense_score.is_none());
    }

    #[test]
    fn test_sparse_delete_removes_from_index() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("delsparse", 2);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        storage
            .upsert(VectorRecord {
                id: "x1".to_string(),
                vector: vec![1.0, 0.0],
                payload: json!({}),
                text: Some("neural network deep learning".to_string()),
                created_at: now,
                updated_at: now,
            })
            .unwrap();

        storage.delete(&"x1".to_string()).unwrap();

        let results = storage.search_sparse("neural", 5).unwrap();
        assert!(results.is_empty());
    }

    // ── Hybrid tests ─────────────────────────────────────────────

    #[test]
    fn test_storage_search_hybrid_both_signals() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("hybcol", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let texts = [
            "machine learning algorithms",
            "deep neural networks",
            "random forest trees",
            "gradient boosting ensemble",
            "support vector machines",
        ];
        let now = Utc::now();
        for (i, text) in texts.iter().enumerate() {
            storage
                .upsert(VectorRecord {
                    id: format!("hyb{i}"),
                    vector: vec![i as f32, i as f32 + 1.0, i as f32 + 0.5, 1.0],
                    payload: json!({ "i": i }),
                    text: Some(text.to_string()),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let query_vec = vec![0.0_f32, 1.0, 0.5, 1.0];
        let results = storage
            .search_hybrid(
                Some(&query_vec),
                Some("machine learning"),
                3,
                0.7,
                crate::hybrid::FusionStrategy::WeightedSum,
            )
            .unwrap();

        assert!(
            results.len() <= 3,
            "expected <= 3 results, got {}",
            results.len()
        );
        for r in &results {
            assert!(
                storage.metadata.get(&r.id).is_ok(),
                "id {} not in metadata",
                r.id
            );
            assert!(r.score >= 0.0, "score {} < 0.0", r.score);
        }
        // Sorted descending.
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_storage_search_hybrid_dense_only() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("denonly", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        for i in 0..5usize {
            storage
                .upsert(VectorRecord {
                    id: format!("d{i}"),
                    vector: vec![i as f32, 1.0, 0.0, 0.0],
                    payload: json!({}),
                    text: None, // no text
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let query_vec = vec![2.0_f32, 1.0, 0.0, 0.0];
        let results = storage
            .search_hybrid(
                Some(&query_vec),
                None,
                3,
                0.7,
                crate::hybrid::FusionStrategy::WeightedSum,
            )
            .unwrap();

        assert!(!results.is_empty(), "dense-only hybrid must return results");
    }

    #[test]
    fn test_storage_search_hybrid_sparse_only() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("sparonly", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        let texts = [
            "neural network deep learning",
            "gradient descent optimization",
            "convolutional neural network image",
            "recurrent network sequence model",
            "transformer attention mechanism",
        ];
        for (i, text) in texts.iter().enumerate() {
            storage
                .upsert(VectorRecord {
                    id: format!("sp{i}"),
                    vector: vec![i as f32, 0.0, 0.0, 1.0],
                    payload: json!({}),
                    text: Some(text.to_string()),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let results = storage
            .search_hybrid(
                None,
                Some("neural network"),
                3,
                0.7,
                crate::hybrid::FusionStrategy::WeightedSum,
            )
            .unwrap();

        assert!(
            !results.is_empty(),
            "sparse-only hybrid must return results"
        );
    }

    #[test]
    fn test_storage_search_hybrid_no_inputs_errors() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("errtest", 4);
        let storage = Storage::create(dir.path(), &config).unwrap();

        let result = storage.search_hybrid(
            None,
            None,
            3,
            0.7,
            crate::hybrid::FusionStrategy::WeightedSum,
        );
        assert!(
            matches!(result, Err(VecDbError::InvalidQuery(_))),
            "expected InvalidQuery error"
        );
    }

    #[test]
    fn test_storage_search_dense_method() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("denmeth", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        for i in 0..5usize {
            storage
                .upsert(VectorRecord {
                    id: format!("v{i}"),
                    vector: vec![i as f32, i as f32 + 1.0, 0.0, 1.0],
                    payload: json!({ "i": i }),
                    text: Some(format!("text {i}")),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let query_vec = vec![2.0_f32, 3.0, 0.0, 1.0];
        let results = storage.search_dense(&query_vec, 3).unwrap();

        assert!(!results.is_empty(), "search_dense must return results");
        for r in &results {
            assert!(r.dense_score.is_some(), "dense_score must be Some");
            assert!(r.sparse_score.is_none(), "sparse_score must be None");
        }
    }

    #[test]
    fn test_save_indexes_and_reload() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("saveidx", 3);

        {
            let mut storage = Storage::create(dir.path(), &config).unwrap();
            let now = Utc::now();
            storage
                .upsert(VectorRecord {
                    id: "s1".to_string(),
                    vector: vec![1.0, 2.0, 3.0],
                    payload: json!({}),
                    text: Some("inverted index bm25 scoring".to_string()),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
            storage.save_indexes().unwrap();
        }

        let storage = Storage::open(dir.path(), "saveidx").unwrap();
        let results = storage.search_sparse("bm25", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "s1");
    }
}
