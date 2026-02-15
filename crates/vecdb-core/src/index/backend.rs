use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::index::distance;
use crate::types::{CollectionConfig, DistanceMetric, IndexType, Vector, VectorId};

// ──────────────────────────────────────────────
// IndexBackend trait
// ──────────────────────────────────────────────

pub trait IndexBackend: Send + Sync {
    fn build(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()>;
    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()>;
    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>>;
    fn delete(&mut self, id: &VectorId) -> Result<()>;
    fn save(&self, path: &Path) -> Result<()>;
    fn load_from(path: &Path, config: &CollectionConfig) -> Result<Box<dyn IndexBackend>>
    where
        Self: Sized;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn index_type(&self) -> IndexType;
    fn rebuild(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.build(vectors)
    }
}

// ──────────────────────────────────────────────
// Point wrapper for instant-distance
// ──────────────────────────────────────────────

/// Wrapper around a raw vector that carries the distance metric.
/// Required by the `instant_distance::Point` trait.
#[derive(Clone)]
struct Point {
    data: Vec<f32>,
    metric: DistanceMetric,
}

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        distance::compute_distance(&self.data, &other.data, &self.metric)
    }
}

// ──────────────────────────────────────────────
// HnswConfig
// ──────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfig {
    /// Graph degree (M in the paper). Note: instant-distance hard-codes M=32;
    /// this field is kept for API compatibility and future backends.
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub metric: DistanceMetric,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            metric: DistanceMetric::Cosine,
        }
    }
}

// ──────────────────────────────────────────────
// HnswIndex
// ──────────────────────────────────────────────

pub struct HnswIndex {
    pub dimension: usize,
    pub metric: DistanceMetric,
    pub config: HnswConfig,
    /// VectorId → position in `vectors` / `pos_to_id`
    id_to_pos: HashMap<VectorId, usize>,
    /// Position → VectorId
    pos_to_id: Vec<VectorId>,
    /// Raw vectors in insertion order (parallel to pos_to_id)
    vectors: Vec<Vector>,
    /// Soft-deleted ids
    deleted: HashSet<VectorId>,
    /// Built HNSW structure; None until build() is called
    hnsw: Option<instant_distance::HnswMap<Point, VectorId>>,
    /// True when inserts/deletes have occurred since the last build
    dirty: bool,
}

impl HnswIndex {
    pub fn new(dimension: usize, config: HnswConfig) -> Self {
        let metric = config.metric.clone();
        Self {
            dimension,
            metric,
            config,
            id_to_pos: HashMap::new(),
            pos_to_id: Vec::new(),
            vectors: Vec::new(),
            deleted: HashSet::new(),
            hnsw: None,
            dirty: false,
        }
    }

    pub fn from_collection_config(config: &CollectionConfig) -> Self {
        let hnsw_config = HnswConfig {
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            metric: config.metric.clone(),
        };
        Self::new(config.dimension, hnsw_config)
    }

    /// Load a saved index from disk and return a concrete `HnswIndex`
    /// (avoids the need to downcast the trait object returned by `load_from`).
    pub fn load_file(path: &Path, config: &CollectionConfig) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| VecDbError::StorageError(format!("hnsw load failed: {e}")))?;
        let saved: SerializedHnswIndex = serde_json::from_str(&json)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;

        let hnsw_config = HnswConfig {
            m: config.hnsw_m,
            ef_construction: config.hnsw_ef_construction,
            ef_search: config.hnsw_ef_search,
            metric: config.metric.clone(),
        };
        let metric = hnsw_config.metric.clone();

        let mut index = Self {
            dimension: saved.dimension,
            metric,
            config: hnsw_config,
            id_to_pos: saved.id_to_pos,
            pos_to_id: saved.pos_to_id,
            vectors: saved.vectors,
            deleted: saved.deleted.into_iter().collect(),
            hnsw: None,
            dirty: false,
        };

        // Rebuild HNSW from stored vectors (live only)
        let live: Vec<(VectorId, Vector)> = index
            .pos_to_id
            .iter()
            .zip(index.vectors.iter())
            .filter(|(id, _)| !index.deleted.contains(*id))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();
        if !live.is_empty() {
            index.build(live)?;
        }

        tracing::info!("HNSW index loaded from {:?}", path);
        Ok(index)
    }

    // ── private helpers ──────────────────────────────

    fn maybe_normalize(&self, v: Vec<f32>) -> Vec<f32> {
        if matches!(self.metric, DistanceMetric::Cosine) {
            distance::normalize(&v)
        } else {
            v
        }
    }

    fn to_score(&self, dist: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => 1.0 - dist,
            DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
            DistanceMetric::DotProduct => 1.0 - dist,
        }
    }

    fn brute_force_search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        let q = self.maybe_normalize(query.clone());
        let mut scored: Vec<(VectorId, f32)> = self
            .pos_to_id
            .iter()
            .enumerate()
            .filter(|(_, id)| !self.deleted.contains(*id))
            .map(|(pos, id)| {
                let dist = distance::compute_distance(&q, &self.vectors[pos], &self.metric);
                (id.clone(), self.to_score(dist))
            })
            .collect();

        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    fn collect_live_vectors(&self) -> Vec<(VectorId, Vector)> {
        self.pos_to_id
            .iter()
            .zip(self.vectors.iter())
            .filter(|(id, _)| !self.deleted.contains(*id))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect()
    }
}

// ──────────────────────────────────────────────
// IndexBackend implementation
// ──────────────────────────────────────────────

impl IndexBackend for HnswIndex {
    fn build(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.id_to_pos.clear();
        self.pos_to_id.clear();
        self.vectors.clear();
        self.deleted.clear();

        if vectors.is_empty() {
            self.hnsw = None;
            self.dirty = false;
            return Ok(());
        }

        for (id, raw_vec) in vectors {
            if raw_vec.len() != self.dimension {
                return Err(VecDbError::DimensionMismatch {
                    expected: self.dimension,
                    got: raw_vec.len(),
                });
            }
            let v = self.maybe_normalize(raw_vec);
            let pos = self.vectors.len();
            self.id_to_pos.insert(id.clone(), pos);
            self.pos_to_id.push(id.clone());
            self.vectors.push(v);
        }

        let points: Vec<Point> = self
            .vectors
            .iter()
            .map(|v| Point {
                data: v.clone(),
                metric: self.metric.clone(),
            })
            .collect();
        let values: Vec<VectorId> = self.pos_to_id.clone();

        let hnsw = instant_distance::Builder::default()
            .ef_construction(self.config.ef_construction)
            .ef_search(self.config.ef_search)
            .build(points, values);

        self.hnsw = Some(hnsw);
        self.dirty = false;

        tracing::info!(
            "HNSW index built: {} vectors, metric={:?}",
            self.vectors.len(),
            self.metric
        );
        Ok(())
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        let v = self.maybe_normalize(vector);

        if let Some(&pos) = self.id_to_pos.get(&id) {
            // Existing entry (live or deleted)
            self.deleted.remove(&id);
            self.vectors[pos] = v;
        } else {
            // New entry
            let pos = self.vectors.len();
            self.vectors.push(v);
            self.id_to_pos.insert(id.clone(), pos);
            self.pos_to_id.push(id.clone());
        }

        self.dirty = true;
        tracing::debug!("HNSW insert: id={}, dirty={}", id, self.dirty);

        // Auto-rebuild for small collections or at batch boundaries
        let n = self.vectors.len();
        if n <= 100 || n.is_multiple_of(1000) {
            let live = self.collect_live_vectors();
            self.build(live)?;
        }

        Ok(())
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        if query.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        // Fall back to brute force when index unavailable or dirty
        if self.hnsw.is_none() || self.dirty {
            return self.brute_force_search(query, k);
        }

        let q = self.maybe_normalize(query.clone());
        let query_point = Point {
            data: q,
            metric: self.metric.clone(),
        };

        let hnsw = self.hnsw.as_ref().unwrap();
        let mut search = instant_distance::Search::default();

        let mut results: Vec<(VectorId, f32)> = hnsw
            .search(&query_point, &mut search)
            .filter(|item| !self.deleted.contains(item.value))
            .map(|item| (item.value.clone(), self.to_score(item.distance)))
            .collect();

        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    fn delete(&mut self, id: &VectorId) -> Result<()> {
        if !self.id_to_pos.contains_key(id) {
            return Err(VecDbError::NotFound { id: id.clone() });
        }
        self.deleted.insert(id.clone());
        self.dirty = true;

        tracing::debug!(
            "HNSW delete: id={}, deleted_count={}",
            id,
            self.deleted.len()
        );

        // Compact when dead weight exceeds 25 %
        if self.deleted.len() > self.vectors.len() / 4 {
            let live = self.collect_live_vectors();
            self.build(live)?;
        }

        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let saved = SerializedHnswIndex {
            dimension: self.dimension,
            config: self.config.clone(),
            id_to_pos: self.id_to_pos.clone(),
            pos_to_id: self.pos_to_id.clone(),
            vectors: self.vectors.clone(),
            deleted: self.deleted.iter().cloned().collect(),
            version: 1,
        };
        let json = serde_json::to_string(&saved)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| VecDbError::StorageError(format!("hnsw save failed: {e}")))?;
        tracing::info!("HNSW index saved to {:?}", path);
        Ok(())
    }

    fn load_from(path: &Path, config: &CollectionConfig) -> Result<Box<dyn IndexBackend>>
    where
        Self: Sized,
    {
        Ok(Box::new(Self::load_file(path, config)?))
    }

    fn len(&self) -> usize {
        self.vectors.len().saturating_sub(self.deleted.len())
    }

    fn index_type(&self) -> IndexType {
        IndexType::HNSW
    }
}

// ──────────────────────────────────────────────
// Serialization helper
// ──────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct SerializedHnswIndex {
    dimension: usize,
    config: HnswConfig,
    id_to_pos: HashMap<VectorId, usize>,
    pos_to_id: Vec<VectorId>,
    vectors: Vec<Vector>,
    deleted: Vec<VectorId>,
    version: u32,
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CollectionConfig, VectorRecord};
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_index(dim: usize, metric: DistanceMetric) -> HnswIndex {
        HnswIndex::new(
            dim,
            HnswConfig {
                m: 16,
                ef_construction: 100,
                ef_search: 50,
                metric,
            },
        )
    }

    fn dummy_record(id: &str, vector: Vector) -> VectorRecord {
        VectorRecord {
            id: id.to_string(),
            vector,
            payload: json!({ "id": id }),
            text: Some(format!("text for {id}")),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    // ── Test 1 ───────────────────────────────────────
    #[test]
    fn test_distance_cosine_basic() {
        let mut idx = make_index(3, DistanceMetric::Cosine);
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]).unwrap();
        idx.insert("c".into(), vec![0.0, 0.0, 1.0]).unwrap();
        idx.insert("d".into(), vec![1.0, 0.0, 0.0]).unwrap();

        let results = idx.search(&vec![1.0, 0.0, 0.0], 2).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"), "expected 'a' in top-2");
        assert!(ids.contains(&"d"), "expected 'd' in top-2");
        assert!(results.iter().all(|(_, s)| *s > 0.99));
    }

    // ── Test 2 ───────────────────────────────────────
    #[test]
    fn test_euclidean_search() {
        let mut idx = make_index(2, DistanceMetric::Euclidean);
        idx.insert("near".into(), vec![1.0, 1.0]).unwrap();
        idx.insert("far".into(), vec![100.0, 100.0]).unwrap();
        idx.insert("medium".into(), vec![5.0, 5.0]).unwrap();

        let results = idx.search(&vec![1.1, 1.1], 1).unwrap();
        assert_eq!(results[0].0, "near");
    }

    // ── Test 3 ───────────────────────────────────────
    #[test]
    fn test_brute_force_fallback() {
        // Insert without triggering auto-build by using dim > 100
        // Actually auto-rebuild triggers for n <= 100; to avoid it we'd need n > 100.
        // Instead just verify brute-force path works (it's also used on dirty index).
        let mut idx = make_index(2, DistanceMetric::Euclidean);
        // Force a state where hnsw is None by building only temporarily
        // Insert 5, then call build with empty to clear hnsw, then search
        for i in 0..5usize {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0]).unwrap();
        }
        // Clear hnsw to force brute force
        idx.hnsw = None;
        idx.dirty = false; // dirty=false + hnsw=None → brute force path

        let results = idx.search(&vec![2.0, 0.0], 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "v2");
    }

    // ── Test 4 ───────────────────────────────────────
    #[test]
    fn test_soft_delete() {
        let mut idx = make_index(2, DistanceMetric::Euclidean);
        idx.insert("a".into(), vec![1.0, 0.0]).unwrap();
        idx.insert("b".into(), vec![2.0, 0.0]).unwrap();
        idx.insert("c".into(), vec![3.0, 0.0]).unwrap();
        idx.insert("d".into(), vec![4.0, 0.0]).unwrap();
        idx.insert("e".into(), vec![5.0, 0.0]).unwrap();

        idx.delete(&"a".to_string()).unwrap();
        assert_eq!(idx.len(), 4);

        let results = idx.search(&vec![1.0, 0.0], 5).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            !ids.contains(&"a"),
            "deleted 'a' must not appear in results"
        );
    }

    // ── Test 5 ───────────────────────────────────────
    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.hnsw.json");
        let config = CollectionConfig::new("test", 3);

        let mut idx = HnswIndex::from_collection_config(&config);
        for i in 0..10usize {
            idx.insert(
                format!("v{i}"),
                vec![i as f32, (i + 1) as f32, (i + 2) as f32],
            )
            .unwrap();
        }

        let original = idx.search(&vec![5.0, 6.0, 7.0], 3).unwrap();
        idx.save(&path).unwrap();

        let loaded = HnswIndex::load_file(&path, &config).unwrap();
        let after = loaded.search(&vec![5.0, 6.0, 7.0], 3).unwrap();

        assert_eq!(original.len(), after.len());
        // Top result should be the same
        assert_eq!(original[0].0, after[0].0);
    }

    // ── Test 6 ───────────────────────────────────────
    #[test]
    fn test_dimension_mismatch_on_insert() {
        let mut idx = make_index(4, DistanceMetric::Cosine);
        let res = idx.insert("bad".into(), vec![1.0, 2.0, 3.0]);
        assert!(
            matches!(
                res,
                Err(VecDbError::DimensionMismatch {
                    expected: 4,
                    got: 3
                })
            ),
            "expected DimensionMismatch"
        );
    }

    // ── Test 7 ───────────────────────────────────────
    #[test]
    fn test_storage_search_integration() {
        use crate::storage::Storage;

        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("srch", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        for i in 0..10usize {
            let rec = dummy_record(
                &format!("doc{i}"),
                vec![
                    (i as f32 + 1.0),
                    (i as f32 + 2.0),
                    (i as f32 + 3.0),
                    (i as f32 + 4.0),
                ],
            );
            storage.upsert(rec).unwrap();
        }

        let query = vec![5.0f32, 6.0, 7.0, 8.0];
        let results = storage.search(&query, 10).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 10);

        for r in &results {
            assert!(
                storage.metadata.get(&r.id).is_ok(),
                "id {} not in metadata",
                r.id
            );
            assert!(
                r.score >= 0.0 && r.score <= 1.0,
                "score {} out of [0,1]",
                r.score
            );
        }
    }

    // ── Test 8 ───────────────────────────────────────
    #[test]
    fn test_index_rebuild_after_reopen() {
        use crate::storage::Storage;

        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("reopen", 2);

        {
            let mut storage = Storage::create(dir.path(), &config).unwrap();
            for i in 0..5usize {
                storage
                    .upsert(dummy_record(&format!("r{i}"), vec![i as f32, 1.0]))
                    .unwrap();
            }
            storage.save_indexes().unwrap();
        }

        let storage = Storage::open(dir.path(), "reopen").unwrap();
        let results = storage.search(&vec![2.0, 1.0], 3).unwrap();
        assert!(!results.is_empty());
    }

    // ── Test 9 ───────────────────────────────────────
    #[test]
    fn test_auto_rebuild_on_large_insert() {
        let mut idx = make_index(2, DistanceMetric::Euclidean);
        for i in 0..1000usize {
            idx.insert(format!("v{i}"), vec![i as f32, 0.0]).unwrap();
        }
        assert_eq!(idx.len(), 1000);

        let results = idx.search(&vec![500.0, 0.0], 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "v500");
    }
}
