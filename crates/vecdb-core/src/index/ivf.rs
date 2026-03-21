use std::collections::HashSet;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::index::backend::IndexBackend;
use crate::index::distance::compute_distance;
use crate::types::{CollectionConfig, DistanceMetric, IndexType, Vector, VectorId};

const DEFAULT_N_LISTS: usize = 256;
const DEFAULT_N_PROBE: usize = 16;
const DEFAULT_MAX_ITER: usize = 25;

#[derive(Serialize, Deserialize)]
pub struct IvfIndex {
    n_lists: usize,
    n_probe: usize,
    max_iter: usize,
    dimension: usize,
    metric: DistanceMetric,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<(VectorId, Vec<f32>)>>,
    deleted: HashSet<VectorId>,
    is_trained: bool,
}

impl IvfIndex {
    pub fn new(config: &CollectionConfig) -> Self {
        Self {
            n_lists: DEFAULT_N_LISTS,
            n_probe: DEFAULT_N_PROBE,
            max_iter: DEFAULT_MAX_ITER,
            dimension: config.dimension,
            metric: config.metric.clone(),
            centroids: Vec::new(),
            lists: vec![Vec::new()],
            deleted: HashSet::new(),
            is_trained: false,
        }
    }

    pub fn load_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| VecDbError::StorageError(format!("ivf load failed: {e}")))?;
        serde_json::from_str(&json).map_err(|e| VecDbError::SerializationError(e.to_string()))
    }

    fn train(&mut self, vectors: &[(VectorId, Vec<f32>)]) {
        let n = vectors.len();
        if n == 0 {
            self.centroids.clear();
            self.lists = vec![Vec::new()];
            self.is_trained = false;
            return;
        }

        let mut n_lists = self.n_lists;
        if n < n_lists {
            n_lists = n.max(1);
        }

        // Initialize centroids by sampling evenly (deterministic, no RNG)
        let mut centroids: Vec<Vec<f32>> = (0..n_lists)
            .map(|i| vectors[i * (n / n_lists)].1.clone())
            .collect();

        for _iter in 0..self.max_iter {
            // Assign vectors to nearest centroid
            let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); n_lists];
            for (vi, (_, v)) in vectors.iter().enumerate() {
                let ci = Self::nearest_centroid_idx_of(&centroids, v, &self.metric);
                assignments[ci].push(vi);
            }

            // Recompute centroids as mean of assigned vectors
            let mut new_centroids = centroids.clone();
            for (ci, assigned) in assignments.iter().enumerate() {
                if assigned.is_empty() {
                    // Reinitialize deterministically
                    let fallback_idx = (ci * 7) % n;
                    new_centroids[ci] = vectors[fallback_idx].1.clone();
                } else {
                    let dim = self.dimension;
                    let mut mean = vec![0.0f32; dim];
                    for &vi in assigned {
                        for (j, x) in vectors[vi].1.iter().enumerate() {
                            mean[j] += x;
                        }
                    }
                    let count = assigned.len() as f32;
                    for x in &mut mean {
                        *x /= count;
                    }
                    new_centroids[ci] = mean;
                }
            }

            // Break early if converged
            let converged = centroids
                .iter()
                .zip(new_centroids.iter())
                .all(|(old, new)| {
                    old.iter()
                        .zip(new.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-6)
                });
            centroids = new_centroids;
            if converged {
                break;
            }
        }

        self.centroids = centroids;
        self.n_lists = n_lists;

        // Assign every vector to its nearest centroid
        self.lists = vec![Vec::new(); n_lists];
        for (id, v) in vectors {
            let ci = Self::nearest_centroid_idx_of(&self.centroids, v, &self.metric);
            self.lists[ci].push((id.clone(), v.clone()));
        }

        self.is_trained = true;
    }

    fn nearest_centroid_idx_of(
        centroids: &[Vec<f32>],
        query: &[f32],
        metric: &DistanceMetric,
    ) -> usize {
        centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, metric)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    fn nearest_centroid_indices(&self, query: &[f32], n_probe: usize) -> Vec<usize> {
        // Performance: centroid scoring parallelized via rayon.
        // Before: sequential iterator over n_lists centroids
        // After:  par_iter() splits work across CPU cores; ~n_threads× faster for large n_lists
        let mut scored: Vec<(usize, f32)> = self
            .centroids
            .par_iter()
            .enumerate()
            .map(|(i, c)| (i, compute_distance(query, c, &self.metric)))
            .collect();
        scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n_probe);
        scored.iter().map(|(i, _)| *i).collect()
    }

    fn total_count(&self) -> usize {
        self.lists.iter().map(|l| l.len()).sum()
    }

    fn distance_to_score(&self, dist: f32) -> f32 {
        match self.metric {
            DistanceMetric::Cosine => 1.0 - dist,
            DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
            DistanceMetric::DotProduct => 1.0 - dist,
        }
    }

    fn rebuild_from_lists(&mut self) -> Result<()> {
        let all_vectors: Vec<(VectorId, Vec<f32>)> = self
            .lists
            .iter()
            .flat_map(|l| l.iter())
            .filter(|(id, _)| !self.deleted.contains(id))
            .map(|(id, v)| (id.clone(), v.clone()))
            .collect();

        self.deleted.clear();

        if all_vectors.is_empty() {
            self.centroids.clear();
            self.lists = vec![Vec::new()];
            self.is_trained = false;
            return Ok(());
        }

        self.train(&all_vectors);
        Ok(())
    }
}

impl IndexBackend for IvfIndex {
    fn build(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.deleted.clear();
        self.train(&vectors);
        Ok(())
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        // Remove from deleted set if re-inserting
        self.deleted.remove(&id);

        // Remove any existing entry for this id (update case)
        for list in &mut self.lists {
            list.retain(|(lid, _)| lid != &id);
        }

        if self.is_trained && !self.centroids.is_empty() {
            let ci = Self::nearest_centroid_idx_of(&self.centroids, &vector, &self.metric);
            self.lists[ci].push((id, vector));
        } else {
            // Not yet trained — buffer in lists[0]
            if self.lists.is_empty() {
                self.lists.push(Vec::new());
            }
            self.lists[0].push((id, vector));
        }

        let total = self.total_count();
        if !self.is_trained || total.is_multiple_of(1000) {
            self.rebuild_from_lists()?;
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

        let mut candidates: Vec<(VectorId, f32)> = if !self.is_trained || self.centroids.is_empty()
        {
            // Brute-force across all lists.
            // Performance: collect filtered pairs serially (cheap), then score in parallel.
            // Before: sequential map over all vectors
            // After:  par_iter() over pre-filtered pairs; ~n_threads× faster at large scale
            let pairs: Vec<(&VectorId, &Vec<f32>)> = self
                .lists
                .iter()
                .flat_map(|l| l.iter())
                .filter(|(id, _)| !self.deleted.contains(id))
                .map(|(id, v)| (id, v))
                .collect();
            let metric = &self.metric;
            pairs
                .par_iter()
                .map(|(id, v)| {
                    let dist = compute_distance(query, v, metric);
                    let score = match metric {
                        crate::types::DistanceMetric::Cosine => 1.0 - dist,
                        crate::types::DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
                        crate::types::DistanceMetric::DotProduct => 1.0 - dist,
                    };
                    ((*id).clone(), score)
                })
                .collect()
        } else {
            let n_probe = self.n_probe.min(self.centroids.len());
            let centroid_indices = self.nearest_centroid_indices(query, n_probe);
            centroid_indices
                .iter()
                .flat_map(|&ci| self.lists[ci].iter())
                .filter(|(id, _)| !self.deleted.contains(id))
                .map(|(id, v)| {
                    let dist = compute_distance(query, v, &self.metric);
                    (id.clone(), self.distance_to_score(dist))
                })
                .collect()
        };

        candidates
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(k);
        Ok(candidates)
    }

    fn delete(&mut self, id: &VectorId) -> Result<()> {
        self.deleted.insert(id.clone());
        Ok(())
    }

    fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| VecDbError::StorageError(format!("ivf save failed: {e}")))?;
        tracing::info!("IVF index saved to {:?}", path);
        Ok(())
    }

    fn load_from(path: &Path, _config: &CollectionConfig) -> Result<Box<dyn IndexBackend>>
    where
        Self: Sized,
    {
        Ok(Box::new(Self::load_file(path)?))
    }

    fn len(&self) -> usize {
        self.total_count().saturating_sub(self.deleted.len())
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn index_type(&self) -> IndexType {
        IndexType::IVF
    }

    fn rebuild(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.build(vectors)
    }
}
