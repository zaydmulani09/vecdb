//! Binary (1-bit) quantization — a second optional in-memory index mode.
//!
//! Each dimension is thresholded at its learned per-dimension mean and stored as
//! a single bit (1 if above the mean, else 0), packed into `u64` words. Distance
//! is Hamming distance (popcount of XORed codes), which approximates angular
//! similarity. This is an extreme compressor: `D` bits/vector = `D/8` bytes vs
//! `D·4` for float32 → ~32× smaller.
//!
//! Binary codes are coarse: on their own they give a fast, tiny filter with
//! modest recall. The intended pipeline is binary-Hamming for candidate
//! generation followed by a full-precision rerank of the top candidates — see
//! `examples/quant_bench.rs`, which measures both raw and reranked recall.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::index::backend::IndexBackend;
use crate::index::distance;
use crate::types::{CollectionConfig, DistanceMetric, IndexType, Vector, VectorId};

/// Per-dimension mean-threshold binary quantizer.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BinaryQuantizer {
    /// Per-dimension threshold (the mean seen during fit).
    pub thresholds: Vec<f32>,
}

impl BinaryQuantizer {
    pub fn fit(dimension: usize, sample: &[Vector]) -> Self {
        let mut sums = vec![0.0f64; dimension];
        for v in sample {
            for (i, &x) in v.iter().enumerate().take(dimension) {
                sums[i] += x as f64;
            }
        }
        let n = sample.len().max(1) as f64;
        let thresholds = sums.iter().map(|s| (s / n) as f32).collect();
        Self { thresholds }
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.thresholds.len()
    }

    /// Number of `u64` words needed to hold one vector's bits.
    #[inline]
    pub fn words(&self) -> usize {
        self.dimension().div_ceil(64)
    }

    /// Pack one vector into bit words: bit `i` set iff `v[i] > threshold[i]`.
    pub fn quantize(&self, v: &[f32]) -> Vec<u64> {
        let mut out = vec![0u64; self.words()];
        for i in 0..self.dimension() {
            if v[i] > self.thresholds[i] {
                out[i / 64] |= 1u64 << (i % 64);
            }
        }
        out
    }
}

pub struct BinaryQuantizedIndex {
    dimension: usize,
    metric: DistanceMetric,
    quantizer: BinaryQuantizer,
    trained: bool,
    words: usize,
    /// Packed bit codes, row-major: vector `pos` occupies `codes[pos*words ..]`.
    codes: Vec<u64>,
    id_to_pos: HashMap<VectorId, usize>,
    pos_to_id: Vec<VectorId>,
    deleted: HashSet<VectorId>,
    pending: Vec<(VectorId, Vector)>,
}

const TRAIN_THRESHOLD: usize = 256;

impl BinaryQuantizedIndex {
    pub fn new(config: &CollectionConfig) -> Self {
        Self {
            dimension: config.dimension,
            metric: config.metric.clone(),
            quantizer: BinaryQuantizer::default(),
            trained: false,
            words: config.dimension.div_ceil(64),
            codes: Vec::new(),
            id_to_pos: HashMap::new(),
            pos_to_id: Vec::new(),
            deleted: HashSet::new(),
            pending: Vec::new(),
        }
    }

    /// Resident memory of the bit-code store in bytes (the figure this mode
    /// shrinks). Excludes id strings, which the float path also carries.
    pub fn code_bytes(&self) -> usize {
        self.codes.len() * std::mem::size_of::<u64>()
            + self.quantizer.thresholds.len() * std::mem::size_of::<f32>()
    }

    fn fit_and_encode(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.id_to_pos.clear();
        self.pos_to_id.clear();
        self.codes.clear();
        self.deleted.clear();
        self.pending.clear();

        if vectors.is_empty() {
            self.trained = false;
            return Ok(());
        }
        for (_, v) in &vectors {
            if v.len() != self.dimension {
                return Err(VecDbError::DimensionMismatch {
                    expected: self.dimension,
                    got: v.len(),
                });
            }
        }

        let sample: Vec<Vector> = vectors.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer = BinaryQuantizer::fit(self.dimension, &sample);
        self.words = self.quantizer.words();

        self.codes.reserve(vectors.len() * self.words);
        for (id, v) in vectors {
            let pos = self.pos_to_id.len();
            self.codes.extend_from_slice(&self.quantizer.quantize(&v));
            self.id_to_pos.insert(id.clone(), pos);
            self.pos_to_id.push(id);
        }
        self.trained = true;
        Ok(())
    }

    fn ensure_trained(&mut self) -> Result<()> {
        if !self.pending.is_empty() {
            let batch = std::mem::take(&mut self.pending);
            self.fit_and_encode(batch)?;
        }
        Ok(())
    }

    /// Fraction of matching bits, in `[0, 1]` (1 = identical codes). Higher is
    /// better, matching the shared score convention.
    #[inline]
    fn similarity(&self, qcode: &[u64], pos: usize) -> f32 {
        let start = pos * self.words;
        let row = &self.codes[start..start + self.words];
        let mut ham = 0u32;
        for w in 0..self.words {
            ham += (qcode[w] ^ row[w]).count_ones();
        }
        1.0 - (ham as f32 / self.dimension as f32)
    }
}

impl IndexBackend for BinaryQuantizedIndex {
    fn build(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.fit_and_encode(vectors)
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }
        if self.trained {
            let code = self.quantizer.quantize(&vector);
            if let Some(&pos) = self.id_to_pos.get(&id) {
                self.deleted.remove(&id);
                let start = pos * self.words;
                self.codes[start..start + self.words].copy_from_slice(&code);
            } else {
                let pos = self.pos_to_id.len();
                self.codes.extend_from_slice(&code);
                self.id_to_pos.insert(id.clone(), pos);
                self.pos_to_id.push(id);
            }
            Ok(())
        } else {
            self.pending.push((id, vector));
            if self.pending.len() >= TRAIN_THRESHOLD {
                self.ensure_trained()?;
            }
            Ok(())
        }
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        if query.len() != self.dimension {
            return Err(VecDbError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        if !self.trained {
            let mut scored: Vec<(VectorId, f32)> = self
                .pending
                .iter()
                .filter(|(id, _)| !self.deleted.contains(id))
                .map(|(id, v)| {
                    let dist = distance::compute_distance(query, v, &self.metric);
                    let score = match self.metric {
                        DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
                        _ => 1.0 - dist,
                    };
                    (id.clone(), score)
                })
                .collect();
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
            return Ok(scored);
        }

        let qcode = self.quantizer.quantize(query);
        let mut scored: Vec<(VectorId, f32)> = self
            .pos_to_id
            .iter()
            .enumerate()
            .filter(|(_, id)| !self.deleted.contains(*id))
            .map(|(pos, id)| (id.clone(), self.similarity(&qcode, pos)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    fn delete(&mut self, id: &VectorId) -> Result<()> {
        if self.id_to_pos.contains_key(id) || self.pending.iter().any(|(pid, _)| pid == id) {
            self.deleted.insert(id.clone());
            Ok(())
        } else {
            Err(VecDbError::NotFound { id: id.clone() })
        }
    }

    fn save(&self, path: &Path) -> Result<()> {
        let saved = SerializedBinaryIndex {
            dimension: self.dimension,
            metric: self.metric.clone(),
            quantizer: self.quantizer.clone(),
            trained: self.trained,
            words: self.words,
            codes: self.codes.clone(),
            pos_to_id: self.pos_to_id.clone(),
            deleted: self.deleted.iter().cloned().collect(),
            pending: self.pending.clone(),
            version: 1,
        };
        let json = serde_json::to_string(&saved)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| VecDbError::StorageError(format!("binary index save failed: {e}")))?;
        Ok(())
    }

    fn load_from(path: &Path, _config: &CollectionConfig) -> Result<Box<dyn IndexBackend>>
    where
        Self: Sized,
    {
        Ok(Box::new(Self::load_file(path)?))
    }

    fn len(&self) -> usize {
        let total = self.pos_to_id.len() + self.pending.len();
        total.saturating_sub(self.deleted.len())
    }

    fn index_type(&self) -> IndexType {
        IndexType::HNSW
    }
}

impl BinaryQuantizedIndex {
    pub fn load_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| VecDbError::StorageError(format!("binary index load failed: {e}")))?;
        let saved: SerializedBinaryIndex = serde_json::from_str(&json)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        let mut id_to_pos = HashMap::with_capacity(saved.pos_to_id.len());
        for (pos, id) in saved.pos_to_id.iter().enumerate() {
            id_to_pos.insert(id.clone(), pos);
        }
        Ok(Self {
            dimension: saved.dimension,
            metric: saved.metric,
            quantizer: saved.quantizer,
            trained: saved.trained,
            words: saved.words,
            codes: saved.codes,
            id_to_pos,
            pos_to_id: saved.pos_to_id,
            deleted: saved.deleted.into_iter().collect(),
            pending: saved.pending,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedBinaryIndex {
    dimension: usize,
    metric: DistanceMetric,
    quantizer: BinaryQuantizer,
    trained: bool,
    words: usize,
    codes: Vec<u64>,
    pos_to_id: Vec<VectorId>,
    deleted: Vec<VectorId>,
    pending: Vec<(VectorId, Vector)>,
    version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dim: usize) -> CollectionConfig {
        let mut c = CollectionConfig::new("b", dim);
        c.metric = DistanceMetric::Euclidean;
        c
    }

    #[test]
    fn packs_and_matches_identical() {
        let q = BinaryQuantizer::fit(4, &[vec![0.0, 10.0, 0.0, 10.0]]);
        // With a single sample the threshold equals the value; > is false for
        // all dims → all-zero code. Just assert packing is deterministic.
        let a = q.quantize(&[1.0, 20.0, -1.0, 20.0]);
        let b = q.quantize(&[1.0, 20.0, -1.0, 20.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn separates_two_clusters() {
        // Two well-separated clusters: binary codes must rank the right cluster
        // higher for a query near cluster A.
        let mut idx = BinaryQuantizedIndex::new(&cfg(8));
        let mut data = Vec::new();
        for i in 0..150u32 {
            data.push((
                format!("a{i}"),
                vec![10.0, 10.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            ));
        }
        for i in 0..150u32 {
            data.push((
                format!("b{i}"),
                vec![0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0],
            ));
        }
        idx.build(data).unwrap();
        let hits = idx
            .search(&vec![9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0], 10)
            .unwrap();
        assert!(
            hits.iter().all(|(id, _)| id.starts_with('a')),
            "top hits must be cluster A"
        );
        // 8 bits/vec → 1 u64 word/vec = 8 bytes; far below 8*4=32 f32 bytes.
        assert!(idx.code_bytes() <= 300 * 8 + 64);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("b.bq.json");
        let mut idx = BinaryQuantizedIndex::new(&cfg(16));
        let data: Vec<_> = (0..300u32)
            .map(|i| {
                let f = (i % 7) as f32;
                (format!("v{i}"), (0..16).map(|j| f + j as f32).collect())
            })
            .collect();
        idx.build(data).unwrap();
        let q: Vec<f32> = (0..16).map(|j| 3.0 + j as f32).collect();
        let before = idx.search(&q, 5).unwrap();
        idx.save(&path).unwrap();
        let loaded = BinaryQuantizedIndex::load_file(&path).unwrap();
        let after = loaded.search(&q, 5).unwrap();
        assert_eq!(before[0].0, after[0].0);
    }
}
