//! Scalar (int8) quantization — an optional in-memory index mode.
//!
//! Each float32 dimension is linearly mapped to a signed 8-bit code using a
//! per-dimension min/scale learned from the data (asymmetric quantization: the
//! query stays full-precision, only stored vectors are quantized). The index is
//! a flat brute-force scanner over the codes.
//!
//! Memory: codes are 1 byte/dim vs 4 bytes/dim for float32 → ~4× smaller
//! in-memory footprint. The on-disk float32 vectors remain authoritative and
//! are used to retrain the quantizer on rebuild.
//!
//! Distance kernel: reconstructing every stored vector to float per comparison
//! is wasteful. Instead we expand the distance so the per-vector hot loop is a
//! single `f32·i8` dot product, using precomputed norms:
//!
//! A stored vector reconstructs as `x_i = o_i + s_i·c_i` where `o_i = min_i +
//! 128·s_i` and `s_i` is the per-dim scale. Then `⟨q,x⟩ = ⟨q,o⟩ + Σ_i (q_i·s_i)·c_i`.
//! `⟨q,o⟩`, `‖q‖²`, and `b_i = q_i·s_i` are computed once per query; `‖x‖²` is
//! precomputed per stored vector. From `⟨q,x⟩`, `‖q‖²`, `‖x‖²` we recover the
//! exact L2 / cosine / dot score — same numbers as reconstructing, one mul-add
//! per dim over 1-byte codes.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::index::backend::IndexBackend;
use crate::index::distance;
use crate::types::{CollectionConfig, DistanceMetric, IndexType, Vector, VectorId};

// ──────────────────────────────────────────────
// ScalarQuantizer — per-dimension int8 codec
// ──────────────────────────────────────────────

/// Per-dimension linear int8 quantizer. Codes are `i8` in `[-128, 127]`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ScalarQuantizer {
    /// Per-dimension minimum value seen during fit.
    pub dmin: Vec<f32>,
    /// Per-dimension step size: `(max - min) / 255`. Zero-range dims use 1.0.
    pub dscale: Vec<f32>,
}

impl ScalarQuantizer {
    /// Learn per-dimension min/scale from a sample of vectors.
    pub fn fit(dimension: usize, sample: &[Vector]) -> Self {
        let mut dmin = vec![f32::INFINITY; dimension];
        let mut dmax = vec![f32::NEG_INFINITY; dimension];
        for v in sample {
            for (i, &x) in v.iter().enumerate().take(dimension) {
                if x < dmin[i] {
                    dmin[i] = x;
                }
                if x > dmax[i] {
                    dmax[i] = x;
                }
            }
        }
        let mut dscale = vec![1.0f32; dimension];
        for i in 0..dimension {
            if !dmin[i].is_finite() {
                dmin[i] = 0.0;
            }
            let range = dmax[i] - dmin[i];
            dscale[i] = if range > 0.0 { range / 255.0 } else { 1.0 };
        }
        Self { dmin, dscale }
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.dmin.len()
    }

    /// Quantize one full-precision vector to int8 codes.
    pub fn quantize(&self, v: &[f32]) -> Vec<i8> {
        let d = self.dimension();
        let mut out = vec![0i8; d];
        for i in 0..d {
            let u = ((v[i] - self.dmin[i]) / self.dscale[i]).round();
            let code = (u.clamp(0.0, 255.0) as i32) - 128;
            out[i] = code as i8;
        }
        out
    }

    /// Reconstruct dimension `i` of a code back to full precision.
    #[inline]
    pub fn reconstruct_at(&self, code: i8, i: usize) -> f32 {
        self.dmin[i] + ((code as i32 + 128) as f32) * self.dscale[i]
    }

    /// Reconstruct a full code slice into a float vector.
    pub fn dequantize(&self, codes: &[i8]) -> Vec<f32> {
        (0..self.dimension())
            .map(|i| self.reconstruct_at(codes[i], i))
            .collect()
    }

    /// Per-dimension offset `o_i = min_i + 128·scale_i` (the constant part of
    /// the reconstruction `x_i = o_i + scale_i·c_i`).
    fn offsets(&self) -> Vec<f32> {
        (0..self.dimension())
            .map(|i| self.dmin[i] + 128.0 * self.dscale[i])
            .collect()
    }
}

// ──────────────────────────────────────────────
// ScalarQuantizedIndex — flat int8 scanner
// ──────────────────────────────────────────────

pub struct ScalarQuantizedIndex {
    dimension: usize,
    metric: DistanceMetric,
    quantizer: ScalarQuantizer,
    trained: bool,
    /// int8 codes, row-major: vector `pos` occupies `codes[pos*d .. (pos+1)*d]`.
    codes: Vec<i8>,
    /// ‖x‖² of each stored (reconstructed) vector, parallel to `pos_to_id`.
    dnorm: Vec<f32>,
    id_to_pos: HashMap<VectorId, usize>,
    pos_to_id: Vec<VectorId>,
    deleted: HashSet<VectorId>,
    /// Vectors inserted before the quantizer is trained, kept until the next
    /// (re)train materializes them into codes.
    pending: Vec<(VectorId, Vector)>,
}

/// Train once the pending buffer reaches this many vectors (also triggered
/// lazily when a search forces materialization).
const TRAIN_THRESHOLD: usize = 256;

impl ScalarQuantizedIndex {
    pub fn new(config: &CollectionConfig) -> Self {
        Self {
            dimension: config.dimension,
            metric: config.metric.clone(),
            quantizer: ScalarQuantizer::default(),
            trained: false,
            codes: Vec::new(),
            dnorm: Vec::new(),
            id_to_pos: HashMap::new(),
            pos_to_id: Vec::new(),
            deleted: HashSet::new(),
            pending: Vec::new(),
        }
    }

    /// Approximate resident memory of the code store, in bytes (the figure the
    /// quantization mode is meant to shrink). Counts codes + per-dim params +
    /// the per-vector norm cache; excludes id strings, which the float path
    /// also carries.
    pub fn code_bytes(&self) -> usize {
        self.codes.len() * std::mem::size_of::<i8>()
            + self.dnorm.len() * std::mem::size_of::<f32>()
            + self.quantizer.dmin.len() * std::mem::size_of::<f32>() * 2
    }

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

    /// (Re)train the quantizer from `vectors` and encode them all, replacing any
    /// existing state. Cosine vectors are normalized before encoding.
    fn fit_and_encode(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        self.id_to_pos.clear();
        self.pos_to_id.clear();
        self.codes.clear();
        self.dnorm.clear();
        self.deleted.clear();
        self.pending.clear();

        if vectors.is_empty() {
            self.trained = false;
            return Ok(());
        }

        let normalized: Vec<(VectorId, Vector)> = vectors
            .into_iter()
            .map(|(id, v)| {
                if v.len() != self.dimension {
                    return Err(VecDbError::DimensionMismatch {
                        expected: self.dimension,
                        got: v.len(),
                    });
                }
                Ok((id, self.maybe_normalize(v)))
            })
            .collect::<Result<_>>()?;

        let sample: Vec<Vector> = normalized.iter().map(|(_, v)| v.clone()).collect();
        self.quantizer = ScalarQuantizer::fit(self.dimension, &sample);

        self.codes.reserve(normalized.len() * self.dimension);
        self.dnorm.reserve(normalized.len());
        for (id, v) in normalized {
            let pos = self.pos_to_id.len();
            let code = self.quantizer.quantize(&v);
            self.push_code(pos, &code);
            self.id_to_pos.insert(id.clone(), pos);
            self.pos_to_id.push(id);
        }
        self.trained = true;
        Ok(())
    }

    /// Append code row + its reconstructed ‖x‖² cache.
    fn push_code(&mut self, _pos: usize, code: &[i8]) {
        self.codes.extend_from_slice(code);
        self.dnorm.push(self.code_norm_sq(code));
    }

    /// ‖x‖² of the reconstruction of a code row.
    fn code_norm_sq(&self, code: &[i8]) -> f32 {
        let mut s = 0.0f32;
        for (i, &c) in code.iter().enumerate() {
            let x = self.quantizer.reconstruct_at(c, i);
            s += x * x;
        }
        s
    }

    fn ensure_trained(&mut self) -> Result<()> {
        if !self.pending.is_empty() {
            let batch = std::mem::take(&mut self.pending);
            let mut all: Vec<(VectorId, Vector)> = self.live_vectors_reconstructed();
            all.extend(batch);
            self.fit_and_encode(all)?;
        }
        Ok(())
    }

    fn live_vectors_reconstructed(&self) -> Vec<(VectorId, Vector)> {
        self.pos_to_id
            .iter()
            .enumerate()
            .filter(|(_, id)| !self.deleted.contains(*id))
            .map(|(pos, id)| {
                let start = pos * self.dimension;
                let codes = &self.codes[start..start + self.dimension];
                (id.clone(), self.quantizer.dequantize(codes))
            })
            .collect()
    }

    /// Score every live stored vector against `query` using the precomputed-norm
    /// dot expansion, and return the top-`k` (id, score) with the shared score
    /// convention (higher = better).
    fn scan(&self, query: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        let d = self.dimension;
        let o = self.quantizer.offsets();
        let s = &self.quantizer.dscale;

        // Per-query precomputation.
        let qnorm: f32 = query.iter().map(|v| v * v).sum();
        let qo: f32 = query.iter().zip(o.iter()).map(|(q, o)| q * o).sum();
        let b: Vec<f32> = query.iter().zip(s.iter()).map(|(q, s)| q * s).collect();

        let mut scored: Vec<(VectorId, f32)> = Vec::with_capacity(self.pos_to_id.len());
        for (pos, id) in self.pos_to_id.iter().enumerate() {
            if self.deleted.contains(id) {
                continue;
            }
            let start = pos * d;
            let codes = &self.codes[start..start + d];
            // dot(b, codes): the only per-vector inner loop, f32·i8.
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += b[i] * codes[i] as f32;
            }
            let dot_qx = qo + acc;
            let xnorm = self.dnorm[pos];
            let score = match self.metric {
                DistanceMetric::Euclidean => {
                    let dist2 = (qnorm - 2.0 * dot_qx + xnorm).max(0.0);
                    self.to_score(dist2.sqrt())
                }
                DistanceMetric::Cosine => {
                    let denom = qnorm.sqrt() * xnorm.sqrt();
                    if denom == 0.0 {
                        0.0
                    } else {
                        dot_qx / denom
                    }
                }
                DistanceMetric::DotProduct => dot_qx,
            };
            scored.push((id.clone(), score));
        }
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

impl IndexBackend for ScalarQuantizedIndex {
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
            let v = self.maybe_normalize(vector);
            let code = self.quantizer.quantize(&v);
            let norm = self.code_norm_sq(&code);
            if let Some(&pos) = self.id_to_pos.get(&id) {
                self.deleted.remove(&id);
                let start = pos * self.dimension;
                self.codes[start..start + self.dimension].copy_from_slice(&code);
                self.dnorm[pos] = norm;
            } else {
                let pos = self.pos_to_id.len();
                self.push_code(pos, &code);
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

        // Untrained pending vectors: full-precision brute force over them (the
        // search path takes &self, so we cannot train here).
        if !self.trained {
            let q = self.maybe_normalize(query.clone());
            let mut scored: Vec<(VectorId, f32)> = self
                .pending
                .iter()
                .filter(|(id, _)| !self.deleted.contains(id))
                .map(|(id, v)| {
                    let vn = self.maybe_normalize(v.clone());
                    (
                        id.clone(),
                        self.to_score(distance::compute_distance(&q, &vn, &self.metric)),
                    )
                })
                .collect();
            scored.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(k);
            return Ok(scored);
        }

        let q = self.maybe_normalize(query.clone());
        Ok(self.scan(&q, k))
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
        let saved = SerializedScalarIndex {
            dimension: self.dimension,
            metric: self.metric.clone(),
            quantizer: self.quantizer.clone(),
            trained: self.trained,
            codes: self.codes.clone(),
            dnorm: self.dnorm.clone(),
            pos_to_id: self.pos_to_id.clone(),
            deleted: self.deleted.iter().cloned().collect(),
            pending: self.pending.clone(),
            version: 1,
        };
        let json = serde_json::to_string(&saved)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        std::fs::write(path, json)
            .map_err(|e| VecDbError::StorageError(format!("scalar index save failed: {e}")))?;
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
        // Quantization is tracked on CollectionConfig; reuse HNSW discriminant
        // for the coarse stats field.
        IndexType::HNSW
    }
}

impl ScalarQuantizedIndex {
    pub fn load_file(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| VecDbError::StorageError(format!("scalar index load failed: {e}")))?;
        let saved: SerializedScalarIndex = serde_json::from_str(&json)
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
            codes: saved.codes,
            dnorm: saved.dnorm,
            id_to_pos,
            pos_to_id: saved.pos_to_id,
            deleted: saved.deleted.into_iter().collect(),
            pending: saved.pending,
        })
    }
}

#[derive(Serialize, Deserialize)]
struct SerializedScalarIndex {
    dimension: usize,
    metric: DistanceMetric,
    quantizer: ScalarQuantizer,
    trained: bool,
    codes: Vec<i8>,
    dnorm: Vec<f32>,
    pos_to_id: Vec<VectorId>,
    deleted: Vec<VectorId>,
    pending: Vec<(VectorId, Vector)>,
    version: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(dim: usize, metric: DistanceMetric) -> CollectionConfig {
        let mut c = CollectionConfig::new("q", dim);
        c.metric = metric;
        c
    }

    #[test]
    fn quantizer_roundtrip_is_close() {
        let sample = vec![
            vec![0.0, 10.0, -5.0],
            vec![1.0, 8.0, -1.0],
            vec![0.5, 9.0, -3.0],
        ];
        let q = ScalarQuantizer::fit(3, &sample);
        for v in &sample {
            let deq = q.dequantize(&q.quantize(v));
            for (a, b) in v.iter().zip(deq.iter()) {
                assert!((a - b).abs() < 0.1, "roundtrip {a} vs {b}");
            }
        }
    }

    #[test]
    fn bulk_build_search_euclidean() {
        let mut idx = ScalarQuantizedIndex::new(&cfg(4, DistanceMetric::Euclidean));
        let mut data = Vec::new();
        for i in 0..500u32 {
            let f = i as f32;
            data.push((format!("v{i}"), vec![f, f + 1.0, f + 2.0, f + 3.0]));
        }
        idx.build(data).unwrap();
        assert_eq!(idx.len(), 500);
        // 500 points linearly fill the value range, so adjacent vectors differ
        // by less than one int8 step (range/255) — the top hit lands within a
        // quantization-adjacent neighborhood of the true answer.
        let hits = idx.search(&vec![100.0, 101.0, 102.0, 103.0], 1).unwrap();
        let top: u32 = hits[0].0.trim_start_matches('v').parse().unwrap();
        assert!((99..=101).contains(&top), "top hit v{top} not near v100");
        assert!(idx.code_bytes() < 500 * 4 * 4, "codes must be << f32 size");
    }

    #[test]
    fn kernel_matches_reconstruction() {
        // The dot-expansion score must equal scoring via explicit dequantize.
        let mut idx = ScalarQuantizedIndex::new(&cfg(8, DistanceMetric::Euclidean));
        let data: Vec<_> = (0..300u32)
            .map(|i| {
                let f = i as f32;
                (format!("v{i}"), (0..8).map(|j| f + j as f32 * 0.3).collect())
            })
            .collect();
        idx.build(data).unwrap();
        let q: Vec<f32> = (0..8).map(|j| 50.0 + j as f32).collect();
        let hits = idx.search(&q, 5).unwrap();
        // Recompute the top hit's score by explicit reconstruction.
        let (id, score) = &hits[0];
        let pos = idx.id_to_pos[id];
        let codes = &idx.codes[pos * 8..pos * 8 + 8];
        let x = idx.quantizer.dequantize(codes);
        let dist = distance::compute_distance(&q, &x, &DistanceMetric::Euclidean);
        let expect = idx.to_score(dist);
        assert!((score - expect).abs() < 1e-3, "{score} vs {expect}");
    }

    #[test]
    fn streaming_insert_trains_then_queries() {
        let mut idx = ScalarQuantizedIndex::new(&cfg(3, DistanceMetric::Euclidean));
        idx.insert("a".into(), vec![1.0, 0.0, 0.0]).unwrap();
        idx.insert("b".into(), vec![0.0, 1.0, 0.0]).unwrap();
        assert_eq!(idx.search(&vec![1.0, 0.0, 0.0], 1).unwrap()[0].0, "a");
        for i in 0..TRAIN_THRESHOLD {
            idx.insert(format!("x{i}"), vec![i as f32, i as f32, i as f32])
                .unwrap();
        }
        assert!(idx.trained);
        assert_eq!(idx.search(&vec![1.0, 0.0, 0.0], 1).unwrap()[0].0, "a");
    }

    #[test]
    fn delete_excludes_from_results() {
        let mut idx = ScalarQuantizedIndex::new(&cfg(2, DistanceMetric::Euclidean));
        let data: Vec<_> = (0..300u32)
            .map(|i| (format!("v{i}"), vec![i as f32, 0.0]))
            .collect();
        idx.build(data).unwrap();
        idx.delete(&"v5".to_string()).unwrap();
        let hits = idx.search(&vec![5.0, 0.0], 5).unwrap();
        assert!(hits.iter().all(|(id, _)| id != "v5"));
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("q.sq.json");
        let mut idx = ScalarQuantizedIndex::new(&cfg(4, DistanceMetric::Euclidean));
        let data: Vec<_> = (0..300u32)
            .map(|i| (format!("v{i}"), vec![i as f32, 1.0, 2.0, 3.0]))
            .collect();
        idx.build(data).unwrap();
        let before = idx.search(&vec![50.0, 1.0, 2.0, 3.0], 3).unwrap();
        idx.save(&path).unwrap();
        let loaded = ScalarQuantizedIndex::load_file(&path).unwrap();
        let after = loaded.search(&vec![50.0, 1.0, 2.0, 3.0], 3).unwrap();
        assert_eq!(before[0].0, after[0].0);
        assert_eq!(before.len(), after.len());
    }
}
