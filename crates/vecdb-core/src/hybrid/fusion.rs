use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::errors::Result;
use crate::types::{SearchResult, VectorId};

// ─────────────────────────────────────────────────────────────────
// FusionStrategy
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum FusionStrategy {
    /// Weighted sum of normalized scores: alpha*dense + (1-alpha)*sparse
    #[default]
    WeightedSum,
    /// Reciprocal Rank Fusion: combines ranked positions not raw scores
    ReciprocalRankFusion { k: f32 },
}

// ─────────────────────────────────────────────────────────────────
// Normalization free functions
// ─────────────────────────────────────────────────────────────────

/// Min-max normalize a score slice into [0.0, 1.0].
///
/// - Empty input → empty output.
/// - All-equal scores (range < EPSILON) → all map to 1.0 (avoids divide-by-zero).
/// - Output order matches input order.
///
/// Performance: single-pass fold finds min and max simultaneously.
// Before: two separate fold passes (one for min, one for max) over the score slice
// After:  one fold pass with a tuple accumulator; ~2× fewer memory reads for large slices
pub fn min_max_normalize(scores: &[f32]) -> Vec<f32> {
    const EPSILON: f32 = 1e-9;
    if scores.is_empty() {
        return vec![];
    }
    // Single-pass: find min and max in one traversal.
    let (min, max) = scores
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &s| {
            (mn.min(s), mx.max(s))
        });
    let range = max - min;
    if range < EPSILON {
        // All scores equal — map to 1.0 to avoid NaN from 0/0.
        return vec![1.0f32; scores.len()];
    }
    scores.iter().map(|&s| (s - min) / range).collect()
}

/// Softmax normalize scores.
///
/// Uses the max-subtraction trick for numerical stability.
/// - Empty input → empty output.
/// - Zero sum (should not occur in practice) → uniform 1/n.
pub fn softmax_normalize(scores: &[(VectorId, f32)]) -> Vec<(VectorId, f32)> {
    if scores.is_empty() {
        return vec![];
    }

    let max = scores
        .iter()
        .map(|(_, s)| *s)
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));

    let exp_scores: Vec<f32> = scores.iter().map(|(_, s)| (s - max).exp()).collect();
    let sum: f32 = exp_scores.iter().sum();

    if sum == 0.0 {
        let uniform = 1.0 / scores.len() as f32;
        return scores.iter().map(|(id, _)| (id.clone(), uniform)).collect();
    }

    scores
        .iter()
        .zip(exp_scores.iter())
        .map(|((id, _), exp_s)| (id.clone(), exp_s / sum))
        .collect()
}

// ─────────────────────────────────────────────────────────────────
// Fusion free functions
// ─────────────────────────────────────────────────────────────────

/// Weighted sum fusion.
///
/// `final_score = alpha * dense_norm + (1 - alpha) * sparse_norm`
///
/// Documents present in only one list get 0.0 for the missing component.
/// Output sorted descending by final score.
pub fn weighted_sum_fusion(
    dense_scores: &[(VectorId, f32)],
    sparse_scores: &[(VectorId, f32)],
    alpha: f32,
) -> Vec<(VectorId, f32)> {
    let alpha = alpha.clamp(0.0, 1.0);

    // Extract raw score values, normalize, then zip the IDs back.
    let dense_raw: Vec<f32> = dense_scores.iter().map(|(_, s)| *s).collect();
    let sparse_raw: Vec<f32> = sparse_scores.iter().map(|(_, s)| *s).collect();
    let dense_norm_vals = min_max_normalize(&dense_raw);
    let sparse_norm_vals = min_max_normalize(&sparse_raw);

    // id → (dense_normalized, sparse_normalized)
    let mut combined: HashMap<VectorId, (f32, f32)> = HashMap::new();

    for ((id, _), norm) in dense_scores.iter().zip(dense_norm_vals.iter()) {
        combined.insert(id.clone(), (*norm, 0.0));
    }
    for ((id, _), norm) in sparse_scores.iter().zip(sparse_norm_vals.iter()) {
        combined
            .entry(id.clone())
            .and_modify(|e| e.1 = *norm)
            .or_insert((0.0, *norm));
    }

    let mut results: Vec<(VectorId, f32)> = combined
        .into_iter()
        .map(|(id, (d, s))| (id, alpha * d + (1.0 - alpha) * s))
        .collect();

    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// Reciprocal Rank Fusion.
///
/// `RRF(d) = Σ 1 / (k + rank(d, list))` where rank is 1-indexed.
///
/// Output sorted descending by RRF score.
pub fn reciprocal_rank_fusion(
    dense_results: &[(VectorId, f32)],
    sparse_results: &[(VectorId, f32)],
    k: f32,
) -> Vec<(VectorId, f32)> {
    let mut rrf_scores: HashMap<VectorId, f32> = HashMap::new();

    for (rank, (id, _)) in dense_results.iter().enumerate() {
        let contribution = 1.0 / (k + (rank as f32 + 1.0));
        *rrf_scores.entry(id.clone()).or_insert(0.0) += contribution;
    }
    for (rank, (id, _)) in sparse_results.iter().enumerate() {
        let contribution = 1.0 / (k + (rank as f32 + 1.0));
        *rrf_scores.entry(id.clone()).or_insert(0.0) += contribution;
    }

    let mut results: Vec<(VectorId, f32)> = rrf_scores.into_iter().collect();
    results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─────────────────────────────────────────────────────────────────
// HybridEngine
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridEngine {
    /// Weight for dense scores (1.0 = pure dense, 0.0 = pure sparse).
    pub alpha: f32,
    /// Fusion strategy to use.
    pub strategy: FusionStrategy,
    /// Oversampling factor: dense retrieves k * oversample_factor candidates
    /// before sparse re-scoring.
    pub oversample_factor: usize,
}

impl Default for HybridEngine {
    fn default() -> Self {
        Self {
            alpha: 0.7,
            strategy: FusionStrategy::WeightedSum,
            oversample_factor: 5,
        }
    }
}

impl HybridEngine {
    /// Construct a new engine.
    ///
    /// # Panics
    /// Panics if `alpha` is not in `[0.0, 1.0]` or `oversample_factor < 1`.
    /// These are programming errors, not runtime errors.
    pub fn new(alpha: f32, strategy: FusionStrategy, oversample_factor: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&alpha),
            "alpha must be in [0.0, 1.0], got {alpha}"
        );
        assert!(
            oversample_factor >= 1,
            "oversample_factor must be >= 1, got {oversample_factor}"
        );
        Self {
            alpha,
            strategy,
            oversample_factor,
        }
    }

    /// Fuse pre-computed dense and sparse result lists into a single ranked list.
    ///
    /// - Both empty → empty result.
    /// - Dense only → pure dense path; sparse_score = None.
    /// - Sparse only → pure sparse path; dense_score = None.
    /// - Both present → fuse with `self.strategy`.
    ///
    /// `dense_score` and `sparse_score` in each `SearchResult` hold the
    /// original (pre-normalization) scores.  `score` holds the fused value.
    /// `payload` and `text` are left as `Null` / `None` — caller hydrates.
    pub fn fuse(
        &self,
        dense_results: Vec<(VectorId, f32)>,
        sparse_results: Vec<(VectorId, f32)>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if dense_results.is_empty() && sparse_results.is_empty() {
            return Ok(vec![]);
        }

        // Pure sparse path.
        if dense_results.is_empty() {
            let results = sparse_results
                .into_iter()
                .take(k)
                .map(|(id, score)| SearchResult {
                    id,
                    score,
                    dense_score: None,
                    sparse_score: Some(score),
                    payload: serde_json::Value::Null,
                    text: None,
                })
                .collect();
            return Ok(results);
        }

        // Pure dense path.
        if sparse_results.is_empty() {
            let results = dense_results
                .into_iter()
                .take(k)
                .map(|(id, score)| SearchResult {
                    id,
                    score,
                    dense_score: Some(score),
                    sparse_score: None,
                    payload: serde_json::Value::Null,
                    text: None,
                })
                .collect();
            return Ok(results);
        }

        // Build original-score lookup maps before consuming the vecs.
        let dense_map: HashMap<VectorId, f32> = dense_results.iter().cloned().collect();
        let sparse_map: HashMap<VectorId, f32> = sparse_results.iter().cloned().collect();

        // Fuse.
        let fused = match &self.strategy {
            FusionStrategy::WeightedSum => {
                weighted_sum_fusion(&dense_results, &sparse_results, self.alpha)
            }
            FusionStrategy::ReciprocalRankFusion { k: rrf_k } => {
                reciprocal_rank_fusion(&dense_results, &sparse_results, *rrf_k)
            }
        };

        let results = fused
            .into_iter()
            .take(k)
            .map(|(id, final_score)| {
                let d_score = dense_map.get(&id).copied();
                let s_score = sparse_map.get(&id).copied();
                SearchResult {
                    id,
                    score: final_score,
                    dense_score: d_score,
                    sparse_score: s_score,
                    payload: serde_json::Value::Null,
                    text: None,
                }
            })
            .collect();

        Ok(results)
    }

    /// How many dense candidates to fetch before sparse re-scoring.
    ///
    /// Returns `k * oversample_factor` (minimum `k`).
    pub fn compute_oversample_k(&self, k: usize) -> usize {
        (k * self.oversample_factor).max(k)
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Test 1 ───────────────────────────────────────────────────
    #[test]
    fn test_min_max_normalize_basic() {
        // Updated for P23: signature is now &[f32] → Vec<f32> (no IDs).
        let scores = vec![0.0_f32, 0.5, 1.0];
        let result = min_max_normalize(&scores);
        assert_eq!(result.len(), 3);
        // Order preserved.
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.5).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
    }

    // ── Test 2 ───────────────────────────────────────────────────
    #[test]
    fn test_min_max_normalize_all_equal() {
        // Updated for P23: all-equal input maps to 1.0 (EPSILON fast-path).
        let scores = vec![0.7_f32, 0.7, 0.7];
        let result = min_max_normalize(&scores);
        assert_eq!(result.len(), 3);
        for &score in &result {
            assert!(
                (score - 1.0).abs() < 1e-6,
                "expected 1.0 for all-equal input, got {score}"
            );
        }
    }

    // ── Test 3 ───────────────────────────────────────────────────
    #[test]
    fn test_min_max_normalize_empty() {
        // Updated for P23: pass &[f32] directly.
        let result = min_max_normalize(&[]);
        assert!(result.is_empty());
    }

    // ── Test 4 ───────────────────────────────────────────────────
    #[test]
    fn test_weighted_sum_fusion_pure_dense() {
        // alpha=1.0 → dense dominates; "a" has highest dense score.
        let dense = vec![
            ("a".to_string(), 0.9_f32),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.1),
        ];
        let sparse = vec![
            ("a".to_string(), 0.1_f32),
            ("b".to_string(), 0.9),
            ("c".to_string(), 0.5),
        ];
        let result = weighted_sum_fusion(&dense, &sparse, 1.0);
        assert!(!result.is_empty());
        assert_eq!(result[0].0, "a", "pure dense: 'a' must rank first");
        for (_, s) in &result {
            assert!(*s >= 0.0 && *s <= 1.0 + 1e-6, "score {s} out of [0,1]");
        }
    }

    // ── Test 5 ───────────────────────────────────────────────────
    #[test]
    fn test_weighted_sum_fusion_pure_sparse() {
        // alpha=0.0 → sparse dominates; "b" has highest sparse score.
        let dense = vec![
            ("a".to_string(), 0.9_f32),
            ("b".to_string(), 0.5),
            ("c".to_string(), 0.1),
        ];
        let sparse = vec![
            ("b".to_string(), 0.9_f32),
            ("a".to_string(), 0.1),
            ("c".to_string(), 0.5),
        ];
        let result = weighted_sum_fusion(&dense, &sparse, 0.0);
        assert!(!result.is_empty());
        assert_eq!(result[0].0, "b", "pure sparse: 'b' must rank first");
    }

    // ── Test 6 ───────────────────────────────────────────────────
    #[test]
    fn test_weighted_sum_fusion_doc_only_in_one_list() {
        // "c" only in sparse; "a","b" only in dense.
        let dense = vec![("a".to_string(), 0.9_f32), ("b".to_string(), 0.5)];
        let sparse = vec![("c".to_string(), 0.8_f32)];
        let result = weighted_sum_fusion(&dense, &sparse, 0.5);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert_eq!(result.len(), 3);
    }

    // ── Test 7 ───────────────────────────────────────────────────
    /// RRF: document ranked 2nd dense + 1st sparse beats others.
    ///
    /// With k=60 and these rankings:
    ///   a: rank1 dense + rank3 sparse  → 1/61 + 1/63 ≈ 0.032266
    ///   b: rank2 dense + rank1 sparse  → 1/62 + 1/61 ≈ 0.032522  ← wins
    ///   c: rank3 dense + rank2 sparse  → 1/63 + 1/62 ≈ 0.032002
    #[test]
    fn test_rrf_basic() {
        let dense = vec![
            ("a".to_string(), 0.9_f32), // rank 1
            ("b".to_string(), 0.5),     // rank 2
            ("c".to_string(), 0.1),     // rank 3
        ];
        // b=rank1 in sparse, c=rank2, a=rank3 → b gets 1/61+1/62 > a's 1/61+1/63
        let sparse = vec![
            ("b".to_string(), 0.9_f32), // rank 1
            ("c".to_string(), 0.5),     // rank 2
            ("a".to_string(), 0.1),     // rank 3
        ];
        let result = reciprocal_rank_fusion(&dense, &sparse, 60.0);
        assert_eq!(result.len(), 3);
        let ids: Vec<&str> = result.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"a"));
        assert!(ids.contains(&"b"));
        assert!(ids.contains(&"c"));
        assert_eq!(
            result[0].0, "b",
            "b (2nd dense + 1st sparse) must rank first"
        );
    }

    // ── Test 8 ───────────────────────────────────────────────────
    #[test]
    fn test_hybrid_engine_fuse_weighted_sum() {
        let engine = HybridEngine::new(0.7, FusionStrategy::WeightedSum, 5);
        let dense = vec![
            ("doc1".to_string(), 0.95_f32),
            ("doc2".to_string(), 0.7),
            ("doc3".to_string(), 0.3),
        ];
        let sparse = vec![
            ("doc2".to_string(), 0.9_f32),
            ("doc1".to_string(), 0.4),
            ("doc3".to_string(), 0.2),
        ];
        let results = engine.fuse(dense, sparse, 3).unwrap();
        assert_eq!(results.len(), 3);
        // All results have both dense and sparse scores.
        for r in &results {
            assert!(r.dense_score.is_some(), "dense_score must be Some");
            assert!(r.sparse_score.is_some(), "sparse_score must be Some");
        }
        // Sorted descending.
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results not sorted: {} < {}",
                results[i - 1].score,
                results[i].score
            );
        }
    }

    // ── Test 9 ───────────────────────────────────────────────────
    #[test]
    fn test_hybrid_engine_fuse_empty_sparse() {
        let engine = HybridEngine::default();
        let dense = vec![("doc1".to_string(), 0.9_f32), ("doc2".to_string(), 0.5)];
        let sparse: Vec<(VectorId, f32)> = vec![];
        let results = engine.fuse(dense, sparse, 2).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(r.dense_score.is_some());
            assert!(
                r.sparse_score.is_none(),
                "pure dense path must have sparse_score=None"
            );
        }
    }

    // ── Test 10 ──────────────────────────────────────────────────
    #[test]
    fn test_hybrid_engine_fuse_empty_dense() {
        let engine = HybridEngine::default();
        let dense: Vec<(VectorId, f32)> = vec![];
        let sparse = vec![("doc1".to_string(), 8.5_f32), ("doc2".to_string(), 3.2)];
        let results = engine.fuse(dense, sparse, 2).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert!(
                r.dense_score.is_none(),
                "pure sparse path must have dense_score=None"
            );
            assert!(r.sparse_score.is_some());
        }
    }
}
