#[cfg(test)]
mod tests {
    use crate::index::distance::{
        cosine_similarity, cosine_similarity_simd, dot_product, dot_product_simd,
    };
    use crate::sparse::inverted::InvertedIndex;
    use crate::types::CollectionConfig;

    // ── Test 1 ───────────────────────────────────────────────────────────────
    /// dot_product_simd must produce the same result as the scalar version for a
    /// typical embedding dimension (1536).
    #[test]
    fn test_dot_product_simd_matches_scalar() {
        let dim = 1536usize;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..dim).map(|i| 1.0 - (i as f32) * 0.0005).collect();

        let scalar = dot_product(&a, &b);
        let simd = dot_product_simd(&a, &b);

        assert!(
            (scalar - simd).abs() < 1e-2,
            "dot_product_simd ({simd}) diverges from scalar ({scalar}) by more than 1e-2"
        );
    }

    // ── Test 2 ───────────────────────────────────────────────────────────────
    /// cosine_similarity_simd must produce the same result as the scalar version
    /// for a typical embedding dimension (1536).
    #[test]
    fn test_cosine_similarity_simd_matches_scalar() {
        let dim = 1536usize;
        let a: Vec<f32> = (0..dim).map(|i| ((i % 17) as f32).sin()).collect();
        let b: Vec<f32> = (0..dim).map(|i| ((i % 13) as f32).cos()).collect();

        let scalar = cosine_similarity(&a, &b);
        let simd = cosine_similarity_simd(&a, &b);

        assert!(
            (scalar - simd).abs() < 1e-4,
            "cosine_similarity_simd ({simd}) diverges from scalar ({scalar}) by more than 1e-4"
        );
    }

    // ── Test 3 ───────────────────────────────────────────────────────────────
    /// For short vectors (n < 8) the SIMD kernels must fall back to the scalar
    /// implementation and produce exact agreement.
    #[test]
    fn test_simd_short_vector_fallback() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // n=4 < 8
        let b = vec![4.0f32, 3.0, 2.0, 1.0];

        let dp_scalar = dot_product(&a, &b);
        let dp_simd = dot_product_simd(&a, &b);
        assert!(
            (dp_scalar - dp_simd).abs() < f32::EPSILON * 10.0,
            "short-vector dot_product: scalar={dp_scalar} simd={dp_simd}"
        );

        let cos_scalar = cosine_similarity(&a, &b);
        let cos_simd = cosine_similarity_simd(&a, &b);
        assert!(
            (cos_scalar - cos_simd).abs() < f32::EPSILON * 10.0,
            "short-vector cosine: scalar={cos_scalar} simd={cos_simd}"
        );
    }

    // ── Test 4 ───────────────────────────────────────────────────────────────
    /// A zero vector must return 0.0 cosine similarity (no NaN / no panic) even
    /// through the SIMD kernel.
    #[test]
    fn test_cosine_simd_zero_vector() {
        let dim = 16usize;
        let zero = vec![0.0f32; dim];
        let other: Vec<f32> = (0..dim).map(|i| i as f32 + 1.0).collect();

        let result = cosine_similarity_simd(&zero, &other);
        assert_eq!(
            result, 0.0,
            "cosine_similarity_simd with zero vector must return 0.0, got {result}"
        );
    }

    // ── Test 5 ───────────────────────────────────────────────────────────────
    /// IVF search with rayon-parallelized scoring must return the correct nearest
    /// neighbour (identical to the deterministic serial result).
    #[test]
    fn test_ivf_search_parallel_correctness() {
        use crate::index::backend::IndexBackend;
        use crate::index::ivf::IvfIndex;
        use crate::types::IndexType;

        let mut cfg = CollectionConfig::new("perf_ivf", 4);
        cfg.index_type = IndexType::IVF;
        let mut idx = IvfIndex::new(&cfg);

        for i in 0..30usize {
            idx.insert(format!("v{i}"), vec![i as f32, 1.0, 0.0, 0.0])
                .unwrap();
        }

        let query = vec![15.0f32, 1.0, 0.0, 0.0];
        let results = idx.search(&query, 3).unwrap();

        assert!(
            !results.is_empty(),
            "parallel IVF search must return results"
        );
        // The closest vector is v15; it must appear in top-3.
        assert!(
            results.iter().any(|(id, _)| id == "v15"),
            "v15 must be in top-3 results; got: {:?}",
            results
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>()
        );
        // Results must be sorted descending by score.
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 >= results[i].1,
                "results not sorted at position {i}: {} < {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    // ── Test 6 ───────────────────────────────────────────────────────────────
    /// `total_tokens` must be incremented on insert and decremented on remove,
    /// matching the sum of `doc_lengths` at every point.
    #[test]
    fn test_inverted_total_tokens_maintained() {
        let mut idx = InvertedIndex::default();

        idx.index_document(&"d1".to_string(), "the quick brown fox jumps")
            .unwrap();
        let after_d1: usize = idx.doc_lengths.values().sum();
        assert_eq!(
            idx.total_tokens, after_d1,
            "total_tokens must equal sum(doc_lengths) after first insert"
        );

        idx.index_document(&"d2".to_string(), "lazy dog sleeps all day")
            .unwrap();
        let after_d2: usize = idx.doc_lengths.values().sum();
        assert_eq!(
            idx.total_tokens, after_d2,
            "total_tokens must equal sum(doc_lengths) after second insert"
        );

        idx.remove_document(&"d1".to_string()).unwrap();
        let after_remove: usize = idx.doc_lengths.values().sum();
        assert_eq!(
            idx.total_tokens, after_remove,
            "total_tokens must equal sum(doc_lengths) after remove"
        );
    }

    // ── Test 7 ───────────────────────────────────────────────────────────────
    /// The O(1) `avg_doc_length` (via `total_tokens / total_docs`) must equal
    /// the manually computed O(n) average at every state.
    #[test]
    fn test_inverted_avgdl_matches_manual() {
        let mut idx = InvertedIndex::default();

        let docs = [
            ("d1", "rust systems programming language"),
            ("d2", "python scripting automation"),
            ("d3", "java enterprise spring boot framework"),
            ("d4", "go concurrency channels goroutines"),
        ];

        for (id, text) in &docs {
            idx.index_document(&id.to_string(), text).unwrap();

            let manual_avg = if idx.total_docs == 0 {
                0.0f32
            } else {
                idx.doc_lengths.values().sum::<usize>() as f32 / idx.total_docs as f32
            };
            assert!(
                (idx.avg_doc_length() - manual_avg).abs() < 1e-4,
                "avgdl mismatch after inserting {id}: O(1)={} manual={}",
                idx.avg_doc_length(),
                manual_avg
            );
        }

        // Also verify after a remove.
        idx.remove_document(&"d2".to_string()).unwrap();
        let manual_avg = idx.doc_lengths.values().sum::<usize>() as f32 / idx.total_docs as f32;
        assert!(
            (idx.avg_doc_length() - manual_avg).abs() < 1e-4,
            "avgdl mismatch after remove: O(1)={} manual={}",
            idx.avg_doc_length(),
            manual_avg
        );
    }

    // ── Test 8 ───────────────────────────────────────────────────────────────
    /// Verify the new `min_max_normalize(&[f32]) -> Vec<f32>` API covers all
    /// edge cases: basic, all-equal (EPSILON fast-path), and empty input.
    #[test]
    fn test_min_max_normalize_new_api() {
        use crate::hybrid::min_max_normalize;

        // Basic normalization.
        let basic = min_max_normalize(&[2.0f32, 4.0, 6.0]);
        assert_eq!(basic.len(), 3);
        assert!((basic[0] - 0.0).abs() < 1e-6, "min must map to 0.0");
        assert!((basic[1] - 0.5).abs() < 1e-6, "midpoint must map to 0.5");
        assert!((basic[2] - 1.0).abs() < 1e-6, "max must map to 1.0");

        // All-equal — EPSILON fast-path returns vec of 1.0.
        let equal = min_max_normalize(&[3.0f32, 3.0, 3.0, 3.0]);
        assert!(
            equal.iter().all(|&s| (s - 1.0).abs() < 1e-6),
            "all-equal input must produce all-1.0 output"
        );

        // Empty input.
        let empty = min_max_normalize(&[]);
        assert!(empty.is_empty(), "empty input must produce empty output");
    }
}
