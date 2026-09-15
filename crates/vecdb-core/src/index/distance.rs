use crate::types::DistanceMetric;

/// Cosine similarity between two vectors.
/// Returns 1.0 for identical direction, -1.0 for opposite, 0.0 for orthogonal.
/// Returns 0.0 when either vector is zero (avoids divide-by-zero).
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Cosine distance = 1 - cosine_similarity. Range [0, 2]; 0 = identical direction.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Euclidean (L2) distance between two vectors.
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Dot product of two vectors.
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot-product distance = 1 - dot_product(a, b).
/// NOTE: Only meaningful as a proper distance metric for unit-normalized vectors.
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - dot_product(a, b)
}

/// Return a unit-normalized copy of `v`.
/// If magnitude < 1e-10 (near-zero vector), returns `v` unchanged to avoid instability.
pub fn normalize(v: &[f32]) -> Vec<f32> {
    let magnitude: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude < 1e-10 {
        return v.to_vec();
    }
    v.iter().map(|x| x / magnitude).collect()
}

/// Dot product using 8-element loop unrolling.
///
/// LLVM auto-vectorizes to AVX2/SSE on x86-64 with `-C target-cpu=native`.
/// Falls back to the scalar implementation for `n < 8`.
///
/// Performance (dim=1536, AVX2): ~2–4× faster than the scalar loop.
// Before: scalar iterator chain, ~1 instruction per float
// After:  8-lane unrolled accumulate, LLVM fuses to VFMADD256 x192 + scalar tail
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 8 {
        return dot_product(&a[..n], &b[..n]);
    }
    let chunks = n / 8;
    let rem = n % 8;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut s4, mut s5, mut s6, mut s7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..chunks {
        let base = i * 8;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
        s4 += a[base + 4] * b[base + 4];
        s5 += a[base + 5] * b[base + 5];
        s6 += a[base + 6] * b[base + 6];
        s7 += a[base + 7] * b[base + 7];
    }
    let mut acc = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7;
    let start = chunks * 8;
    for i in 0..rem {
        acc += a[start + i] * b[start + i];
    }
    acc
}

/// Cosine similarity using 8-element loop unrolling.
///
/// Single-pass computation of dot product, ‖a‖², and ‖b‖² simultaneously.
/// LLVM auto-vectorizes to AVX2/SSE on x86-64 with `-C target-cpu=native`.
/// Falls back to the scalar implementation for `n < 8`.
///
/// Performance (dim=1536, AVX2): ~2–4× faster than the two-pass scalar version.
// Before: three separate iterator passes (dot, mag_a, mag_b)
// After:  single 8-lane pass accumulating all three quantities in parallel
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 8 {
        return cosine_similarity(&a[..n], &b[..n]);
    }
    let chunks = n / 8;
    let rem = n % 8;
    let (mut d0, mut d1, mut d2, mut d3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut d4, mut d5, mut d6, mut d7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut aa0, mut aa1, mut aa2, mut aa3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut aa4, mut aa5, mut aa6, mut aa7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut bb0, mut bb1, mut bb2, mut bb3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut bb4, mut bb5, mut bb6, mut bb7) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..chunks {
        let base = i * 8;
        let (a0, a1, a2, a3) = (a[base], a[base + 1], a[base + 2], a[base + 3]);
        let (a4, a5, a6, a7) = (a[base + 4], a[base + 5], a[base + 6], a[base + 7]);
        let (b0, b1, b2, b3) = (b[base], b[base + 1], b[base + 2], b[base + 3]);
        let (b4, b5, b6, b7) = (b[base + 4], b[base + 5], b[base + 6], b[base + 7]);
        d0 += a0 * b0;
        d1 += a1 * b1;
        d2 += a2 * b2;
        d3 += a3 * b3;
        d4 += a4 * b4;
        d5 += a5 * b5;
        d6 += a6 * b6;
        d7 += a7 * b7;
        aa0 += a0 * a0;
        aa1 += a1 * a1;
        aa2 += a2 * a2;
        aa3 += a3 * a3;
        aa4 += a4 * a4;
        aa5 += a5 * a5;
        aa6 += a6 * a6;
        aa7 += a7 * a7;
        bb0 += b0 * b0;
        bb1 += b1 * b1;
        bb2 += b2 * b2;
        bb3 += b3 * b3;
        bb4 += b4 * b4;
        bb5 += b5 * b5;
        bb6 += b6 * b6;
        bb7 += b7 * b7;
    }
    let mut dot = d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7;
    let mut mag_a_sq = aa0 + aa1 + aa2 + aa3 + aa4 + aa5 + aa6 + aa7;
    let mut mag_b_sq = bb0 + bb1 + bb2 + bb3 + bb4 + bb5 + bb6 + bb7;
    let start = chunks * 8;
    for i in 0..rem {
        let ai = a[start + i];
        let bi = b[start + i];
        dot += ai * bi;
        mag_a_sq += ai * ai;
        mag_b_sq += bi * bi;
    }
    let mag_a = mag_a_sq.sqrt();
    let mag_b = mag_b_sq.sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Dispatch distance computation based on `metric`.
///
/// Uses SIMD-unrolled kernels for Cosine and DotProduct; scalar for Euclidean.
pub fn compute_distance(a: &[f32], b: &[f32], metric: &DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - cosine_similarity_simd(a, b),
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::DotProduct => 1.0 - dot_product_simd(a, b),
    }
}

/// Convert a `compute_distance` value into a similarity score in which higher is
/// better (the ranking/hydration convention used across all index backends).
pub fn to_score(dist: f32, metric: &DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Cosine => 1.0 - dist,
        DistanceMetric::Euclidean => 1.0 / (1.0 + dist),
        DistanceMetric::DotProduct => 1.0 - dist,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0f32, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < f32::EPSILON * 10.0);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < f32::EPSILON * 10.0);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0f32, 0.0];
        let b = vec![-1.0f32, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < f32::EPSILON * 10.0);
    }

    #[test]
    fn test_euclidean_zero() {
        let v = vec![3.0f32, 1.0, -2.0];
        assert_eq!(euclidean_distance(&v, &v), 0.0);
    }

    #[test]
    fn test_euclidean_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = vec![3.0f32, 4.0];
        let n = normalize(&v);
        let mag: f32 = n.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_vector_cosine() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let other = vec![1.0f32, 2.0, 3.0];
        // Must not panic, must return 0.0
        let sim = cosine_similarity(&zero, &other);
        assert_eq!(sim, 0.0);
    }
}
