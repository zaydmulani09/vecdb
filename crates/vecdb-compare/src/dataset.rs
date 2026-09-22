//! SIFT (.fvecs/.ivecs) loading, ground truth, and recall.

use std::io::Read;
use std::path::Path;

use rayon::prelude::*;

pub fn read_fvecs(path: &Path, max: Option<usize>) -> std::io::Result<(Vec<Vec<f32>>, usize)> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut dim = 0usize;
    let mut off = 0usize;
    while off < buf.len() {
        if let Some(m) = max {
            if out.len() >= m {
                break;
            }
        }
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        dim = d;
        off += 4;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(f32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok((out, dim))
}

pub fn read_ivecs(path: &Path, max: Option<usize>) -> std::io::Result<Vec<Vec<u32>>> {
    let mut buf = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut buf)?;
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        if let Some(m) = max {
            if out.len() >= m {
                break;
            }
        }
        let d = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut v = Vec::with_capacity(d);
        for _ in 0..d {
            v.push(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.push(v);
    }
    Ok(out)
}

/// Exact top-k L2 ground truth over `base` for each query (ids are base
/// indices as strings). Parallel; used when benchmarking a subset, where the
/// dataset's own ground truth (indices into the full 1M) doesn't apply.
pub fn exact_ground_truth(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<String>> {
    queries
        .par_iter()
        .map(|q| {
            let mut scored: Vec<(usize, f32)> = base
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i, d)
                })
                .collect();
            scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored
                .into_iter()
                .take(k)
                .map(|(i, _)| i.to_string())
                .collect()
        })
        .collect()
}

/// recall@k = mean over queries of |returned_topk ∩ truth_topk| / k.
pub fn recall_at_k(results: &[Vec<String>], truth: &[Vec<String>], k: usize) -> f64 {
    let mut sum = 0.0;
    for (r, t) in results.iter().zip(truth) {
        let set: std::collections::HashSet<&String> = t.iter().take(k).collect();
        let hit = r.iter().take(k).filter(|id| set.contains(id)).count();
        sum += hit as f64 / k as f64;
    }
    sum / results.len().max(1) as f64
}

pub fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() - 1) as f64).round() as usize;
    sorted_us[idx]
}
