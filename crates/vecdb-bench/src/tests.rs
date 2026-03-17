#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use crate::datasets::{generate_synthetic, load_jsonl};
    use crate::runner::{compute_recall, percentile};

    #[test]
    fn test_synthetic_generates_correct_count() {
        let vecs = generate_synthetic(100, 32, 42);
        assert_eq!(vecs.len(), 100);
        for (_, v) in &vecs {
            assert_eq!(v.len(), 32);
        }
    }

    #[test]
    fn test_synthetic_vectors_are_normalized() {
        let vecs = generate_synthetic(10, 8, 1);
        for (_, v) in &vecs {
            let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((mag - 1.0f32).abs() < 1e-5, "magnitude {mag} is not ~1.0");
        }
    }

    #[test]
    fn test_synthetic_is_reproducible() {
        let v1 = generate_synthetic(10, 4, 99);
        let v2 = generate_synthetic(10, 4, 99);
        for ((_, a), (_, b)) in v1.iter().zip(v2.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < f32::EPSILON, "vectors differ: {x} != {y}");
            }
        }
    }

    #[test]
    fn test_synthetic_ids_are_correct() {
        let vecs = generate_synthetic(10, 4, 42);
        for (i, (id, _)) in vecs.iter().enumerate() {
            assert_eq!(id, &format!("syn-{i}"));
        }
    }

    #[test]
    fn test_load_jsonl_parses_correctly() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"id":"a","vector":[1.0,2.0,3.0]}}"#).unwrap();
        writeln!(file, r#"{{"id":"b","vector":[4.0,5.0,6.0]}}"#).unwrap();
        writeln!(file, r#"{{"id":"c","vector":[7.0,8.0,9.0]}}"#).unwrap();
        let records = load_jsonl(file.path()).unwrap();
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].0, "a");
        assert_eq!(records[0].1, vec![1.0f32, 2.0, 3.0]);
        assert_eq!(records[1].0, "b");
        assert_eq!(records[2].0, "c");
    }

    #[test]
    fn test_load_jsonl_skips_blank_lines() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, r#"{{"id":"a","vector":[1.0,2.0]}}"#).unwrap();
        writeln!(file).unwrap();
        writeln!(file, r#"{{"id":"b","vector":[3.0,4.0]}}"#).unwrap();
        writeln!(file).unwrap();
        writeln!(file, r#"{{"id":"c","vector":[5.0,6.0]}}"#).unwrap();
        let records = load_jsonl(file.path()).unwrap();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_recall_computation() {
        let ground_truth = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let server_results = vec!["a".to_string(), "b".to_string(), "d".to_string()];
        let recall = compute_recall(&ground_truth, &server_results, 3);
        assert!(
            (recall - 2.0 / 3.0).abs() < 1e-9,
            "expected recall=2/3, got {recall}"
        );
    }

    #[test]
    fn test_percentile_computation() {
        let sorted: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentile(&sorted, 50), 50.0);
        assert_eq!(percentile(&sorted, 95), 95.0);
        assert_eq!(percentile(&sorted, 99), 99.0);
    }
}
