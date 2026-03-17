use anyhow::{Context, Result};
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::rngs::StdRng;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn generate_synthetic(n: usize, dim: usize, seed: u64) -> Vec<(String, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0f32, 1.0f32);

    (0..n)
        .map(|i| {
            let mut v: Vec<f32> = (0..dim).map(|_| dist.sample(&mut rng)).collect();
            let mag = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if mag > 0.0 {
                v.iter_mut().for_each(|x| *x /= mag);
            }
            (format!("syn-{}", i), v)
        })
        .collect()
}

pub fn load_jsonl(path: &Path) -> Result<Vec<(String, Vec<f32>)>> {
    let file =
        std::fs::File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", line_num + 1))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obj: serde_json::Value = serde_json::from_str(line)
            .with_context(|| format!("invalid JSON on line {}", line_num + 1))?;
        let id = obj["id"]
            .as_str()
            .with_context(|| format!("missing 'id' field on line {}", line_num + 1))?
            .to_string();
        let vector: Vec<f32> = obj["vector"]
            .as_array()
            .with_context(|| format!("missing 'vector' field on line {}", line_num + 1))?
            .iter()
            .enumerate()
            .map(|(j, v)| {
                v.as_f64().map(|f| f as f32).with_context(|| {
                    format!(
                        "non-numeric value at vector[{}] on line {}",
                        j,
                        line_num + 1
                    )
                })
            })
            .collect::<Result<Vec<f32>>>()?;
        records.push((id, vector));
    }

    Ok(records)
}
