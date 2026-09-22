pub mod chroma;
pub mod pgvector;
pub mod qdrant;
pub mod vecdb;

use std::path::Path;

/// Sum of file sizes under `dir` (recursive), in bytes.
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                total += dir_size(&p);
            } else if let Ok(m) = p.metadata() {
                total += m.len();
            }
        }
    }
    total
}

/// Summed resident memory (MB) of all running processes whose name contains
/// `substr` (case-insensitive) — the native-server equivalent of docker stats.
/// Best effort; approximate (postgres/python spawn multiple processes).
pub fn process_mem_mb(substr: &str) -> Option<f64> {
    use sysinfo::System;
    let mut s = System::new();
    s.refresh_all();
    let needle = substr.to_lowercase();
    let total: u64 = s
        .processes()
        .values()
        .filter(|p| p.name().to_string_lossy().to_lowercase().contains(&needle))
        .map(|p| p.memory())
        .sum();
    if total == 0 {
        None
    } else {
        Some(total as f64 / 1e6)
    }
}

/// On-disk size (MB) of the directory named by env var `var`, if set.
pub fn env_disk_mb(var: &str) -> f64 {
    std::env::var(var)
        .ok()
        .map(|p| dir_size(Path::new(&p)) as f64 / 1e6)
        .unwrap_or(0.0)
}

/// Current process resident memory, in bytes (best effort).
pub fn rss_bytes() -> u64 {
    use sysinfo::{get_current_pid, System};
    let pid = match get_current_pid() {
        Ok(p) => p,
        Err(_) => return 0,
    };
    let mut s = System::new();
    s.refresh_all();
    s.process(pid).map(|p| p.memory()).unwrap_or(0)
}
