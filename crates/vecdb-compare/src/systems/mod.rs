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

/// A container's resident memory in MB via `docker stats` (best effort).
pub fn container_mem_mb(name: &str) -> Option<f64> {
    let out = std::process::Command::new("docker")
        .args(["stats", "--no-stream", "--format", "{{.MemUsage}}", name])
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout);
    // e.g. "153.6MiB / 4GiB"
    let used = s.split('/').next()?.trim();
    parse_size_mb(used)
}

/// A container path's size in MB via `docker exec du` (best effort; `du` may be
/// absent in minimal images, in which case this returns `None`).
pub fn container_disk_mb(name: &str, path: &str) -> Option<f64> {
    let out = std::process::Command::new("docker")
        .args(["exec", name, "du", "-sb", path])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let bytes: f64 = s.split_whitespace().next()?.parse().ok()?;
    Some(bytes / 1e6)
}

fn parse_size_mb(s: &str) -> Option<f64> {
    let s = s.trim();
    let (num, unit) = s.split_at(s.find(|c: char| c.is_alphabetic())?);
    let v: f64 = num.trim().parse().ok()?;
    Some(match unit.trim() {
        "B" => v / 1e6,
        "KiB" | "kB" | "KB" => v / 1e3,
        "MiB" | "MB" => v,
        "GiB" | "GB" => v * 1e3,
        _ => v,
    })
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
