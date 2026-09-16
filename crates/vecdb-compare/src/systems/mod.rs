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
