use crate::types::{CollectionResponse, HealthResponse, SearchResponse, UpsertResponse};

/// Print server health in a key-value table.
pub fn print_health(h: &HealthResponse) {
    print_kv_table(&[
        ("status:", h.status.clone()),
        ("version:", h.version.clone()),
        ("collection:", h.collection.clone()),
        ("vector_count:", h.vector_count.to_string()),
    ]);
}

/// Print collection details in a key-value table.
pub fn print_collection(c: &CollectionResponse) {
    print_kv_table(&[
        ("name:", c.name.clone()),
        ("dimension:", c.dimension.to_string()),
        ("metric:", c.metric.clone()),
        ("index_type:", c.index_type.clone()),
        ("vector_count:", c.vector_count.to_string()),
        ("created_at:", c.created_at.clone()),
    ]);
}

/// Print search results as an aligned table.
///
/// Columns: ID (≤40 chars), SCORE, DENSE, SPARSE.
/// Text and payload print on indented lines below each result row.
pub fn print_search_results(resp: &SearchResponse) {
    println!(
        "{:<40} {:>10} {:>10} {:>10}",
        "ID", "SCORE", "DENSE", "SPARSE"
    );
    println!("{}", "-".repeat(74));
    for r in &resp.results {
        let dense = r
            .dense_score
            .map(|s| format!("{:.4}", s))
            .unwrap_or_else(|| "-".to_string());
        let sparse = r
            .sparse_score
            .map(|s| format!("{:.4}", s))
            .unwrap_or_else(|| "-".to_string());
        let id_display = if r.id.len() > 40 {
            format!("{}…", &r.id[..39])
        } else {
            r.id.clone()
        };
        println!(
            "{:<40} {:>10.4} {:>10} {:>10}",
            id_display, r.score, dense, sparse
        );
        if let Some(text) = &r.text {
            println!("  text:    {}", text);
        }
        let empty_obj = serde_json::Value::Object(Default::default());
        if r.payload != empty_obj {
            println!("  payload: {}", r.payload);
        }
    }
    println!();
    println!("{} results in {}ms", resp.count, resp.time_ms);
}

/// Print upsert summary to stdout; errors to stderr.
pub fn print_upsert_result(resp: &UpsertResponse) {
    println!(
        "inserted: {}  updated: {}  errors: {}  time: {}ms",
        resp.inserted,
        resp.updated,
        resp.errors.len(),
        resp.time_ms
    );
    for e in &resp.errors {
        eprintln!("  error [{}]: {}", e.id, e.error);
    }
}

/// Print aligned key-value pairs. Each key is left-padded to 20 chars.
pub fn print_kv_table(pairs: &[(&str, String)]) {
    for (k, v) in pairs {
        println!("{:<20} {}", k, v);
    }
}
