use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::DateTime;
use r2d2::Pool;
use r2d2_sqlite::SqliteConnectionManager;
use rusqlite::params;

use crate::errors::{Result, VecDbError};
use crate::types::{CollectionConfig, VectorRecord};

// Schema: DDL only — PRAGMAs are applied per-connection via the pool's `with_init`.
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS vectors (
    id          TEXT PRIMARY KEY NOT NULL,
    mmap_index  INTEGER NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    text        TEXT,
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    deleted     INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_vectors_deleted ON vectors(deleted);
CREATE INDEX IF NOT EXISTS idx_vectors_created_at ON vectors(created_at);

CREATE TABLE IF NOT EXISTS collections (
    name        TEXT PRIMARY KEY NOT NULL,
    config      TEXT NOT NULL,
    created_at  INTEGER NOT NULL
);
"#;

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

fn ts_to_dt(secs: i64) -> DateTime<chrono::Utc> {
    DateTime::from_timestamp(secs, 0).unwrap_or_else(chrono::Utc::now)
}

fn build_pool(path: &Path, max_size: u32) -> Result<Pool<SqliteConnectionManager>> {
    let manager = SqliteConnectionManager::file(path).with_init(|conn| {
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; \
             PRAGMA busy_timeout=5000; \
             PRAGMA synchronous=NORMAL; \
             PRAGMA foreign_keys=ON;",
        )
    });
    r2d2::Pool::builder()
        .max_size(max_size)
        .build(manager)
        .map_err(|e| VecDbError::StorageError(format!("connection pool: {e}")))
}

#[derive(Clone)]
pub struct MetadataStore {
    pool: Pool<SqliteConnectionManager>,
    pub path: PathBuf,
}

impl MetadataStore {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_pool_size(path, 8)
    }

    pub fn open_with_pool_size(path: &Path, max_size: u32) -> Result<Self> {
        let pool = build_pool(path, max_size)?;
        let conn = pool
            .get()
            .map_err(|e| VecDbError::StorageError(format!("pool init: {e}")))?;
        conn.execute_batch(SCHEMA)?;
        drop(conn);
        Ok(Self {
            pool,
            path: path.to_path_buf(),
        })
    }

    fn conn(&self) -> Result<r2d2::PooledConnection<SqliteConnectionManager>> {
        self.pool
            .get()
            .map_err(|e| VecDbError::StorageError(format!("pool get: {e}")))
    }

    pub fn upsert(&self, id: &str, mmap_index: usize, record: &VectorRecord) -> Result<()> {
        let payload = serde_json::to_string(&record.payload)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        let now = now_unix();
        self.conn()?.execute(
            "INSERT OR REPLACE INTO vectors \
             (id, mmap_index, payload, text, created_at, updated_at, deleted) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            params![id, mmap_index as i64, payload, record.text, now, now],
        )?;
        Ok(())
    }

    /// Insert/replace many records in a single transaction (bulk load).
    pub fn upsert_batch(&self, items: &[(String, usize, &VectorRecord)]) -> Result<()> {
        let mut conn = self.conn()?;
        let now = now_unix();
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO vectors \
                 (id, mmap_index, payload, text, created_at, updated_at, deleted) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, 0)",
            )?;
            for (id, mmap_index, record) in items {
                let payload = serde_json::to_string(&record.payload)
                    .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
                stmt.execute(params![id, *mmap_index as i64, payload, record.text, now, now])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn get(&self, id: &str) -> Result<(usize, VectorRecord)> {
        let conn = self.conn()?;
        let row = conn.query_row(
            "SELECT id, mmap_index, payload, text, created_at, updated_at \
             FROM vectors WHERE id = ?1 AND deleted = 0",
            params![id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        );

        match row {
            Ok((rid, mmap_idx, payload_str, text, created_ts, updated_ts)) => {
                let payload = serde_json::from_str(&payload_str)
                    .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
                let record = VectorRecord {
                    id: rid,
                    vector: vec![],
                    payload,
                    text,
                    created_at: ts_to_dt(created_ts),
                    updated_at: ts_to_dt(updated_ts),
                };
                Ok((mmap_idx as usize, record))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(VecDbError::NotFound { id: id.to_string() })
            }
            Err(e) => Err(VecDbError::SqliteError(e)),
        }
    }

    pub fn delete(&self, id: &str) -> Result<()> {
        let now = now_unix();
        let rows = self.conn()?.execute(
            "UPDATE vectors SET deleted = 1, updated_at = ?1 WHERE id = ?2 AND deleted = 0",
            params![now, id],
        )?;
        if rows == 0 {
            return Err(VecDbError::NotFound { id: id.to_string() });
        }
        Ok(())
    }

    pub fn list_active(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, mmap_index FROM vectors WHERE deleted = 0 ORDER BY mmap_index ASC",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)? as usize))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    pub fn count_active(&self) -> Result<usize> {
        let n: i64 = self.conn()?.query_row(
            "SELECT COUNT(*) FROM vectors WHERE deleted = 0",
            [],
            |row| row.get(0),
        )?;
        Ok(n as usize)
    }

    pub fn save_collection(&self, config: &CollectionConfig) -> Result<()> {
        let json = serde_json::to_string(config)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        let now = now_unix();
        self.conn()?.execute(
            "INSERT OR REPLACE INTO collections (name, config, created_at) VALUES (?1, ?2, ?3)",
            params![config.name, json, now],
        )?;
        Ok(())
    }

    pub fn load_collection(&self, name: &str) -> Result<CollectionConfig> {
        let conn = self.conn()?;
        match conn.query_row(
            "SELECT config FROM collections WHERE name = ?1",
            params![name],
            |row| row.get::<_, String>(0),
        ) {
            Ok(json) => serde_json::from_str(&json)
                .map_err(|e| VecDbError::SerializationError(e.to_string())),
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                Err(VecDbError::CollectionNotFound(name.to_string()))
            }
            Err(e) => Err(VecDbError::SqliteError(e)),
        }
    }

    /// Ensure a SQLite expression index exists for each declared payload field,
    /// so selective filters on them are served from the index. Idempotent.
    pub fn ensure_payload_indexes(&self, fields: &[String]) -> Result<()> {
        let conn = self.conn()?;
        for f in fields {
            if !is_safe_json_path(f) {
                continue;
            }
            let idx_name = format!("idx_pf_{}", f.replace('.', "_"));
            let sql = format!(
                "CREATE INDEX IF NOT EXISTS {idx_name} ON vectors (json_extract(payload, '$.{f}'))"
            );
            conn.execute(&sql, [])?;
        }
        Ok(())
    }

    /// Return ids of all active records whose payload matches `filter`.
    pub fn filter_ids(
        &self,
        filter: &serde_json::Value,
        indexed: &[String],
    ) -> Result<Vec<crate::types::VectorId>> {
        Ok(self
            .filter_ids_indexed(filter, indexed)?
            .into_iter()
            .map(|(id, _)| id)
            .collect())
    }

    /// Like [`filter_ids`], but also returns each match's `mmap_index` so the
    /// caller can fetch its vector directly for a filtered brute-force scan
    /// (predicate pushdown: candidates come from the filter, not the ANN index).
    ///
    /// Fast path: the predicate is pushed into SQLite via `json_extract`, so the
    /// engine returns only referenced scalars and we never full-parse the 95%+
    /// of payloads a selective filter excludes. Falls back to a full in-process
    /// scan for filter shapes the pushdown can't represent identically.
    pub fn filter_ids_indexed(
        &self,
        filter: &serde_json::Value,
        indexed: &[String],
    ) -> Result<Vec<(crate::types::VectorId, usize)>> {
        if let Some(rows) = self.filter_pushdown(filter, indexed)? {
            return Ok(rows);
        }
        self.filter_scan(filter)
    }

    /// SQL-pushdown fast path. Returns `None` (caller falls back) when the
    /// filter shape can't be translated with identical semantics.
    fn filter_pushdown(
        &self,
        filter: &serde_json::Value,
        indexed: &[String],
    ) -> Result<Option<Vec<(crate::types::VectorId, usize)>>> {
        use crate::planner::executor::{apply_filter_with, collect_filter_fields};

        let indexed_set: std::collections::HashSet<&str> =
            indexed.iter().map(|s| s.as_str()).collect();

        let mut fields = std::collections::BTreeSet::new();
        if !collect_filter_fields(filter, &mut fields) || filter_contains_bool(filter) {
            return Ok(None);
        }
        // Fields extracted via json_extract for the in-process re-eval, in a
        // stable order (excluding "id", which comes from its own column). Bail
        // if any field name can't be safely embedded in a JSON path.
        let mut extracted: Vec<String> = Vec::new();
        for f in &fields {
            if f == "id" {
                continue;
            }
            if !is_safe_json_path(f) {
                return Ok(None);
            }
            extracted.push(f.clone());
        }

        // Build a SQL predicate that is a *superset* of the exact filter (it may
        // over-match, e.g. non-numeric coerced by CAST, case-insensitive LIKE),
        // so SQLite returns only the candidate rows and the Rust re-eval below
        // trims to the exact set. If we can't build one, fall back.
        let mut params: Vec<rusqlite::types::Value> = Vec::new();
        let Some(where_super) = filter_to_sql_superset(filter, &mut params, &indexed_set) else {
            return Ok(None);
        };

        let mut select = String::from("SELECT id, mmap_index");
        for f in &extracted {
            select.push_str(&format!(", json_extract(payload, '$.{f}')"));
        }
        select.push_str(" FROM vectors WHERE deleted = 0 AND ");
        select.push_str(&where_super);

        let conn = self.conn()?;
        let mut stmt = match conn.prepare(&select) {
            Ok(s) => s,
            Err(_) => return Ok(None), // e.g. JSON1 unavailable → fall back
        };
        let ncols = extracted.len();
        let rows_res = stmt.query_map(rusqlite::params_from_iter(params.iter()), |row| {
            let id: String = row.get(0)?;
            let idx: i64 = row.get(1)?;
            let mut vals: Vec<rusqlite::types::Value> = Vec::with_capacity(ncols);
            for c in 0..ncols {
                vals.push(row.get::<usize, rusqlite::types::Value>(2 + c)?);
            }
            Ok((id, idx as usize, vals))
        });
        let rows: Vec<(String, usize, Vec<rusqlite::types::Value>)> = match rows_res {
            Ok(iter) => iter.collect::<std::result::Result<_, _>>()?,
            Err(_) => return Ok(None),
        };

        let mut out = Vec::new();
        for (id, idx, vals) in rows {
            let resolve = |field: &str| -> Option<serde_json::Value> {
                if field == "id" {
                    return Some(serde_json::Value::String(id.clone()));
                }
                let pos = extracted.iter().position(|f| f == field)?;
                sqlite_to_json(&vals[pos])
            };
            if apply_filter_with(filter, &resolve) {
                out.push((id, idx));
            }
        }
        Ok(Some(out))
    }

    /// Fallback: load active rows and evaluate the filter in-process.
    fn filter_scan(
        &self,
        filter: &serde_json::Value,
    ) -> Result<Vec<(crate::types::VectorId, usize)>> {
        use crate::planner::executor::apply_json_filter;
        use crate::types::SearchResult;

        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT id, mmap_index, payload, text FROM vectors WHERE deleted = 0")?;
        let rows: Vec<(String, i64, String, Option<String>)> = stmt
            .query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?
            .collect::<std::result::Result<_, _>>()?;

        let mut out = Vec::new();
        for (id, idx, payload_str, text) in rows {
            let payload: serde_json::Value = serde_json::from_str(&payload_str)
                .unwrap_or_else(|_| serde_json::Value::Object(Default::default()));
            let fake = SearchResult {
                id: id.clone(),
                score: 0.0,
                dense_score: None,
                sparse_score: None,
                payload,
                text,
            };
            if apply_json_filter(&fake, filter) {
                out.push((id, idx as usize));
            }
        }
        Ok(out)
    }

    pub fn list_collections(&self) -> Result<Vec<CollectionConfig>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare("SELECT config FROM collections ORDER BY created_at ASC")?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        rows.iter()
            .map(|json| {
                serde_json::from_str(json)
                    .map_err(|e| VecDbError::SerializationError(e.to_string()))
            })
            .collect()
    }
}

/// A field name is safe to embed in a `'$.<field>'` JSON path when it contains
/// only characters that can't break out of the quoted path literal.
fn is_safe_json_path(field: &str) -> bool {
    !field.is_empty()
        && field
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

/// `json_extract` maps a JSON boolean to SQLite integer 0/1, which would not
/// compare equal to a `Bool` filter value the way the in-process path does — so
/// any bool in the filter forces the fallback.
fn filter_contains_bool(filter: &serde_json::Value) -> bool {
    match filter {
        serde_json::Value::Bool(_) => true,
        serde_json::Value::Array(a) => a.iter().any(filter_contains_bool),
        serde_json::Value::Object(o) => o.values().any(filter_contains_bool),
        _ => false,
    }
}

/// SQL expression that reads a field: the `id` column, or `json_extract` of a
/// (safe) payload path. `None` if the path is unsafe.
fn field_sql_expr(field: &str) -> Option<String> {
    if field == "id" {
        return Some("id".to_string());
    }
    if !is_safe_json_path(field) {
        return None;
    }
    Some(format!("json_extract(payload, '$.{field}')"))
}

/// Text and numeric forms of a scalar filter value, for the equality superset
/// (`CAST(... AS TEXT) = text OR CAST(... AS REAL) = real`). Numeric form is
/// `Null` when the value isn't numeric.
fn value_forms(v: &serde_json::Value) -> Option<(rusqlite::types::Value, rusqlite::types::Value)> {
    use rusqlite::types::Value as V;
    match v {
        serde_json::Value::String(s) => {
            let real = s.parse::<f64>().ok().map(V::Real).unwrap_or(V::Null);
            Some((V::Text(s.clone()), real))
        }
        serde_json::Value::Number(n) => {
            let f = n.as_f64()?;
            Some((V::Text(n.to_string()), V::Real(f)))
        }
        _ => None,
    }
}

fn value_as_real(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// Natural-typed SQLite param for a JSON number (integer stays integer so it
/// matches an integer-valued expression index).
fn natural_number_param(n: &serde_json::Number) -> Option<rusqlite::types::Value> {
    use rusqlite::types::Value as V;
    if let Some(i) = n.as_i64() {
        Some(V::Integer(i))
    } else {
        n.as_f64().map(V::Real)
    }
}

/// One leaf condition → SQL predicate. For an **indexed** field the predicate
/// references the bare `json_extract(...)` expression (no `CAST`) so SQLite can
/// use the expression index; correctness for that field then assumes a
/// consistent scalar type (the normal contract for an indexed column), and the
/// Rust re-eval still runs. For non-indexed fields it emits a `CAST` superset
/// that never excludes a true match. `None` forces the caller to fall back.
fn leaf_sql_superset(
    field: &str,
    op: &str,
    value: &serde_json::Value,
    params: &mut Vec<rusqlite::types::Value>,
    indexed: &std::collections::HashSet<&str>,
) -> Option<String> {
    use rusqlite::types::Value as V;
    let expr = field_sql_expr(field)?;
    let is_indexed = indexed.contains(field);

    if is_indexed {
        // Index-friendly: bare expression compared to a natural-typed param.
        match op {
            "=" => {
                let p = match value {
                    serde_json::Value::Number(n) => natural_number_param(n)?,
                    serde_json::Value::String(s) => V::Text(s.clone()),
                    _ => return None,
                };
                params.push(p);
                return Some(format!("{expr} = ?"));
            }
            "<" | "<=" | ">" | ">=" => {
                let r = value_as_real(value)?;
                params.push(V::Real(r));
                return Some(format!("{expr} {op} ?"));
            }
            "LIKE" => {
                let s = value.as_str()?;
                params.push(V::Text(s.to_string()));
                return Some(format!("{expr} LIKE ?"));
            }
            _ => return None,
        }
    }

    // Non-indexed: CAST superset (over-matches, trimmed by the Rust re-eval).
    match op {
        "=" => {
            let (text, real) = value_forms(value)?;
            params.push(text);
            params.push(real);
            Some(format!(
                "(CAST({expr} AS TEXT) = ? OR CAST({expr} AS REAL) = ?)"
            ))
        }
        "<" | "<=" | ">" | ">=" => {
            let r = value_as_real(value)?;
            params.push(V::Real(r));
            Some(format!("CAST({expr} AS REAL) {op} ?"))
        }
        "LIKE" => {
            let s = value.as_str()?;
            params.push(V::Text(s.to_string()));
            Some(format!("CAST({expr} AS TEXT) LIKE ?"))
        }
        _ => None, // e.g. "!=" — rarely selective; let the caller fall back
    }
}

/// Translate a filter into a superset SQL predicate (never excludes a true
/// match; the Rust re-eval trims over-matches). `None` if any part is
/// unsupported.
fn filter_to_sql_superset(
    filter: &serde_json::Value,
    params: &mut Vec<rusqlite::types::Value>,
    indexed: &std::collections::HashSet<&str>,
) -> Option<String> {
    match filter {
        serde_json::Value::Array(cs) => {
            let mut parts = Vec::with_capacity(cs.len());
            for c in cs {
                parts.push(filter_to_sql_superset(c, params, indexed)?);
            }
            if parts.is_empty() {
                return None;
            }
            Some(format!("({})", parts.join(" AND ")))
        }
        serde_json::Value::Object(obj) => {
            if obj.contains_key("field") && obj.contains_key("op") && obj.contains_key("value") {
                let field = obj.get("field")?.as_str()?;
                let op = obj.get("op")?.as_str()?;
                let value = obj.get("value")?;
                leaf_sql_superset(field, op, value, params, indexed)
            } else {
                if obj.is_empty() {
                    return None;
                }
                let mut parts = Vec::with_capacity(obj.len());
                for (k, v) in obj {
                    parts.push(leaf_sql_superset(k, "=", v, params, indexed)?);
                }
                Some(format!("({})", parts.join(" AND ")))
            }
        }
        _ => None,
    }
}

/// Convert a scalar returned by `json_extract` back to a JSON value, matching
/// what the in-process resolver would produce (nulls → absent field).
fn sqlite_to_json(v: &rusqlite::types::Value) -> Option<serde_json::Value> {
    use rusqlite::types::Value as V;
    match v {
        V::Null => None,
        V::Integer(i) => Some(serde_json::Value::Number((*i).into())),
        V::Real(f) => serde_json::Number::from_f64(*f).map(serde_json::Value::Number),
        V::Text(s) => Some(serde_json::Value::String(s.clone())),
        V::Blob(_) => None,
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use chrono::Utc;
    use serde_json::json;
    use std::sync::Arc;
    use tempfile::tempdir;

    use crate::types::VectorRecord;

    fn dummy_record(id: &str) -> VectorRecord {
        let now = Utc::now();
        VectorRecord {
            id: id.to_string(),
            vector: vec![],
            payload: json!({ "id": id }),
            text: None,
            created_at: now,
            updated_at: now,
        }
    }

    // Test 1 — four threads read concurrently; all must succeed and see 5 records.
    #[test]
    fn test_pool_concurrent_reads() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("conc_reads.db");

        let store = MetadataStore::open(&path).unwrap();
        for i in 0..5usize {
            store
                .upsert(&format!("r{i}"), i, &dummy_record(&format!("r{i}")))
                .unwrap();
        }

        let store = Arc::new(store);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                let active = s.list_active().expect("list_active failed");
                assert_eq!(active.len(), 5, "expected 5 active records");
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
    }

    // Test 2 — two writers and two readers run concurrently; no errors expected.
    #[test]
    fn test_pool_concurrent_upsert_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("conc_rw.db");

        let store = Arc::new(MetadataStore::open(&path).unwrap());
        let mut handles = Vec::new();

        // Two writer threads — each upserts 10 records with unique ids.
        for t in 0..2usize {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                for i in 0..10usize {
                    let id = format!("t{t}r{i}");
                    s.upsert(&id, t * 10 + i, &dummy_record(&id))
                        .expect("upsert failed");
                }
            }));
        }

        // Two reader threads — each calls count_active 10 times.
        for _ in 0..2usize {
            let s = Arc::clone(&store);
            handles.push(std::thread::spawn(move || {
                for _ in 0..10 {
                    let _ = s.count_active().expect("count_active failed");
                }
            }));
        }

        for h in handles {
            h.join().expect("thread panicked");
        }

        // All 20 records must be present after the writers finish.
        assert_eq!(store.count_active().unwrap(), 20);
    }

    // Test 3 — pool with max_size(2) correctly recycles connections over 20 ops.
    #[test]
    fn test_metadata_survives_pool_exhaustion() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pool2.db");

        let store = MetadataStore::open_with_pool_size(&path, 2).unwrap();

        // 20 sequential operations — each acquires and releases a connection.
        for i in 0..20usize {
            let id = format!("p{i}");
            store
                .upsert(&id, i, &dummy_record(&id))
                .unwrap_or_else(|e| panic!("op {i} failed: {e}"));
        }

        assert_eq!(store.count_active().unwrap(), 20);
    }
}
