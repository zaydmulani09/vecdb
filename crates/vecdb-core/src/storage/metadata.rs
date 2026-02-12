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

    /// Return ids of all active records whose payload matches `filter`.
    ///
    /// Loads all active records from SQLite and evaluates the filter in-process.
    /// Used by the pre-filter optimisation in `PlanExecutor`.
    pub fn filter_ids(&self, filter: &serde_json::Value) -> Result<Vec<crate::types::VectorId>> {
        use crate::planner::executor::apply_json_filter;
        use crate::types::SearchResult;

        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT id, payload, text FROM vectors WHERE deleted = 0")?;

        let rows: Vec<(String, String, Option<String>)> = stmt
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))?
            .collect::<std::result::Result<_, _>>()?;

        let mut ids = Vec::new();
        for (id, payload_str, text) in rows {
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
                ids.push(id);
            }
        }
        Ok(ids)
    }

    pub fn list_collections(&self) -> Result<Vec<CollectionConfig>> {
        let conn = self.conn()?;
        let mut stmt =
            conn.prepare("SELECT config FROM collections ORDER BY created_at ASC")?;
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
