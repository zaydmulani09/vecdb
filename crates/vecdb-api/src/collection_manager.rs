use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::Mutex;
use vecdb_core::{errors::Result, storage::Storage, CollectionConfig, VecDbError};

pub struct CollectionManager {
    collections: HashMap<String, Arc<Mutex<Storage>>>,
    data_dir: PathBuf,
}

impl CollectionManager {
    pub fn new(data_dir: PathBuf) -> Self {
        Self {
            collections: HashMap::new(),
            data_dir,
        }
    }

    /// Scan `data_dir` for `*.db` files, open each as a Storage, and load them
    /// into the map.  Collections that fail to open are logged and skipped.
    pub async fn load_existing(&mut self) -> Result<usize> {
        let entries = std::fs::read_dir(&self.data_dir)
            .map_err(|e| VecDbError::StorageError(format!("read_dir failed: {e}")))?;

        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "db").unwrap_or(false) {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    let name = stem.to_string();
                    match Storage::open(&self.data_dir, &name) {
                        Ok(storage) => {
                            tracing::info!("Loaded collection '{}'", name);
                            self.collections.insert(name, Arc::new(Mutex::new(storage)));
                            count += 1;
                        }
                        Err(e) => {
                            tracing::error!("Failed to load collection '{}': {}", name, e);
                        }
                    }
                }
            }
        }
        Ok(count)
    }

    /// Create a new collection.  Returns `CollectionAlreadyExists` if the name
    /// is already registered.
    pub async fn create(&mut self, config: &CollectionConfig) -> Result<()> {
        if self.collections.contains_key(&config.name) {
            return Err(VecDbError::CollectionAlreadyExists(config.name.clone()));
        }
        let storage = Storage::create(&self.data_dir, config)?;
        self.collections
            .insert(config.name.clone(), Arc::new(Mutex::new(storage)));
        Ok(())
    }

    /// Return a cloned `Arc` for the named collection, or `None` if not found.
    pub fn get(&self, name: &str) -> Option<Arc<Mutex<Storage>>> {
        self.collections.get(name).cloned()
    }

    /// Remove the collection from the map and delete its files from disk.
    pub async fn delete(&mut self, name: &str) -> Result<()> {
        if self.collections.remove(name).is_none() {
            return Err(VecDbError::CollectionNotFound(name.to_string()));
        }

        for suffix in &[
            ".db",
            ".wal",
            ".vectors",
            ".hnsw.json",
            ".ivf.json",
            ".sq.json",
            ".sparse.json",
        ] {
            let path = self.data_dir.join(format!("{name}{suffix}"));
            if path.exists() {
                let _ = std::fs::remove_file(&path);
            }
        }
        tracing::info!("Deleted collection '{}'", name);
        Ok(())
    }

    /// Sorted list of all collection names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.collections.keys().cloned().collect();
        names.sort();
        names
    }

    pub fn len(&self) -> usize {
        self.collections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.collections.is_empty()
    }

    /// Cloned Arcs for every collection (for health aggregation, etc.).
    pub fn all_arcs(&self) -> Vec<Arc<Mutex<Storage>>> {
        self.collections.values().cloned().collect()
    }

    /// Direct sync insertion used by test helpers.
    pub fn insert(&mut self, name: impl Into<String>, storage: Storage) {
        self.collections
            .insert(name.into(), Arc::new(Mutex::new(storage)));
    }
}
