pub mod backend;
pub mod distance;
pub mod ivf;
pub mod scalar;

pub use backend::{HnswConfig, HnswIndex, IndexBackend};
pub use distance::{
    compute_distance, cosine_similarity, cosine_similarity_simd, dot_product, dot_product_simd,
    euclidean_distance, normalize,
};
pub use ivf::IvfIndex;
pub use scalar::{ScalarQuantizedIndex, ScalarQuantizer};

use crate::errors::Result;
use crate::types::{CollectionConfig, IndexType, Quantization, Vector, VectorId};

// ──────────────────────────────────────────────
// AnyIndex — dispatch enum over all backends
// ──────────────────────────────────────────────

pub enum AnyIndex {
    Hnsw(HnswIndex),
    Ivf(IvfIndex),
    ScalarQuantized(ScalarQuantizedIndex),
}

impl IndexBackend for AnyIndex {
    fn build(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.build(vectors),
            AnyIndex::Ivf(i) => i.build(vectors),
            AnyIndex::ScalarQuantized(s) => s.build(vectors),
        }
    }

    fn insert(&mut self, id: VectorId, vector: Vector) -> Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.insert(id, vector),
            AnyIndex::Ivf(i) => i.insert(id, vector),
            AnyIndex::ScalarQuantized(s) => s.insert(id, vector),
        }
    }

    fn search(&self, query: &Vector, k: usize) -> Result<Vec<(VectorId, f32)>> {
        match self {
            AnyIndex::Hnsw(h) => h.search(query, k),
            AnyIndex::Ivf(i) => i.search(query, k),
            AnyIndex::ScalarQuantized(s) => s.search(query, k),
        }
    }

    fn delete(&mut self, id: &VectorId) -> Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.delete(id),
            AnyIndex::Ivf(i) => i.delete(id),
            AnyIndex::ScalarQuantized(s) => s.delete(id),
        }
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.save(path),
            AnyIndex::Ivf(i) => i.save(path),
            AnyIndex::ScalarQuantized(s) => s.save(path),
        }
    }

    fn load_from(path: &std::path::Path, config: &CollectionConfig) -> Result<Box<dyn IndexBackend>>
    where
        Self: Sized,
    {
        // Dispatch based on file extension convention
        if path.extension().map(|e| e == "json").unwrap_or(false) {
            let stem = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default();
            if stem.ends_with(".ivf.json") {
                return IvfIndex::load_from(path, config);
            }
            if stem.ends_with(".sq.json") {
                return ScalarQuantizedIndex::load_from(path, config);
            }
        }
        HnswIndex::load_from(path, config)
    }

    fn len(&self) -> usize {
        match self {
            AnyIndex::Hnsw(h) => h.len(),
            AnyIndex::Ivf(i) => i.len(),
            AnyIndex::ScalarQuantized(s) => s.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            AnyIndex::Hnsw(h) => h.is_empty(),
            AnyIndex::Ivf(i) => i.is_empty(),
            AnyIndex::ScalarQuantized(s) => s.is_empty(),
        }
    }

    fn index_type(&self) -> IndexType {
        match self {
            AnyIndex::Hnsw(h) => h.index_type(),
            AnyIndex::Ivf(i) => i.index_type(),
            AnyIndex::ScalarQuantized(s) => s.index_type(),
        }
    }

    fn rebuild(&mut self, vectors: Vec<(VectorId, Vector)>) -> Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.rebuild(vectors),
            AnyIndex::Ivf(i) => i.rebuild(vectors),
            AnyIndex::ScalarQuantized(s) => s.rebuild(vectors),
        }
    }
}

// Safety: HnswIndex and IvfIndex are both Send+Sync, so AnyIndex is too.
unsafe impl Send for AnyIndex {}
unsafe impl Sync for AnyIndex {}

impl AnyIndex {
    /// Construct the correct variant for the given collection config.
    ///
    /// `Quantization::ScalarInt8` selects the int8 scalar-quantized flat index
    /// regardless of `index_type` (HNSW-over-int8 is not yet implemented); the
    /// full-precision on-disk vectors remain authoritative.
    pub fn from_config(config: &CollectionConfig) -> Self {
        if config.quantization == Quantization::ScalarInt8 {
            return AnyIndex::ScalarQuantized(ScalarQuantizedIndex::new(config));
        }
        match config.index_type {
            IndexType::IVF => AnyIndex::Ivf(IvfIndex::new(config)),
            IndexType::HNSW => AnyIndex::Hnsw(HnswIndex::from_collection_config(config)),
        }
    }

    /// Load a saved index from `data_dir` or construct a fresh one.
    /// Returns `(index, was_loaded_from_disk)`.
    pub fn load_or_create(
        data_dir: &std::path::Path,
        name: &str,
        config: &CollectionConfig,
    ) -> (Self, bool) {
        if config.quantization == Quantization::ScalarInt8 {
            let path = data_dir.join(format!("{name}.sq.json"));
            if path.exists() {
                match ScalarQuantizedIndex::load_file(&path) {
                    Ok(idx) => return (AnyIndex::ScalarQuantized(idx), true),
                    Err(_) => return (AnyIndex::ScalarQuantized(ScalarQuantizedIndex::new(config)), false),
                }
            }
            return (AnyIndex::ScalarQuantized(ScalarQuantizedIndex::new(config)), false);
        }
        match config.index_type {
            IndexType::IVF => {
                let path = data_dir.join(format!("{name}.ivf.json"));
                if path.exists() {
                    match IvfIndex::load_file(&path) {
                        Ok(idx) => (AnyIndex::Ivf(idx), true),
                        Err(_) => (AnyIndex::Ivf(IvfIndex::new(config)), false),
                    }
                } else {
                    (AnyIndex::Ivf(IvfIndex::new(config)), false)
                }
            }
            IndexType::HNSW => {
                let path = data_dir.join(format!("{name}.hnsw.json"));
                if path.exists() {
                    match HnswIndex::load_file(&path, config) {
                        Ok(idx) => (AnyIndex::Hnsw(idx), true),
                        Err(_) => (
                            AnyIndex::Hnsw(HnswIndex::from_collection_config(config)),
                            false,
                        ),
                    }
                } else {
                    (
                        AnyIndex::Hnsw(HnswIndex::from_collection_config(config)),
                        false,
                    )
                }
            }
        }
    }

    /// Persist the index to the appropriate file under `data_dir`.
    pub fn save_for_collection(
        &self,
        data_dir: &std::path::Path,
        name: &str,
    ) -> crate::errors::Result<()> {
        match self {
            AnyIndex::Hnsw(h) => h.save(&data_dir.join(format!("{name}.hnsw.json"))),
            AnyIndex::Ivf(i) => i.save(&data_dir.join(format!("{name}.ivf.json"))),
            AnyIndex::ScalarQuantized(s) => s.save(&data_dir.join(format!("{name}.sq.json"))),
        }
    }
}

#[cfg(test)]
mod ivf_tests;

#[cfg(test)]
mod perf_tests;
