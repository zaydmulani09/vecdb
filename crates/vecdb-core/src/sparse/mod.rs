pub mod inverted;
pub mod tokenizer;

pub use inverted::{InvertedIndex, PostingEntry, PostingList};
pub use tokenizer::{Tokenizer, TokenizerConfig};

use std::path::{Path, PathBuf};

use crate::errors::Result;
use crate::types::VectorId;

// ──────────────────────────────────────────────
// SparseIndex facade
// ──────────────────────────────────────────────

/// High-level facade over `InvertedIndex`.
///
/// Owns a file path and delegates all indexing/search operations to the
/// underlying `InvertedIndex`. Persist with `save()`; reconstruct from disk
/// with `open()`.
pub struct SparseIndex {
    pub index: InvertedIndex,
    path: PathBuf,
}

impl SparseIndex {
    /// Create a fresh empty index that will persist to `path`.
    pub fn create(path: &Path) -> Self {
        Self {
            index: InvertedIndex::default(),
            path: path.to_path_buf(),
        }
    }

    /// Open an existing index from `path`, or create empty if file absent.
    pub fn open(path: &Path) -> Result<Self> {
        let index = if path.exists() {
            InvertedIndex::load(path)?
        } else {
            InvertedIndex::default()
        };
        Ok(Self {
            index,
            path: path.to_path_buf(),
        })
    }

    /// Add or update a document.
    pub fn index_document(&mut self, id: &VectorId, text: &str) -> Result<()> {
        self.index.index_document(id, text)
    }

    /// Remove a document. Returns `NotFound` if absent.
    pub fn remove_document(&mut self, id: &VectorId) -> Result<()> {
        self.index.remove_document(id)
    }

    /// Full-text BM25 search. Returns up to `k` results sorted by score desc.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(VectorId, f32)>> {
        self.index.search(query, k)
    }

    /// Persist the index to its file path.
    pub fn save(&self) -> Result<()> {
        self.index.save(&self.path)
    }

    /// Score `query_text` against a specific set of candidate document IDs.
    ///
    /// Tokenizes the query internally. Returns BM25 scores only for the
    /// provided candidates. If `candidate_ids` is empty, all documents are
    /// scored (same as `score_all` but without a `k` truncation).
    pub fn score(
        &self,
        query_text: &str,
        candidate_ids: &[VectorId],
    ) -> Result<Vec<(VectorId, f32)>> {
        self.index.score_candidates(query_text, candidate_ids)
    }

    /// Score `query_text` against all indexed documents. Returns top `k`.
    pub fn score_all(&self, query_text: &str, k: usize) -> Result<Vec<(VectorId, f32)>> {
        self.index.search(query_text, k)
    }

    pub fn doc_count(&self) -> usize {
        self.index.doc_count()
    }
}
