use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::errors::{Result, VecDbError};
use crate::types::VectorId;

use super::tokenizer::Tokenizer;

// ──────────────────────────────────────────────
// PostingEntry / PostingList
// ──────────────────────────────────────────────

/// One document's entry in a term's posting list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostingEntry {
    pub doc_id: VectorId,
    /// Normalized term frequency (raw_count / doc_length).
    pub term_frequency: f32,
    /// Raw occurrence count in the document.
    pub raw_count: usize,
}

/// A posting list for one term. Entries are kept sorted by `doc_id`
/// so binary search is O(log n).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PostingList {
    pub entries: Vec<PostingEntry>,
}

impl PostingList {
    /// Insert or update an entry for `doc_id`. Maintains sorted order.
    pub fn insert(&mut self, doc_id: VectorId, term_frequency: f32, raw_count: usize) {
        match self.entries.binary_search_by(|e| e.doc_id.cmp(&doc_id)) {
            Ok(pos) => {
                self.entries[pos].term_frequency = term_frequency;
                self.entries[pos].raw_count = raw_count;
            }
            Err(pos) => {
                self.entries.insert(
                    pos,
                    PostingEntry {
                        doc_id,
                        term_frequency,
                        raw_count,
                    },
                );
            }
        }
    }

    /// Remove the entry for `doc_id` (no-op if absent).
    pub fn remove(&mut self, doc_id: &VectorId) {
        if let Ok(pos) = self.entries.binary_search_by(|e| e.doc_id.cmp(doc_id)) {
            self.entries.remove(pos);
        }
    }

    /// Look up an entry by `doc_id`. O(log n).
    pub fn get(&self, doc_id: &VectorId) -> Option<&PostingEntry> {
        self.entries
            .binary_search_by(|e| e.doc_id.cmp(doc_id))
            .ok()
            .map(|pos| &self.entries[pos])
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ──────────────────────────────────────────────
// InvertedIndex
// ──────────────────────────────────────────────

/// BM25-scored inverted index.
///
/// Scoring formula:
/// ```text
/// IDF(t) = ln((N − df + 0.5) / (df + 0.5) + 1)
/// score(q, d) = Σ IDF(tᵢ) × tf × (k1 + 1) / (tf + k1 × (1 − b + b × |d| / avgdl))
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvertedIndex {
    postings: HashMap<String, PostingList>,
    idf: HashMap<String, f32>,
    pub doc_lengths: HashMap<VectorId, usize>,
    /// Term lists per document (used when removing a document).
    doc_terms: HashMap<VectorId, Vec<String>>,
    avg_doc_length: f32,
    /// Total number of indexed documents.
    ///
    /// Renamed from `doc_count` in P23; the serde alias preserves backward-compatible
    /// deserialization of existing JSON files that still use `"doc_count"`.
    #[serde(alias = "doc_count")]
    pub total_docs: usize,
    /// Running sum of all document lengths; enables O(1) `avg_doc_length` recomputation.
    ///
    /// Defaults to 0 for backward compat; `load()` migrates old files that lack this field.
    #[serde(default)]
    pub total_tokens: usize,
    k1: f32,
    b: f32,
    /// Tokenizer is not serialized; reconstructed on load.
    #[serde(skip)]
    tokenizer: Tokenizer,
}

impl InvertedIndex {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            postings: HashMap::new(),
            idf: HashMap::new(),
            doc_lengths: HashMap::new(),
            doc_terms: HashMap::new(),
            avg_doc_length: 0.0,
            total_docs: 0,
            total_tokens: 0,
            k1,
            b,
            tokenizer: Tokenizer::default(),
        }
    }

    /// Compute IDF(t) = ln((N - df + 0.5) / (df + 0.5) + 1).
    fn compute_idf(doc_count: usize, df: usize) -> f32 {
        let n = doc_count as f32;
        let df_f = df as f32;
        ((n - df_f + 0.5) / (df_f + 0.5) + 1.0).ln()
    }

    /// Recompute IDF cache for every term. Called after any structural change.
    fn recompute_idf(&mut self) {
        let doc_count = self.total_docs;
        let postings = &self.postings;
        self.idf = postings
            .iter()
            .map(|(term, list)| (term.clone(), Self::compute_idf(doc_count, list.len())))
            .collect();
    }

    fn recompute_avg_doc_length(&mut self) {
        if self.total_docs == 0 {
            self.avg_doc_length = 0.0;
        } else {
            // Performance: O(1) — total_tokens is maintained as a running sum.
            // Before: O(n) values().sum() over all doc_lengths on every insert/remove
            // After:  direct division; no heap scan regardless of collection size
            self.avg_doc_length = self.total_tokens as f32 / self.total_docs as f32;
        }
    }

    /// Add or update a document. If the document already exists it is first removed.
    pub fn index_document(&mut self, id: &VectorId, text: &str) -> Result<()> {
        // Upsert: remove old entry if updating an existing doc.
        if self.doc_terms.contains_key(id) {
            self.remove_document(id)?;
        }

        let tokens = self.tokenizer.tokenize(text);
        if tokens.is_empty() {
            return Ok(());
        }

        let doc_length = tokens.len();
        let term_freqs = Tokenizer::compute_term_frequencies(&tokens);
        let terms: Vec<String> = term_freqs.keys().cloned().collect();

        for (term, &count) in &term_freqs {
            let tf_normalized = count as f32 / doc_length as f32;
            self.postings
                .entry(term.clone())
                .or_default()
                .insert(id.clone(), tf_normalized, count);
        }

        self.doc_lengths.insert(id.clone(), doc_length);
        self.doc_terms.insert(id.clone(), terms);
        self.total_docs += 1;
        self.total_tokens += doc_length;

        self.recompute_avg_doc_length();
        self.recompute_idf();
        Ok(())
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, id: &VectorId) -> Result<()> {
        let terms = self
            .doc_terms
            .remove(id)
            .ok_or_else(|| VecDbError::NotFound { id: id.clone() })?;

        // Capture doc length before removing it so total_tokens can be decremented.
        let removed_len = self.doc_lengths.get(id).copied().unwrap_or(0);

        for term in &terms {
            if let Some(list) = self.postings.get_mut(term) {
                list.remove(id);
                if list.is_empty() {
                    self.postings.remove(term);
                }
            }
            self.idf.remove(term);
        }

        self.doc_lengths.remove(id);
        if self.total_docs > 0 {
            self.total_docs -= 1;
        }
        self.total_tokens = self.total_tokens.saturating_sub(removed_len);

        self.recompute_avg_doc_length();
        self.recompute_idf();
        Ok(())
    }

    /// Score a pre-tokenized list of query terms against a set of candidate documents.
    ///
    /// If `candidate_ids` is empty all documents in the posting lists are scored.
    /// Results are sorted descending by BM25 score.
    pub fn score(
        &self,
        query_terms: &[String],
        candidate_ids: &[VectorId],
    ) -> Result<Vec<(VectorId, f32)>> {
        let filter: Option<std::collections::HashSet<&VectorId>> = if candidate_ids.is_empty() {
            None
        } else {
            Some(candidate_ids.iter().collect())
        };

        let mut scores: HashMap<VectorId, f32> = HashMap::new();
        let avg_dl = self.avg_doc_length.max(1.0);
        let k1 = self.k1;
        let b = self.b;

        for term in query_terms {
            let Some(list) = self.postings.get(term) else {
                continue;
            };
            let idf = self.idf.get(term).copied().unwrap_or(0.0);

            for entry in &list.entries {
                if let Some(ref set) = filter {
                    if !set.contains(&entry.doc_id) {
                        continue;
                    }
                }
                let doc_len = self.doc_lengths.get(&entry.doc_id).copied().unwrap_or(1) as f32;
                let tf = entry.raw_count as f32;
                let numerator = tf * (k1 + 1.0);
                let denominator = tf + k1 * (1.0 - b + b * (doc_len / avg_dl));
                let term_score = idf * numerator / denominator;
                *scores.entry(entry.doc_id.clone()).or_insert(0.0) += term_score;
            }
        }

        let mut results: Vec<(VectorId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Tokenize `query` then score all matching documents. Returns top `k`.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<(VectorId, f32)>> {
        let terms = self.tokenizer.tokenize_query(query);
        let mut results = self.score(&terms, &[])?;
        results.truncate(k);
        Ok(results)
    }

    /// Tokenize `query_text` and score against `candidate_ids`.
    ///
    /// If `candidate_ids` is empty, scores all matching documents.
    /// Results sorted descending by BM25 score.
    pub fn score_candidates(
        &self,
        query_text: &str,
        candidate_ids: &[VectorId],
    ) -> Result<Vec<(VectorId, f32)>> {
        let terms = self.tokenizer.tokenize_query(query_text);
        self.score(&terms, candidate_ids)
    }

    /// Serialize to JSON and write to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data =
            serde_json::to_vec(self).map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load from a JSON file written by `save`. Reconstructs the tokenizer.
    ///
    /// Migration: if `total_tokens` is 0 (absent in old files) but `doc_lengths` is
    /// non-empty, the running sum is recomputed once via O(n) scan so that subsequent
    /// calls to `recompute_avg_doc_length` are O(1).
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path)?;
        let mut index: Self = serde_json::from_slice(&data)
            .map_err(|e| VecDbError::SerializationError(e.to_string()))?;
        index.tokenizer = Tokenizer::default();
        // One-time migration: backfill total_tokens from doc_lengths when loading old files.
        if index.total_tokens == 0 && !index.doc_lengths.is_empty() {
            index.total_tokens = index.doc_lengths.values().sum();
        }
        Ok(index)
    }

    pub fn doc_count(&self) -> usize {
        self.total_docs
    }

    pub fn avg_doc_length(&self) -> f32 {
        self.avg_doc_length
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new(1.5, 0.75)
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn idx() -> InvertedIndex {
        InvertedIndex::default()
    }

    #[test]
    fn test_index_and_doc_count() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "the quick brown fox")
            .unwrap();
        idx.index_document(&"d2".to_string(), "lazy dog sleeps")
            .unwrap();
        assert_eq!(idx.doc_count(), 2);
    }

    #[test]
    fn test_bm25_relevance_order() {
        let mut idx = idx();
        // d1 mentions "rust" twice; d2 once.
        idx.index_document(&"d1".to_string(), "rust programming rust systems")
            .unwrap();
        idx.index_document(&"d2".to_string(), "python scripting rust web")
            .unwrap();
        idx.index_document(&"d3".to_string(), "java enterprise spring boot")
            .unwrap();

        let results = idx.search("rust", 10).unwrap();
        assert!(!results.is_empty());
        // d3 has no "rust" → must not appear
        assert!(!results.iter().any(|(id, _)| id == "d3"));
        // d1 should score higher than d2
        let d1_score = results.iter().find(|(id, _)| id == "d1").map(|(_, s)| *s);
        let d2_score = results.iter().find(|(id, _)| id == "d2").map(|(_, s)| *s);
        assert!(d1_score.unwrap() > d2_score.unwrap());
    }

    #[test]
    fn test_remove_document() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "hello world rust")
            .unwrap();
        idx.index_document(&"d2".to_string(), "hello python")
            .unwrap();
        idx.remove_document(&"d1".to_string()).unwrap();
        assert_eq!(idx.doc_count(), 1);
        let results = idx.search("rust", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_nonexistent_returns_err() {
        let mut idx = idx();
        let result = idx.remove_document(&"ghost".to_string());
        assert!(matches!(result, Err(VecDbError::NotFound { .. })));
    }

    #[test]
    fn test_upsert_updates_scores() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "apple banana")
            .unwrap();
        // Re-index with new content.
        idx.index_document(&"d1".to_string(), "rust programming language")
            .unwrap();
        assert_eq!(idx.doc_count(), 1);
        // Old terms gone.
        let old = idx.search("banana", 10).unwrap();
        assert!(old.is_empty());
        // New terms present.
        let new = idx.search("rust", 10).unwrap();
        assert!(!new.is_empty());
    }

    #[test]
    fn test_candidate_filter() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "machine learning neural")
            .unwrap();
        idx.index_document(&"d2".to_string(), "machine learning rust")
            .unwrap();
        let terms = vec!["machine".to_string()];
        // Only score d1.
        let results = idx.score(&terms, &["d1".to_string()]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "d1");
    }

    #[test]
    fn test_save_and_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sparse.json");
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "vector database rust storage")
            .unwrap();
        idx.index_document(&"d2".to_string(), "full text search bm25")
            .unwrap();
        idx.save(&path).unwrap();

        let loaded = InvertedIndex::load(&path).unwrap();
        assert_eq!(loaded.doc_count(), 2);
        let results = loaded.search("vector", 5).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "d1");
    }

    #[test]
    fn test_idf_decreases_with_more_docs() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "common term here")
            .unwrap();
        let idf1 = *idx.idf.get("common").unwrap_or(&0.0);

        idx.index_document(&"d2".to_string(), "common term also here")
            .unwrap();
        let idf2 = *idx.idf.get("common").unwrap_or(&0.0);
        // IDF should decrease as "common" appears in more docs.
        assert!(idf2 <= idf1, "idf1={idf1} idf2={idf2}");
    }

    #[test]
    fn test_avg_doc_length_updates() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "one two three")
            .unwrap();
        idx.index_document(&"d2".to_string(), "one two three four five six")
            .unwrap();
        // avg_doc_length must be between 2 and 6 (stopwords filtered)
        assert!(idx.avg_doc_length() > 0.0);
    }

    #[test]
    fn test_empty_query_returns_empty() {
        let mut idx = idx();
        idx.index_document(&"d1".to_string(), "hello world")
            .unwrap();
        let results = idx.search("", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_posting_list_sorted() {
        let mut list = PostingList::default();
        list.insert("z_doc".to_string(), 0.5, 2);
        list.insert("a_doc".to_string(), 0.8, 4);
        list.insert("m_doc".to_string(), 0.3, 1);
        // Verify sorted order.
        let ids: Vec<&str> = list.entries.iter().map(|e| e.doc_id.as_str()).collect();
        assert_eq!(ids, ["a_doc", "m_doc", "z_doc"]);
    }

    #[test]
    fn test_posting_list_get() {
        let mut list = PostingList::default();
        list.insert("alpha".to_string(), 0.5, 2);
        list.insert("beta".to_string(), 0.3, 1);
        assert!(list.get(&"alpha".to_string()).is_some());
        assert!(list.get(&"gamma".to_string()).is_none());
    }
}
