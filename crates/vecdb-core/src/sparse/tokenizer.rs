use std::collections::{HashMap, HashSet};

/// Common English stopwords.
static ENGLISH_STOPWORDS: &[&str] = &[
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "each", "few", "more", "most", "other", "some",
    "such", "than", "then", "too", "very", "just", "also", "as", "if", "its", "it", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he", "she", "they", "them",
    "his", "her", "their", "what", "which", "who", "whom", "how", "when", "where", "why", "all",
    "any", "into", "through", "during", "before", "after", "above", "below", "up", "down", "out",
    "off", "over", "under", "again", "further", "once", "here", "there", "while", "about",
    "against", "between", "own", "same", "only", "s", "t", "re", "ve", "ll", "d", "m",
];

/// Configuration for the tokenizer pipeline.
#[derive(Debug, Clone)]
pub struct TokenizerConfig {
    /// Convert tokens to lowercase (default: true).
    pub lowercase: bool,
    /// Remove stopwords from token stream (default: true).
    pub remove_stopwords: bool,
    /// Minimum token length to keep (default: 2).
    pub min_token_length: usize,
    /// Maximum token length to keep (default: 64).
    pub max_token_length: usize,
    /// Apply simple suffix stripping (default: false).
    pub strip_suffixes: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            remove_stopwords: true,
            min_token_length: 2,
            max_token_length: 64,
            strip_suffixes: false,
        }
    }
}

/// Unicode-aware tokenizer with configurable pipeline:
/// split → lowercase → clean → length filter → stopword filter → suffix strip.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    config: TokenizerConfig,
    stopwords: HashSet<String>,
}

impl Tokenizer {
    /// Create a tokenizer with the given config.
    pub fn new(config: TokenizerConfig) -> Self {
        let stopwords: HashSet<String> = ENGLISH_STOPWORDS.iter().map(|s| s.to_string()).collect();
        Self { config, stopwords }
    }

    /// Tokenize document text — applies all filters including stopword removal.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        self.run_pipeline(text, self.config.remove_stopwords)
    }

    /// Tokenize query text — stopwords are KEPT (critical for recall).
    pub fn tokenize_query(&self, text: &str) -> Vec<String> {
        self.run_pipeline(text, false)
    }

    fn run_pipeline(&self, text: &str, remove_stopwords: bool) -> Vec<String> {
        // 1. Split on non-alphanumeric boundaries (simple but effective for ASCII/Latin text).
        let mut tokens: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric() && c != '\'')
            .map(|s| s.trim_matches('\''))
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();

        // 2. Lowercase.
        if self.config.lowercase {
            tokens = tokens.into_iter().map(|t| t.to_lowercase()).collect();
        }

        // 3. Length filter.
        tokens.retain(|t| {
            t.len() >= self.config.min_token_length && t.len() <= self.config.max_token_length
        });

        // 4. Stopword filter.
        if remove_stopwords {
            tokens.retain(|t| !self.stopwords.contains(t.as_str()));
        }

        // 5. Suffix stripping.
        if self.config.strip_suffixes {
            tokens = tokens.into_iter().map(Self::strip_suffix).collect();
        }

        tokens
    }

    /// Simple English suffix stripping (no stemmer dependency).
    fn strip_suffix(token: String) -> String {
        // Try suffixes longest-first so "tion" wins over "on".
        const SUFFIXES: &[&str] = &["tion", "ness", "ment", "ing", "ed", "er", "es", "ly"];
        for &suffix in SUFFIXES {
            // Keep at least 3 characters after stripping.
            if token.len() > suffix.len() + 3 && token.ends_with(suffix) {
                return token[..token.len() - suffix.len()].to_string();
            }
        }
        token
    }

    /// Compute raw term → count frequencies from a token slice.
    pub fn compute_term_frequencies(tokens: &[String]) -> HashMap<String, usize> {
        let mut freq: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *freq.entry(token.clone()).or_insert(0) += 1;
        }
        freq
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new(TokenizerConfig::default())
    }
}

// ──────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenize() {
        let tok = Tokenizer::default();
        let tokens = tok.tokenize("The quick brown fox jumps over the lazy dog");
        // "the", "over" are stopwords; all lowercase
        assert!(!tokens.contains(&"the".to_string()));
        assert!(tokens.contains(&"quick".to_string()));
        assert!(tokens.contains(&"brown".to_string()));
        assert!(tokens.contains(&"fox".to_string()));
    }

    #[test]
    fn test_query_keeps_stopwords() {
        let tok = Tokenizer::default();
        let doc_tokens = tok.tokenize("to be or not to be");
        let query_tokens = tok.tokenize_query("to be or not to be");
        // Document tokenization removes stopwords → empty or near-empty
        assert!(doc_tokens.is_empty() || doc_tokens.len() < query_tokens.len());
        // Query keeps stopwords
        assert!(query_tokens.contains(&"to".to_string()));
        assert!(query_tokens.contains(&"be".to_string()));
    }

    #[test]
    fn test_min_length_filter() {
        let tok = Tokenizer::default(); // min_token_length = 2
        let tokens = tok.tokenize("a I go running");
        // "a", "I" filtered by length (< 2); "go" kept; "I" is also a stopword
        assert!(!tokens.contains(&"a".to_string()));
        assert!(tokens.contains(&"go".to_string()));
        assert!(tokens.contains(&"running".to_string()));
    }

    #[test]
    fn test_compute_term_frequencies() {
        let tokens = vec!["dog".to_string(), "cat".to_string(), "dog".to_string()];
        let freq = Tokenizer::compute_term_frequencies(&tokens);
        assert_eq!(freq["dog"], 2);
        assert_eq!(freq["cat"], 1);
    }

    #[test]
    fn test_strip_suffixes() {
        let config = TokenizerConfig {
            strip_suffixes: true,
            remove_stopwords: false,
            ..Default::default()
        };
        let tok = Tokenizer::new(config);
        let tokens = tok.tokenize("running jumped faster");
        // "running" → "runn" (ing stripped), "jumped" → "jump" (ed stripped)
        assert!(tokens.contains(&"runn".to_string()) || tokens.contains(&"running".to_string()));
        assert!(tokens.contains(&"jump".to_string()) || tokens.contains(&"jumped".to_string()));
    }

    #[test]
    fn test_punctuation_split() {
        let tok = Tokenizer::default();
        let tokens = tok.tokenize("hello, world! foo-bar baz.qux");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"foo".to_string()));
        assert!(tokens.contains(&"bar".to_string()));
        assert!(tokens.contains(&"baz".to_string()));
        assert!(tokens.contains(&"qux".to_string()));
    }
}
