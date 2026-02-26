use crate::errors::{Result, VecDbError};
use crate::hybrid::FusionStrategy;
use crate::types::{CollectionConfig, IndexType, SearchRequest};

// ─────────────────────────────────────────────────────────────────
// SortKey
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SortKey {
    /// Sort by the vector similarity score (default).
    Score,
    /// Sort by a metadata field name.
    Field(String),
}

// ─────────────────────────────────────────────────────────────────
// LogicalNode
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum LogicalNode {
    /// Scan the vector index for nearest neighbors.
    VectorScan {
        collection: String,
        k: usize,
        /// Oversampling multiplier (dense fetches k * oversample candidates).
        oversample: usize,
        /// Alpha for hybrid fusion (1.0 = pure dense, 0.0 = pure sparse).
        alpha: f32,
        /// Fusion strategy.
        strategy: FusionStrategy,
    },
    /// Full sparse BM25 scan (no dense pre-filtering).
    SparseScan { collection: String, k: usize },
    /// Apply a scalar metadata filter.
    Filter {
        /// Predicate string (serialised JSON or SQL-like).
        predicate: String,
        input: Box<LogicalNode>,
    },
    /// Join vector candidates with a metadata table.
    Join {
        left: Box<LogicalNode>,
        /// Name of the metadata table / collection to join against.
        right_collection: String,
        /// Join key on the left side (field in vector payload).
        left_key: String,
        /// Join key on the right side (field in metadata).
        right_key: String,
    },
    /// Project (select) specific fields from results.
    Project {
        columns: Vec<String>,
        input: Box<LogicalNode>,
    },
    /// Sort results by a field or score.
    Sort {
        by: SortKey,
        descending: bool,
        input: Box<LogicalNode>,
    },
    /// Limit number of results.
    Limit { n: usize, input: Box<LogicalNode> },
    /// Hybrid scan: dense HNSW + sparse BM25 combined.
    HybridScan {
        collection: String,
        k: usize,
        alpha: f32,
        strategy: FusionStrategy,
        /// Dense oversampling factor before sparse re-scoring.
        oversample: usize,
    },
}

// ─────────────────────────────────────────────────────────────────
// PhysicalPlan
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PhysicalPlan {
    /// The root logical node.
    pub root: LogicalNode,
    /// Which index backend to use.
    pub index_type: IndexType,
    /// Whether to run hybrid retrieval.
    pub use_hybrid: bool,
    /// Whether to apply metadata filter BEFORE the vector scan.
    pub pre_filter: bool,
    /// The scalar predicate to apply (serialised JSON), if any.
    pub filter_predicate: Option<String>,
    /// Estimated cost in arbitrary units.
    pub estimated_cost: f32,
    /// How many dense candidates to fetch (`output_k * oversample`).
    pub candidate_k: usize,
    /// Final k — how many results to return after filtering and sorting.
    pub output_k: usize,
    /// Columns to project onto results. Empty means SELECT * (keep everything).
    pub output_columns: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────
// QueryPlanner
// ─────────────────────────────────────────────────────────────────

pub struct QueryPlanner {
    pub config: CollectionConfig,
}

impl QueryPlanner {
    pub fn new(config: CollectionConfig) -> Self {
        Self { config }
    }

    // ── Cost model ───────────────────────────────────────────────

    /// Estimated cost for HNSW lookup.
    ///
    /// Models O(log N) graph traversal: `ln(N) * k * 0.1`.
    fn estimate_index_cost(&self, vector_count: usize, k: usize) -> f32 {
        (vector_count as f32).ln() * k as f32 * 0.1
    }

    /// Returns `true` if the collection is large enough for HNSW to be
    /// faster than brute force (matches `HnswIndex` auto-build threshold).
    pub fn should_use_hnsw(&self, vector_count: usize) -> bool {
        vector_count >= 100
    }

    /// Returns `true` when pre-filtering (metadata first, then vector scan)
    /// is beneficial.
    ///
    /// - Selectivity < 0.1: filter is highly selective → pre-filter saves work.
    /// - Collection > 100 000 vectors: large collections always benefit.
    pub fn should_pre_filter(&self, estimated_selectivity: f32, vector_count: usize) -> bool {
        estimated_selectivity < 0.1 || vector_count > 100_000
    }

    /// Compute the number of dense candidates to fetch.
    ///
    /// Multipliers:
    /// - hybrid + filter → ×10 (capped)
    /// - hybrid only     → ×5
    /// - filter only     → ×3
    /// - neither         → ×1
    ///
    /// Always at least `output_k`, at most 10 000.
    pub fn compute_candidate_k(
        &self,
        output_k: usize,
        use_hybrid: bool,
        has_filter: bool,
    ) -> usize {
        let mult: usize = match (use_hybrid, has_filter) {
            (true, true) => 10,
            (true, false) => 5,
            (false, true) => 3,
            (false, false) => 1,
        };
        (output_k * mult).min(10_000).max(output_k)
    }

    // ── Plan construction ─────────────────────────────────────────

    /// Build a physical plan for a vector / text search request.
    pub fn plan_search(
        &self,
        request: &SearchRequest,
        vector_count: usize,
    ) -> Result<PhysicalPlan> {
        // Validate: at least one signal required.
        if request.vector.is_none() && request.query_text.is_none() {
            return Err(VecDbError::InvalidQuery(
                "search requires at least one of vector or query_text".into(),
            ));
        }

        let collection = self.config.name.clone();

        // Decide retrieval mode.
        let use_hybrid = request.vector.is_some() && request.query_text.is_some();

        // Index type.
        let index_type = if self.should_use_hnsw(vector_count) {
            IndexType::HNSW
        } else {
            tracing::debug!(
                "collection size {} below HNSW threshold — brute force will be used",
                vector_count
            );
            IndexType::HNSW // IVF not yet implemented; always HNSW
        };

        let has_filter = request.filter.is_some();
        let candidate_k = self.compute_candidate_k(request.k, use_hybrid, has_filter);
        let pre_filter = has_filter && self.should_pre_filter(0.05, vector_count);

        // Serialise the filter predicate for the physical plan and Filter node.
        let filter_predicate: Option<String> = if has_filter {
            request
                .filter
                .as_ref()
                .map(|f| serde_json::to_string(f).unwrap_or_default())
        } else {
            None
        };

        // ── Build logical tree bottom-up ──────────────────────────

        // 1. Scan node.
        let scan: LogicalNode = if use_hybrid {
            LogicalNode::HybridScan {
                collection: collection.clone(),
                k: candidate_k,
                alpha: request.alpha,
                strategy: FusionStrategy::WeightedSum,
                oversample: 5,
            }
        } else if request.vector.is_some() {
            LogicalNode::VectorScan {
                collection: collection.clone(),
                k: candidate_k,
                oversample: 1,
                alpha: 1.0,
                strategy: FusionStrategy::WeightedSum,
            }
        } else {
            // query_text only.
            LogicalNode::SparseScan {
                collection: collection.clone(),
                k: candidate_k,
            }
        };

        // 2. Optional filter wrapper.
        let after_filter: LogicalNode = if let Some(ref pred) = filter_predicate {
            LogicalNode::Filter {
                predicate: pred.clone(),
                input: Box::new(scan),
            }
        } else {
            scan
        };

        // 3. Sort.
        let sorted = LogicalNode::Sort {
            by: SortKey::Score,
            descending: true,
            input: Box::new(after_filter),
        };

        // 4. Limit.
        let root = LogicalNode::Limit {
            n: request.k,
            input: Box::new(sorted),
        };

        let estimated_cost = self.estimate_index_cost(vector_count, candidate_k);

        Ok(PhysicalPlan {
            root,
            index_type,
            use_hybrid,
            pre_filter,
            filter_predicate,
            estimated_cost,
            candidate_k,
            output_k: request.k,
            output_columns: vec![],
        })
    }

    /// Build a physical plan from a SQL string.
    ///
    /// The SQL must contain a `VECTOR_SIM(col, [...]) op threshold` condition.
    /// An optional scalar equality filter (`AND field = value`) is translated
    /// into the plan's `filter_predicate`.
    pub fn plan_sql(&self, sql: &str, vector_count: usize) -> Result<PhysicalPlan> {
        use crate::planner::sql::{AstConverter, SqlParser};
        let stmt = SqlParser::parse(sql)?;
        let converter = AstConverter::new(QueryPlanner::new(self.config.clone()));
        converter.convert(stmt, vector_count)
    }

    /// Return a human-readable description of the physical plan.
    pub fn explain(&self, plan: &PhysicalPlan) -> String {
        let filter_str = plan.filter_predicate.as_deref().unwrap_or("None");
        let tree = Self::format_node(&plan.root, 2);
        format!(
            "PhysicalPlan {{\n  index_type: {:?}\n  use_hybrid: {}\n  pre_filter: {}\n  candidate_k: {}\n  output_k: {}\n  estimated_cost: {:.2}\n  filter: {}\n  logical_tree:\n{}}}\n",
            plan.index_type,
            plan.use_hybrid,
            plan.pre_filter,
            plan.candidate_k,
            plan.output_k,
            plan.estimated_cost,
            filter_str,
            tree,
        )
    }

    /// Recursively format a logical node with indentation.
    ///
    /// Each nesting level adds 2 spaces. `indent=2` gives 4-space indent for
    /// the first tree level (matching the `logical_tree:` header).
    fn format_node(node: &LogicalNode, indent: usize) -> String {
        let pad = " ".repeat(indent * 2);
        let next = indent + 1;
        match node {
            LogicalNode::VectorScan { k, alpha, .. } => {
                format!("{pad}VectorScan(k={k}, alpha={alpha:.2})\n")
            }
            LogicalNode::SparseScan { k, .. } => {
                format!("{pad}SparseScan(k={k})\n")
            }
            LogicalNode::HybridScan {
                k, alpha, strategy, ..
            } => {
                let strat = match strategy {
                    FusionStrategy::WeightedSum => "WeightedSum",
                    FusionStrategy::ReciprocalRankFusion { .. } => "RRF",
                };
                format!("{pad}HybridScan(k={k}, alpha={alpha:.2}, strategy={strat})\n")
            }
            LogicalNode::Filter { predicate, input } => {
                format!(
                    "{pad}Filter({predicate})\n{}",
                    Self::format_node(input, next)
                )
            }
            LogicalNode::Sort {
                by,
                descending,
                input,
            } => {
                let dir = if *descending { "desc" } else { "asc" };
                let by_str = match by {
                    SortKey::Score => "score".to_string(),
                    SortKey::Field(f) => f.clone(),
                };
                format!(
                    "{pad}Sort({by_str} {dir})\n{}",
                    Self::format_node(input, next)
                )
            }
            LogicalNode::Limit { n, input } => {
                format!("{pad}Limit({n})\n{}", Self::format_node(input, next))
            }
            LogicalNode::Project { columns, input } => {
                format!(
                    "{pad}Project([{}])\n{}",
                    columns.join(", "),
                    Self::format_node(input, next)
                )
            }
            LogicalNode::Join {
                left,
                right_collection,
                left_key,
                right_key,
            } => {
                format!(
                    "{pad}Join({right_collection}, {left_key}={right_key})\n{}",
                    Self::format_node(left, next)
                )
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::Storage;
    use crate::types::{CollectionConfig, SearchRequest, VectorRecord};
    use chrono::Utc;
    use serde_json::json;
    use tempfile::tempdir;

    fn make_planner() -> QueryPlanner {
        QueryPlanner::new(CollectionConfig::new("testcol", 4))
    }

    fn base_request() -> SearchRequest {
        SearchRequest {
            vector: Some(vec![0.1_f32, 0.2, 0.3, 0.4]),
            query_text: None,
            k: 10,
            alpha: 0.7,
            collection: "testcol".into(),
            filter: None,
        }
    }

    /// Walk the logical tree and return the scan node type as a static str.
    fn scan_type(node: &LogicalNode) -> &'static str {
        match node {
            LogicalNode::VectorScan { .. } => "VectorScan",
            LogicalNode::SparseScan { .. } => "SparseScan",
            LogicalNode::HybridScan { .. } => "HybridScan",
            LogicalNode::Filter { input, .. } => scan_type(input),
            LogicalNode::Sort { input, .. } => scan_type(input),
            LogicalNode::Limit { input, .. } => scan_type(input),
            LogicalNode::Project { input, .. } => scan_type(input),
            LogicalNode::Join { left, .. } => scan_type(left),
        }
    }

    /// Walk the tree looking for a Filter node.
    fn has_filter_node(node: &LogicalNode) -> bool {
        match node {
            LogicalNode::Filter { .. } => true,
            LogicalNode::Sort { input, .. }
            | LogicalNode::Limit { input, .. }
            | LogicalNode::Project { input, .. } => has_filter_node(input),
            LogicalNode::Join { left, .. } => has_filter_node(left),
            _ => false,
        }
    }

    // ── Test 1 ───────────────────────────────────────────────────
    #[test]
    fn test_plan_dense_only() {
        let planner = make_planner();
        let request = base_request(); // vector=Some, text=None
        let plan = planner.plan_search(&request, 1000).unwrap();

        assert!(!plan.use_hybrid);
        assert_eq!(plan.output_k, 10);
        assert_eq!(plan.index_type, IndexType::HNSW);
        assert!(plan.candidate_k >= 10);
        assert_eq!(scan_type(&plan.root), "VectorScan");
    }

    // ── Test 2 ───────────────────────────────────────────────────
    #[test]
    fn test_plan_sparse_only() {
        let planner = make_planner();
        let request = SearchRequest {
            vector: None,
            query_text: Some("machine learning".into()),
            k: 5,
            ..base_request()
        };
        let plan = planner.plan_search(&request, 1000).unwrap();

        assert!(!plan.use_hybrid);
        assert_eq!(scan_type(&plan.root), "SparseScan");
        assert!(plan.candidate_k >= 5);
    }

    // ── Test 3 ───────────────────────────────────────────────────
    #[test]
    fn test_plan_hybrid() {
        let planner = make_planner();
        let request = SearchRequest {
            vector: Some(vec![0.1_f32, 0.2, 0.3, 0.4]),
            query_text: Some("machine learning".into()),
            k: 10,
            ..base_request()
        };
        let plan = planner.plan_search(&request, 1000).unwrap();

        assert!(plan.use_hybrid);
        assert_eq!(scan_type(&plan.root), "HybridScan");
        assert!(
            plan.candidate_k >= 50,
            "hybrid candidate_k must be >= 50 (10 * 5), got {}",
            plan.candidate_k
        );
    }

    // ── Test 4 ───────────────────────────────────────────────────
    #[test]
    fn test_plan_with_filter() {
        let planner = make_planner();
        let request = SearchRequest {
            filter: Some(json!({"region": "US"})),
            ..base_request()
        };
        let plan = planner.plan_search(&request, 1000).unwrap();

        assert!(plan.filter_predicate.is_some());
        assert!(
            has_filter_node(&plan.root),
            "Filter node must be in the tree"
        );
    }

    // ── Test 5 ───────────────────────────────────────────────────
    #[test]
    fn test_cost_model_small_collection() {
        let planner = make_planner();
        assert!(!planner.should_use_hnsw(50), "50 < 100 → brute force");
        assert!(planner.should_use_hnsw(100), "100 == threshold → HNSW");
        assert!(planner.should_use_hnsw(1_000_000), "large → HNSW");
    }

    // ── Test 6 ───────────────────────────────────────────────────
    #[test]
    fn test_candidate_k_hybrid_with_filter() {
        let planner = make_planner();
        assert_eq!(
            planner.compute_candidate_k(10, true, true),
            100,
            "hybrid+filter → 10 * 10"
        );
        assert_eq!(
            planner.compute_candidate_k(10, true, false),
            50,
            "hybrid only → 10 * 5"
        );
        assert_eq!(
            planner.compute_candidate_k(10, false, false),
            10,
            "neither → 10 * 1"
        );
    }

    // ── Test 7 ───────────────────────────────────────────────────
    #[test]
    fn test_explain_output() {
        let planner = make_planner();
        let plan = PhysicalPlan {
            root: LogicalNode::Limit {
                n: 10,
                input: Box::new(LogicalNode::Sort {
                    by: SortKey::Score,
                    descending: true,
                    input: Box::new(LogicalNode::HybridScan {
                        collection: "testcol".into(),
                        k: 50,
                        alpha: 0.7,
                        strategy: FusionStrategy::WeightedSum,
                        oversample: 5,
                    }),
                }),
            },
            index_type: IndexType::HNSW,
            use_hybrid: true,
            pre_filter: false,
            filter_predicate: None,
            estimated_cost: 12.34,
            candidate_k: 50,
            output_k: 10,
            output_columns: vec![],
        };

        let output = planner.explain(&plan);
        assert!(
            output.contains("HybridScan"),
            "explain must contain 'HybridScan'"
        );
        assert!(
            output.contains("Limit(10)"),
            "explain must contain 'Limit(10)'"
        );
        assert!(
            output.contains("use_hybrid: true"),
            "explain must contain 'use_hybrid: true'"
        );
    }

    // ── Test 8 ───────────────────────────────────────────────────
    #[test]
    fn test_executor_dense_search() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("exdense", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let now = Utc::now();
        for i in 0..10usize {
            storage
                .upsert(VectorRecord {
                    id: format!("v{i}"),
                    vector: vec![i as f32, i as f32 + 1.0, 0.0, 1.0],
                    payload: json!({ "i": i }),
                    text: None,
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let request = SearchRequest {
            vector: Some(vec![4.0_f32, 5.0, 0.0, 1.0]),
            query_text: None,
            k: 5,
            alpha: 0.7,
            collection: "exdense".into(),
            filter: None,
        };

        let results = storage.execute_search(&request).unwrap();
        assert!(
            !results.is_empty(),
            "dense execute_search must return results"
        );
        assert!(results.len() <= 5);
        for r in &results {
            assert!(r.score >= 0.0, "score {} < 0.0", r.score);
        }
        // Sorted descending.
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results not sorted at index {i}"
            );
        }
    }

    // ── Test 9 ───────────────────────────────────────────────────
    #[test]
    fn test_executor_hybrid_search() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("exhybrid", 4);
        let mut storage = Storage::create(dir.path(), &config).unwrap();

        let texts = [
            "machine learning classification",
            "deep neural network training",
            "random forest ensemble method",
            "gradient boosting decision tree",
            "support vector machine kernel",
            "convolutional network image recognition",
            "recurrent network sequence modeling",
            "attention transformer language model",
            "unsupervised clustering algorithm",
            "reinforcement learning reward signal",
        ];
        let now = Utc::now();
        for (i, text) in texts.iter().enumerate() {
            storage
                .upsert(VectorRecord {
                    id: format!("h{i}"),
                    vector: vec![i as f32, i as f32 + 1.0, 0.5, 1.0],
                    payload: json!({ "i": i }),
                    text: Some(text.to_string()),
                    created_at: now,
                    updated_at: now,
                })
                .unwrap();
        }

        let request = SearchRequest {
            vector: Some(vec![3.0_f32, 4.0, 0.5, 1.0]),
            query_text: Some("machine learning".into()),
            k: 5,
            alpha: 0.7,
            collection: "exhybrid".into(),
            filter: None,
        };

        let results = storage.execute_search(&request).unwrap();
        assert!(!results.is_empty());
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "results not sorted at index {i}"
            );
        }
    }

    // ── Test 10 ──────────────────────────────────────────────────
    #[test]
    fn test_executor_invalid_no_inputs() {
        let dir = tempdir().unwrap();
        let config = CollectionConfig::new("exinvalid", 4);
        let storage = Storage::create(dir.path(), &config).unwrap();

        let request = SearchRequest {
            vector: None,
            query_text: None,
            k: 5,
            alpha: 0.7,
            collection: "exinvalid".into(),
            filter: None,
        };

        let result = storage.execute_search(&request);
        assert!(
            matches!(result, Err(VecDbError::InvalidQuery(_))),
            "expected InvalidQuery error"
        );
    }

    // ── Test 11 ──────────────────────────────────────────────────
    #[test]
    fn test_pre_filter_threshold() {
        let planner = make_planner();

        assert!(
            planner.should_pre_filter(0.05, 50_000),
            "selectivity 0.05 < 0.1 → pre-filter"
        );
        assert!(
            !planner.should_pre_filter(0.5, 50_000),
            "selectivity 0.5 >= 0.1 and count <= 100k → no pre-filter"
        );
        assert!(
            planner.should_pre_filter(0.5, 200_000),
            "count > 100k → always pre-filter"
        );
    }
}
