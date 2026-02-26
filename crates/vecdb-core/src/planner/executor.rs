use std::collections::HashSet;

use crate::errors::{Result, VecDbError};
use crate::hybrid::FusionStrategy;
use crate::storage::Storage;
use crate::types::{SearchResult, Vector};

use super::plan::{LogicalNode, PhysicalPlan};

// ─────────────────────────────────────────────────────────────────
// PlanExecutor
// ─────────────────────────────────────────────────────────────────

pub struct PlanExecutor<'a> {
    storage: &'a Storage,
}

impl<'a> PlanExecutor<'a> {
    pub fn new(storage: &'a Storage) -> Self {
        Self { storage }
    }

    /// Execute a physical plan against the storage layer.
    ///
    /// Steps:
    /// 0. Pre-filter: if `plan.pre_filter`, fetch matching ids from metadata.
    /// 1. Run the appropriate scan (hybrid / dense / sparse).
    /// 2. Apply allowed-id filter (pre-filter narrowing) and full predicate filter.
    /// 3. Re-sort by score descending.
    /// 4. Limit to `plan.output_k`.
    /// 5. Apply column projection.
    pub fn execute(
        &self,
        plan: &PhysicalPlan,
        query_vector: Option<&Vector>,
        query_text: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        // ── Stage 0: Pre-filter (allowed id set) ─────────────────
        let allowed_ids: Option<HashSet<String>> = if plan.pre_filter {
            plan.filter_predicate.as_deref().and_then(|pred| {
                match serde_json::from_str::<serde_json::Value>(pred) {
                    Ok(fv) => match self.storage.metadata.filter_ids(&fv) {
                        Ok(ids) => Some(ids.into_iter().collect()),
                        Err(e) => {
                            tracing::warn!("pre-filter failed ({e}), falling back to post-filter");
                            None
                        }
                    },
                    Err(e) => {
                        tracing::warn!("pre-filter: could not parse predicate ({e}), skipping");
                        None
                    }
                }
            })
        } else {
            None
        };

        // ── Stage 1: Scan ─────────────────────────────────────────
        let mut results: Vec<SearchResult> = if plan.use_hybrid
            || matches!(
                find_scan_node(&plan.root),
                Some(LogicalNode::HybridScan { .. })
            ) {
            self.storage.search_hybrid(
                query_vector,
                query_text,
                plan.candidate_k,
                extract_alpha(plan),
                extract_strategy(plan),
            )?
        } else {
            let scan = find_scan_node(&plan.root);
            let is_sparse = matches!(scan, Some(LogicalNode::SparseScan { .. }));

            if is_sparse {
                let qt = query_text.ok_or_else(|| {
                    VecDbError::InvalidQuery("sparse search requires query_text".into())
                })?;
                self.storage.search_sparse(qt, plan.candidate_k)?
            } else {
                let qv = query_vector.ok_or_else(|| {
                    VecDbError::InvalidQuery("dense search requires query_vector".into())
                })?;
                self.storage.search_dense(qv, plan.candidate_k)?
            }
        };

        // ── Stage 2a: Narrow to pre-filtered allowed ids ──────────
        if let Some(ref ids) = allowed_ids {
            results.retain(|r| ids.contains(&r.id));
        }

        // ── Stage 2b: Post-scan predicate filter ──────────────────
        if let Some(ref pred) = plan.filter_predicate {
            match serde_json::from_str::<serde_json::Value>(pred) {
                Ok(fv) => results.retain(|r| apply_json_filter(r, &fv)),
                Err(e) => {
                    tracing::warn!(
                        "could not parse filter predicate as JSON ({e}), skipping filter"
                    );
                }
            }
        }

        // ── Stage 3: Sort ─────────────────────────────────────────
        results.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── Stage 4: Limit ────────────────────────────────────────
        results.truncate(plan.output_k);

        // ── Stage 5: Projection ───────────────────────────────────
        results = apply_projection(results, &plan.output_columns);

        Ok(results)
    }
}

// ─────────────────────────────────────────────────────────────────
// Private helpers — scan-tree navigation
// ─────────────────────────────────────────────────────────────────

fn find_scan_node(node: &LogicalNode) -> Option<&LogicalNode> {
    match node {
        LogicalNode::VectorScan { .. }
        | LogicalNode::SparseScan { .. }
        | LogicalNode::HybridScan { .. } => Some(node),
        LogicalNode::Filter { input, .. }
        | LogicalNode::Sort { input, .. }
        | LogicalNode::Limit { input, .. }
        | LogicalNode::Project { input, .. } => find_scan_node(input),
        LogicalNode::Join { left, .. } => find_scan_node(left),
    }
}

fn extract_alpha(plan: &PhysicalPlan) -> f32 {
    fn from_node(node: &LogicalNode) -> Option<f32> {
        match node {
            LogicalNode::HybridScan { alpha, .. } | LogicalNode::VectorScan { alpha, .. } => {
                Some(*alpha)
            }
            LogicalNode::Filter { input, .. }
            | LogicalNode::Sort { input, .. }
            | LogicalNode::Limit { input, .. }
            | LogicalNode::Project { input, .. } => from_node(input),
            LogicalNode::Join { left, .. } => from_node(left),
            LogicalNode::SparseScan { .. } => None,
        }
    }
    from_node(&plan.root).unwrap_or(0.7)
}

fn extract_strategy(plan: &PhysicalPlan) -> FusionStrategy {
    fn from_node(node: &LogicalNode) -> Option<FusionStrategy> {
        match node {
            LogicalNode::HybridScan { strategy, .. } | LogicalNode::VectorScan { strategy, .. } => {
                Some(strategy.clone())
            }
            LogicalNode::Filter { input, .. }
            | LogicalNode::Sort { input, .. }
            | LogicalNode::Limit { input, .. }
            | LogicalNode::Project { input, .. } => from_node(input),
            LogicalNode::Join { left, .. } => from_node(left),
            LogicalNode::SparseScan { .. } => None,
        }
    }
    from_node(&plan.root).unwrap_or_default()
}

// ─────────────────────────────────────────────────────────────────
// Filter evaluation
// ─────────────────────────────────────────────────────────────────

/// Evaluate `filter` against a single `SearchResult`.
///
/// Supported filter shapes:
/// - **Structured condition**: `{"field":"f","op":"=","value":v}` — single predicate.
/// - **Array of conditions**: `[cond, ...]` — AND of all conditions.
/// - **Legacy flat map**: `{"key": value}` — equality for every key (HTTP API compat).
pub(crate) fn apply_json_filter(record: &SearchResult, filter: &serde_json::Value) -> bool {
    match filter {
        serde_json::Value::Array(conditions) => {
            conditions.iter().all(|c| apply_json_filter(record, c))
        }
        serde_json::Value::Object(obj) => {
            if obj.contains_key("field") && obj.contains_key("op") && obj.contains_key("value") {
                eval_condition(record, obj)
            } else {
                // Legacy flat-map equality (HTTP API SearchRequest.filter).
                obj.iter().all(|(k, v)| {
                    resolve_field(record, k)
                        .map(|rv| eq_values(&rv, v))
                        .unwrap_or(false)
                })
            }
        }
        _ => false,
    }
}

fn eval_condition(record: &SearchResult, obj: &serde_json::Map<String, serde_json::Value>) -> bool {
    let Some(field) = obj.get("field").and_then(|v| v.as_str()) else {
        return false;
    };
    let Some(op) = obj.get("op").and_then(|v| v.as_str()) else {
        return false;
    };
    let Some(filter_val) = obj.get("value") else {
        return false;
    };
    let Some(record_val) = resolve_field(record, field) else {
        return false;
    };
    eval_op(&record_val, op, filter_val)
}

/// Resolve a field path from a `SearchResult`.
///
/// - `"id"` → `record.id`
/// - `"a.b.c"` → navigate `record.payload` via nested object keys.
fn resolve_field(record: &SearchResult, field: &str) -> Option<serde_json::Value> {
    if field == "id" {
        return Some(serde_json::Value::String(record.id.clone()));
    }
    let mut current = &record.payload;
    for part in field.split('.') {
        current = current.get(part)?;
    }
    Some(current.clone())
}

fn eval_op(rv: &serde_json::Value, op: &str, fv: &serde_json::Value) -> bool {
    match op {
        "=" => eq_values(rv, fv),
        "!=" => !eq_values(rv, fv),
        "<" | "<=" | ">" | ">=" => {
            let Some(r) = to_f64(rv) else {
                return false;
            };
            let Some(f) = to_f64(fv) else {
                return false;
            };
            match op {
                "<" => r < f,
                "<=" => r <= f,
                ">" => r > f,
                _ => r >= f,
            }
        }
        "LIKE" => {
            let (Some(s), Some(p)) = (rv.as_str(), fv.as_str()) else {
                return false;
            };
            like_match(s.as_bytes(), p.as_bytes())
        }
        _ => false,
    }
}

fn eq_values(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::String(s1), serde_json::Value::String(s2)) => s1 == s2,
        (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) => b1 == b2,
        _ => match (to_f64(a), to_f64(b)) {
            (Some(fa), Some(fb)) => fa == fb,
            _ => a == b,
        },
    }
}

fn to_f64(v: &serde_json::Value) -> Option<f64> {
    match v {
        serde_json::Value::Number(n) => n.as_f64(),
        serde_json::Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

/// `%` wildcard matcher (case-sensitive, no regex).
fn like_match(text: &[u8], pattern: &[u8]) -> bool {
    match pattern.first() {
        None => text.is_empty(),
        Some(b'%') => (0..=text.len()).any(|i| like_match(&text[i..], &pattern[1..])),
        Some(&pc) => text
            .first()
            .map(|&tc| tc == pc && like_match(&text[1..], &pattern[1..]))
            .unwrap_or(false),
    }
}

// ─────────────────────────────────────────────────────────────────
// Projection
// ─────────────────────────────────────────────────────────────────

/// Keep only the columns listed in `columns` from each result's payload.
///
/// - Empty `columns` or `["*"]` → return results unchanged (SELECT *).
/// - `"id"`, `"score"`, `"similarity"` → top-level fields always kept.
/// - `"text"` → kept only when listed; otherwise set to `None`.
/// - Any other name → kept from `payload` if present.
fn apply_projection(results: Vec<SearchResult>, columns: &[String]) -> Vec<SearchResult> {
    if columns.is_empty() || columns.iter().any(|c| c == "*") {
        return results;
    }
    let keep_text = columns.iter().any(|c| c == "text");
    results
        .into_iter()
        .map(|mut r| {
            if let serde_json::Value::Object(ref orig) = r.payload.clone() {
                let mut new_payload = serde_json::Map::new();
                for col in columns {
                    match col.as_str() {
                        "id" | "score" | "similarity" | "text" => {}
                        key => {
                            if let Some(v) = orig.get(key) {
                                new_payload.insert(key.to_string(), v.clone());
                            }
                        }
                    }
                }
                r.payload = serde_json::Value::Object(new_payload);
            }
            if !keep_text {
                r.text = None;
            }
            r
        })
        .collect()
}
