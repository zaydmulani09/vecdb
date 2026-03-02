use crate::errors::{Result, VecDbError};
use crate::planner::plan::{PhysicalPlan, QueryPlanner};
use crate::types::{SearchRequest, Vector};

use super::ast::{Condition, Literal, Operator, ScalarCondition, SelectStatement};

// ─────────────────────────────────────────────────────────────────
// AstConverter
// ─────────────────────────────────────────────────────────────────

/// Converts a parsed [`SelectStatement`] into a [`PhysicalPlan`] by delegating
/// to `QueryPlanner::plan_search`.
pub struct AstConverter {
    pub planner: QueryPlanner,
}

impl AstConverter {
    pub fn new(planner: QueryPlanner) -> Self {
        Self { planner }
    }

    /// Convert a statement into a physical plan.
    ///
    /// - Extracts the first `VECTOR_SIM` condition as the query vector.
    /// - Collects ALL AND-connected scalar conditions into structured filter JSON.
    /// - Sets `output_columns` from the SELECT column list.
    /// - Falls back to `k = 10` when no LIMIT clause is present.
    pub fn convert(&self, stmt: SelectStatement, vector_count: usize) -> Result<PhysicalPlan> {
        // Capture projection columns before stmt fields are moved.
        let output_columns: Vec<String> = stmt.columns.clone();

        // Extract query vector from the WHERE clause.
        let query_vector: Option<Vector> = stmt
            .where_clause
            .as_ref()
            .and_then(find_vector_condition)
            .map(|vc| vc.vector.clone());

        if query_vector.is_none() {
            return Err(VecDbError::InvalidQuery(
                "SQL query must contain a VECTOR_SIM condition".into(),
            ));
        }

        // Build structured scalar filter from the WHERE clause (all operators).
        let scalar_filter: Option<serde_json::Value> =
            stmt.where_clause.as_ref().and_then(collect_all_conditions);

        let k = stmt.limit.unwrap_or(10);

        let request = SearchRequest {
            vector: query_vector,
            query_text: None,
            k,
            alpha: 0.7,
            collection: stmt.table,
            filter: scalar_filter,
        };

        let mut plan = self.planner.plan_search(&request, vector_count)?;

        // Propagate SELECT column list into the plan.
        // Empty list means SELECT * — keep output_columns empty (no projection).
        plan.output_columns = output_columns;

        Ok(plan)
    }

    /// Extract the query vector from a `SelectStatement`'s WHERE clause.
    pub fn extract_query_vector(stmt: &SelectStatement) -> Option<Vector> {
        stmt.where_clause
            .as_ref()
            .and_then(find_vector_condition)
            .map(|vc| vc.vector.clone())
    }
}

// ─────────────────────────────────────────────────────────────────
// Private helpers
// ─────────────────────────────────────────────────────────────────

/// Recursively find the first `VECTOR_SIM` condition in the tree.
pub fn find_vector_condition(cond: &Condition) -> Option<&super::ast::VectorCondition> {
    match cond {
        Condition::Vector(vc) => Some(vc),
        Condition::And(l, r) | Condition::Or(l, r) => {
            find_vector_condition(l).or_else(|| find_vector_condition(r))
        }
        Condition::Not(inner) => find_vector_condition(inner),
        Condition::Scalar(_) => None,
    }
}

/// Walk the condition tree and collect ALL AND-reachable scalar conditions into
/// a `serde_json::Value`.
///
/// - Single condition → `{"field":"…","op":"…","value":…}`
/// - Multiple AND conditions → `[cond1, cond2, …]`
/// - `OR` branches and `NOT` are skipped (never over-filter).
fn collect_all_conditions(cond: &Condition) -> Option<serde_json::Value> {
    let mut items: Vec<serde_json::Value> = Vec::new();
    collect_scalar_into(cond, &mut items);
    match items.len() {
        0 => None,
        1 => Some(items.remove(0)),
        _ => Some(serde_json::Value::Array(items)),
    }
}

fn collect_scalar_into(cond: &Condition, out: &mut Vec<serde_json::Value>) {
    match cond {
        Condition::Scalar(sc) => {
            if let Some(obj) = scalar_to_json(sc) {
                out.push(obj);
            }
        }
        Condition::And(l, r) => {
            collect_scalar_into(l, out);
            collect_scalar_into(r, out);
        }
        // Vector, Or, Not → skip
        _ => {}
    }
}

fn scalar_to_json(sc: &ScalarCondition) -> Option<serde_json::Value> {
    let op_str = match sc.op {
        Operator::Eq => "=",
        Operator::NotEq => "!=",
        Operator::Lt => "<",
        Operator::LtEq => "<=",
        Operator::Gt => ">",
        Operator::GtEq => ">=",
        Operator::Like => "LIKE",
    };
    Some(serde_json::json!({
        "field": sc.field,
        "op": op_str,
        "value": literal_to_json(&sc.value),
    }))
}

fn literal_to_json(lit: &Literal) -> serde_json::Value {
    match lit {
        Literal::String(s) => serde_json::Value::String(s.clone()),
        Literal::Float(f) => serde_json::json!(f),
        Literal::Integer(i) => serde_json::json!(i),
        Literal::Bool(b) => serde_json::Value::Bool(*b),
    }
}
