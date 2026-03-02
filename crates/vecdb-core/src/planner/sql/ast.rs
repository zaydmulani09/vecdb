use crate::types::Vector;

// ─────────────────────────────────────────────────────────────────
// Top-level statement
// ─────────────────────────────────────────────────────────────────

/// A parsed `SELECT` statement.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectStatement {
    /// The table / collection name (FROM clause).
    pub table: String,
    /// Projected columns.  Empty means `SELECT *`.
    pub columns: Vec<String>,
    /// Optional WHERE clause.
    pub where_clause: Option<Condition>,
    /// Optional ORDER BY clause.
    pub order_by: Option<OrderBy>,
    /// Optional LIMIT clause.
    pub limit: Option<usize>,
}

// ─────────────────────────────────────────────────────────────────
// Condition tree
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// `VECTOR_SIM(col, [...]) op threshold`
    Vector(VectorCondition),
    /// `field op literal`
    Scalar(ScalarCondition),
    /// `left AND right`
    And(Box<Condition>, Box<Condition>),
    /// `left OR right`
    Or(Box<Condition>, Box<Condition>),
    /// `NOT condition`
    Not(Box<Condition>),
}

// ─────────────────────────────────────────────────────────────────
// VectorCondition
// ─────────────────────────────────────────────────────────────────

/// `VECTOR_SIM(column, [f32, ...]) op threshold`
#[derive(Debug, Clone, PartialEq)]
pub struct VectorCondition {
    /// The vector column name (first argument to VECTOR_SIM).
    pub column: String,
    /// The query vector (second argument).
    pub vector: Vector,
    /// Comparison operator — typically `Gt` or `GtEq`.
    pub op: Operator,
    /// Similarity threshold.
    pub threshold: f32,
}

// ─────────────────────────────────────────────────────────────────
// ScalarCondition
// ─────────────────────────────────────────────────────────────────

/// `field op literal`
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarCondition {
    /// Field name (may include dot / arrow path like `meta->>'key'`).
    pub field: String,
    pub op: Operator,
    pub value: Literal,
}

// ─────────────────────────────────────────────────────────────────
// Operator
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Like,
}

// ─────────────────────────────────────────────────────────────────
// Literal
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    String(String),
    Float(f64),
    Integer(i64),
    Bool(bool),
}

// ─────────────────────────────────────────────────────────────────
// OrderBy
// ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct OrderBy {
    pub field: String,
    pub descending: bool,
}
