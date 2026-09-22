pub mod config;
pub mod db;
pub mod errors;
pub mod hybrid;
pub mod index;
pub mod planner;
pub mod sparse;
pub mod storage;
pub mod types;

// Embedded API — the primary, no-server entry point.
pub use db::{Collection, Db};
pub use errors::{Result, VecDbError};
pub use hybrid::{min_max_normalize, softmax_normalize, FusionStrategy, HybridEngine};
pub use planner::sql::{
    AstConverter, Condition, Literal, Operator as SqlOperator, OrderBy, ScalarCondition,
    SelectStatement, SqlParser, VectorCondition,
};
pub use planner::{LogicalNode, PhysicalPlan, PlanExecutor, QueryPlanner, SortKey};
pub use types::*;
