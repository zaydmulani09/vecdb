//! chroma adapter — container-backed. Implemented in the Phase 4 container pass;
//! returns `None` (skipped) until then so the vecdb row runs standalone.

use crate::{Bench, Row};

pub fn run(_b: &Bench) -> Option<Result<Row, String>> {
    eprintln!("  (chroma) not yet wired — start the compose stack, then implement this adapter");
    None
}
