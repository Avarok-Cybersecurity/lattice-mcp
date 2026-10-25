//! Shared value types for the optional LLM-augmented `ChatProvider` methods.
//!
//! Each is returned wrapped in `Option` by the trait's default-`None`
//! methods. The contract (PCND):
//! - `Ok(None)`  → provider declined; caller uses the deterministic fallback.
//! - `Ok(Some)`  → provider produced a result.
//! - `Err(_)`    → provider failed; error propagates (no silent fallback).

use crate::prelude::*;

/// Logical relationship of a *candidate* statement to an *existing* one.
/// Used by write-time contradiction handling (Step 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntailmentVerdict {
    /// The candidate asserts something mutually exclusive with the existing memory.
    Contradicts,
    /// The candidate restates / is entailed by the existing memory.
    Entails,
    /// No strong logical relationship.
    Neutral,
}

/// Structured distillation of a memory cluster (Step 7).
/// `provenance` holds the source memory IDs the distillation was derived from.
#[derive(Debug, Clone)]
pub struct DistilledKnowledge {
    /// The essential consolidated knowledge statement.
    pub core: String,
    /// Supporting background that disambiguates `core`.
    pub context: String,
    /// Short topic/theme tag.
    pub theme: String,
    /// Source memory IDs this knowledge was distilled from.
    pub provenance: Vec<u64>,
}

/// A transformed retrieval query (Step 10). `rewritten` is the text that
/// will actually be embedded for the vector search.
#[derive(Debug, Clone)]
pub struct QueryTransform {
    pub rewritten: String,
}
