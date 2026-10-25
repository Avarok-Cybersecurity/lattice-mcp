//! Errors raised by the pure model. The repo has no coded-error registry;
//! this follows lattice-mcp's `thiserror` style, and the host maps each
//! variant onto the `McpMemoryError` it raised before the extraction (same
//! variant, same message).

use crate::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryCoreError {
    /// A persisted point lacks a field the record needs.
    #[error("missing payload field: {key}")]
    MissingPayloadField { key: String },

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// `slot_contradiction::settle` was given a winner outside the pair.
    #[error("settlement winner {winner} is not in the conflict ({a}, {b})")]
    SettlementWinnerNotInConflict { winner: u64, a: u64, b: u64 },
}

pub type Result<T> = core::result::Result<T, MemoryCoreError>;
