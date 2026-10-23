// Accepted project-wide: existing test setups build `ImportanceState` etc.
// via `Default::default()` then assign fields for readability. This is a
// pre-existing, deliberate style choice — not worth churning many test
// modules to satisfy a pedantic style lint.
#![allow(clippy::field_reassign_with_default)]

pub mod config;
pub mod embedding;
pub mod error;
pub mod memory;
pub mod reranker;
pub mod store;
pub mod tools;
